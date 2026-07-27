"""Re-evaluate saved NFVI rebuttal checkpoints with the finalization protocol.

This script intentionally mirrors the metric order and random-number handling in
``finalization.runner_eval.evaluate_one_run``:

1. seed Python, NumPy, and Torch before constructing the runner;
2. load the saved variational checkpoint;
3. evaluate ELBO first;
4. evaluate W2 second with the official baseline subsampling helper.

NFVI has an exact marginal density, so the finalization ELBO batch size and
number of marginal-density batches do not affect its ELBO.  They are retained
in the metadata so the protocol is directly comparable with the semi-implicit
methods.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from finalization.runner_eval import (  # noqa: E402
    evaluate_w2_budgeted,
    remove_file_handlers,
    set_seed,
)
from runner.runners import Runners  # noqa: E402


DEFAULT_METHODS = ("NFVI-4", "NFVI-8")
RUN_FIELDS = (
    "method",
    "seed",
    "flow_layers",
    "checkpoint_path",
    "elbo",
    "elbo_std",
    "elbo_ci_half",
    "elbo_runtime_sec",
    "w2",
    "w2_runtime_sec",
)


@dataclass(frozen=True)
class CheckpointRecord:
    method: str
    seed: int
    flow_layers: int
    run_dir: Path

    @property
    def checkpoint_path(self) -> Path:
        return self.run_dir / "final_vi_model.pt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Re-evaluate NFVI-4/NFVI-8 checkpoints using the official "
            "default_config_grid finalization metric protocol."
        )
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("analysis/nfvi_rebuttal_20260726"),
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=DEFAULT_METHODS,
        default=DEFAULT_METHODS,
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(42, 43, 44, 45, 46),
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument("--elbo-samples", type=int, default=5000)
    parser.add_argument("--elbo-batch-size", type=int, default=2048)
    parser.add_argument("--elbo-batches", type=int, default=20)
    parser.add_argument("--w2-samples", type=int, default=10000)
    parser.add_argument("--w2-projections", type=int, default=5000)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute method/seed pairs already present in the output CSV.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the selected checkpoints without loading or evaluating them.",
    )
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device=cuda requested, but CUDA is unavailable")
        return torch.device("cuda")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint_records(
    metrics_path: Path,
    methods: set[str],
    seeds: set[int],
) -> list[CheckpointRecord]:
    with metrics_path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    records: list[CheckpointRecord] = []
    for row in rows:
        method = str(row["method"])
        seed = int(row["seed"])
        if method not in methods or seed not in seeds:
            continue
        flow_layers_text = row.get("flow_layers") or method.rsplit("-", 1)[1]
        records.append(
            CheckpointRecord(
                method=method,
                seed=seed,
                flow_layers=int(flow_layers_text),
                run_dir=Path(row["run_dir"]),
            )
        )

    records.sort(key=lambda record: (record.flow_layers, record.seed))
    expected = {(method, seed) for method in methods for seed in seeds}
    observed = {(record.method, record.seed) for record in records}
    missing = sorted(expected - observed)
    if missing:
        raise ValueError(f"Missing run records for: {missing}")
    return records


def build_runner(
    record: CheckpointRecord,
    device: torch.device,
    runtime_root: Path,
) -> Any:
    config_path = PROJECT_ROOT / "configs" / "nfvi_8_gaussians.yaml"
    config = OmegaConf.load(config_path)
    config.config_path = str(config_path)
    config.seed = record.seed
    config.device = str(device)
    config.use_cuda = device.type == "cuda"
    config.cuda_visible_devices = "0"
    config.train.checkpoint.enabled = False
    config.setdefault("vi_model", {})
    config.vi_model.num_flow_layers = record.flow_layers
    run_slug = f"{record.method.lower()}_seed{record.seed}"
    config.output = {
        "results_dir": str(runtime_root / "results" / run_slug),
        "tb_dir": str(runtime_root / "tb_logs" / run_slug),
    }

    # Official finalization seeds before runner construction.  Constructing the
    # network advances the RNG stream before ELBO and W2 are evaluated.
    set_seed(record.seed, device.type == "cuda")
    runner = Runners[str(config.runner_type)](config=config)
    runner.writer.close()
    remove_file_handlers()

    state = torch.load(
        record.checkpoint_path,
        map_location=device,
        weights_only=True,
    )
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()
    return runner


def evaluate_record(
    record: CheckpointRecord,
    args: argparse.Namespace,
    device: torch.device,
    runtime_root: Path,
) -> dict[str, Any]:
    runner = build_runner(record, device, runtime_root)
    runner.n_elbo_z_samples = int(args.elbo_samples)
    runner.n_elbo_batch_size = int(args.elbo_batch_size)
    runner.n_elbo_batches = int(args.elbo_batches)
    runner.n_w2_samples = int(args.w2_samples)
    runner.n_w2_projections = int(args.w2_projections)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    elbo, elbo_std, _elbo_std_q, elbo_ci_half = runner.evaluate_elbo()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elbo_runtime = time.perf_counter() - started

    w2_cfg = OmegaConf.create(
        {
            "num_samples": int(args.w2_samples),
            "num_projections": int(args.w2_projections),
        }
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    w2 = evaluate_w2_budgeted(runner, "8_gaussians", w2_cfg)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    w2_runtime = time.perf_counter() - started

    result = {
        "method": record.method,
        "seed": record.seed,
        "flow_layers": record.flow_layers,
        "checkpoint_path": str(record.checkpoint_path),
        "elbo": float(elbo),
        "elbo_std": float(elbo_std),
        "elbo_ci_half": float(elbo_ci_half),
        "elbo_runtime_sec": float(elbo_runtime),
        "w2": float(w2),
        "w2_runtime_sec": float(w2_runtime),
    }
    del runner
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def read_existing(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(
    path: Path,
    rows: list[dict[str, Any]],
    fields: tuple[str, ...] | list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for method in sorted({str(row["method"]) for row in rows}):
        group = [row for row in rows if str(row["method"]) == method]
        if not group:
            continue
        summary: dict[str, Any] = {
            "method": method,
            "n_seeds": len(group),
        }
        for metric in ("elbo", "w2"):
            values = [float(row[metric]) for row in group]
            mean = statistics.mean(values)
            sd = statistics.stdev(values) if len(values) > 1 else 0.0
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_se"] = sd / math.sqrt(len(values))
        summaries.append(summary)
    return summaries


def main() -> None:
    args = parse_args()
    report_dir = (PROJECT_ROOT / args.report_dir).resolve()
    metrics_path = report_dir / "run_metrics.csv"
    output_runs = report_dir / "official_reevaluation_runs.csv"
    output_summary = report_dir / "official_reevaluation_summary.csv"
    output_metadata = report_dir / "official_reevaluation_metadata.json"
    runtime_root = PROJECT_ROOT / "results" / "nfvi_rebuttal_official_reevaluation"

    records = load_checkpoint_records(
        metrics_path,
        methods=set(args.methods),
        seeds=set(args.seeds),
    )
    if args.dry_run:
        for record in records:
            status = "present" if record.checkpoint_path.is_file() else "missing"
            print(
                f"{record.method} seed={record.seed}: "
                f"{record.checkpoint_path} [{status}]"
            )
        return

    missing_files = [
        str(record.checkpoint_path)
        for record in records
        if not record.checkpoint_path.is_file()
    ]
    if missing_files:
        raise FileNotFoundError(
            "Checkpoint files are missing:\n" + "\n".join(missing_files)
        )

    device = resolve_device(args.device)
    existing = [] if args.overwrite else read_existing(output_runs)
    by_key = {
        (str(row["method"]), int(row["seed"])): row
        for row in existing
        if str(row["method"]) in set(args.methods)
        and int(row["seed"]) in set(args.seeds)
    }

    for record in records:
        key = (record.method, record.seed)
        if key in by_key:
            print(f"Skipping completed {record.method} seed={record.seed}")
            continue
        print(f"Evaluating {record.method} seed={record.seed} on {device}...")
        result = evaluate_record(record, args, device, runtime_root)
        by_key[key] = result
        selected_rows = [
            by_key[item_key]
            for item_key in sorted(by_key, key=lambda item: (item[0], item[1]))
        ]
        write_csv(output_runs, selected_rows, RUN_FIELDS)
        print(
            f"  ELBO={result['elbo']:.9f}, W2={result['w2']:.9f}, "
            f"eval_time={result['elbo_runtime_sec'] + result['w2_runtime_sec']:.2f}s"
        )

    final_rows = [
        by_key[key]
        for key in sorted(by_key, key=lambda item: (item[0], item[1]))
    ]
    summary_rows = aggregate(final_rows)
    summary_fields = (
        "method",
        "n_seeds",
        "elbo_mean",
        "elbo_sd",
        "elbo_se",
        "w2_mean",
        "w2_sd",
        "w2_se",
    )
    write_csv(output_runs, final_rows, RUN_FIELDS)
    write_csv(output_summary, summary_rows, summary_fields)

    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else "CPU"
        ),
        "metric_order": ["elbo", "w2"],
        "seed_timing": "before runner construction, matching official finalization",
        "elbo": {
            "num_z_samples": int(args.elbo_samples),
            "marginal_density": "exact normalizing-flow log density",
            "official_semi_implicit_batch_size": int(args.elbo_batch_size),
            "official_semi_implicit_num_batches": int(args.elbo_batches),
        },
        "w2": {
            "num_samples": int(args.w2_samples),
            "num_projections": int(args.w2_projections),
            "baseline_sampling": "official evaluate_w2_budgeted",
        },
        "methods": list(args.methods),
        "seeds": list(args.seeds),
    }
    output_metadata.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )

    print("\nSummary (mean ± SE):")
    for row in summary_rows:
        print(
            f"{row['method']}: ELBO={row['elbo_mean']:.6f} "
            f"± {row['elbo_se']:.6f}; W2={row['w2_mean']:.6f} "
            f"± {row['w2_se']:.6f}"
        )


if __name__ == "__main__":
    main()
