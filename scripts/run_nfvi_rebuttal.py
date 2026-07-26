"""Matched DIVI-versus-NFVI benchmark for the 8-Gaussians rebuttal.

The script deliberately disables intermediate metrics, plots, samples, and
checkpoints while timing training.  It then evaluates every fitted model with
the same ELBO and sliced-W2 settings and writes compact, git-trackable reports
separately from the ignored run artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from runner.runners import Runners  # noqa: E402


@dataclass(frozen=True)
class Variant:
    label: str
    config_path: Path
    flow_layers: int | None = None


NUMERIC_SUMMARY_FIELDS = (
    "elbo",
    "w2",
    "training_time_sec",
    "modes_covered_1pct",
    "near_mode_mass",
    "mode_entropy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a matched DIVI/NFVI 8-Gaussians benchmark."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("DIVI", "NFVI"),
        default=("DIVI", "NFVI"),
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=(42, 43, 44, 45, 46),
    )
    parser.add_argument(
        "--nf-layers",
        nargs="+",
        type=int,
        default=(4, 8, 16),
        help="RealNVP coupling-layer counts to benchmark.",
    )
    parser.add_argument("--epochs", type=int, default=10000)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--w2-samples", type=int, default=10000)
    parser.add_argument("--w2-projections", type=int, default=1000)
    parser.add_argument("--elbo-samples", type=int, default=5000)
    parser.add_argument("--mode-samples", type=int, default=10000)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/nfvi_rebuttal_8_gaussians"),
        help="Ignored directory for full run artifacts.",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=Path("analysis/nfvi_rebuttal_20260726"),
        help="Directory for compact CSV/JSON report artifacts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rerun method/seed pairs already present in run_metrics.csv.",
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


def variants_from_args(args: argparse.Namespace) -> list[Variant]:
    variants: list[Variant] = []
    if "DIVI" in args.methods:
        variants.append(
            Variant(
                label="DIVI",
                config_path=PROJECT_ROOT / "configs/dsivi_8_gaussians.yaml",
            )
        )
    if "NFVI" in args.methods:
        variants.extend(
            Variant(
                label=f"NFVI-{layers}",
                config_path=PROJECT_ROOT / "configs/nfvi_8_gaussians.yaml",
                flow_layers=layers,
            )
            for layers in args.nf_layers
        )
    return variants


def configure_run(
    variant: Variant,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> DictConfig:
    config = OmegaConf.load(variant.config_path)
    config.config_path = str(variant.config_path)
    config.seed = seed
    config.device = str(device)
    config.use_cuda = device.type == "cuda"
    config.cuda_visible_devices = "0"

    config.train.epochs = args.epochs
    # Keep instrumentation out of the timed section.
    config.train.log.metric_log_freq = args.epochs + 1
    config.train.log.loss_log_freq = args.epochs + 1
    config.train.checkpoint.enabled = False
    config.train.sample.freq = args.epochs + 1
    config.train.plot.freq = args.epochs + 1

    config.metric.kl_ite.enabled = False
    config.metric.w2.enabled = True
    config.metric.w2.num_samples = args.w2_samples
    config.metric.w2.num_projections = args.w2_projections
    config.metric.elbo.enabled = True
    config.metric.elbo.num_z_samples = args.elbo_samples
    config.metric.expected_log_marginal.enabled = False
    config.metric.fisher.enabled = False
    config.metric.ksd.enabled = False
    config.metric.mmd.enabled = False
    config.metric.bnn.enabled = False

    run_root = (PROJECT_ROOT / args.output_dir).resolve()
    config.setdefault("output", {})
    config.output.results_dir = str(run_root / "results")
    config.output.tb_dir = str(run_root / "tb_logs")

    if variant.flow_layers is not None:
        config.setdefault("vi_model", {})
        config.vi_model.num_flow_layers = variant.flow_layers
    return config


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def set_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def parameter_counts(runner: Any) -> tuple[int, int, int]:
    vi_parameters = sum(
        parameter.numel()
        for parameter in runner.vi_model.parameters()
        if parameter.requires_grad
    )
    auxiliary_parameters = 0
    reverse_model = getattr(runner, "reverse_model", None)
    if reverse_model is not None:
        auxiliary_parameters = sum(
            parameter.numel()
            for parameter in reverse_model.parameters()
            if parameter.requires_grad
        )
    return vi_parameters, auxiliary_parameters, vi_parameters + auxiliary_parameters


def mode_diagnostics(
    runner: Any,
    num_samples: int,
) -> dict[str, float | int | list[float]]:
    _, samples = runner.vi_model.sampling(num=num_samples)
    centers = runner.target_model._centers
    sigma = float(runner.target_model.sigma)
    distances = torch.cdist(samples, centers)
    min_distances, assignments = distances.min(dim=1)
    # Three component standard deviations includes 98.9% of an isotropic
    # two-dimensional Gaussian component and excludes most bridge mass.
    near_mode = min_distances <= 3.0 * sigma
    near_counts = torch.bincount(
        assignments[near_mode],
        minlength=centers.shape[0],
    ).to(torch.float64)
    masses = near_counts / num_samples
    positive_masses = masses[masses > 0]
    entropy = 0.0
    if positive_masses.numel() > 0:
        normalized = positive_masses / positive_masses.sum()
        entropy = float(
            (-(normalized * normalized.log()).sum() / math.log(centers.shape[0])).item()
        )
    return {
        "modes_covered_1pct": int((masses >= 0.01).sum().item()),
        "near_mode_mass": float(near_mode.to(torch.float32).mean().item()),
        "mode_entropy": entropy,
        "mode_masses": [float(value) for value in masses.cpu().tolist()],
    }


def run_one(
    variant: Variant,
    seed: int,
    args: argparse.Namespace,
    device: torch.device,
) -> dict[str, Any]:
    config = configure_run(variant, seed, args, device)
    set_seed(seed, device)
    runner = Runners[str(config.runner_type)](config=config)
    runner.log_config()
    vi_parameters, auxiliary_parameters, total_parameters = parameter_counts(runner)

    synchronize(device)
    train_start = time.perf_counter()
    runner.learn()
    synchronize(device)
    training_time = time.perf_counter() - train_start

    # Reset evaluation randomness so corresponding method/seed pairs use the
    # same sample and projection streams.
    evaluation_seed = 1_000_000 + seed
    set_seed(evaluation_seed, device)
    w2 = runner.evaluate_vi_to_baseline_w2()
    set_seed(evaluation_seed + 1, device)
    elbo, elbo_std, elbo_std_q, elbo_ci_half = runner.evaluate_elbo()
    set_seed(evaluation_seed + 2, device)
    modes = mode_diagnostics(runner, args.mode_samples)

    record: dict[str, Any] = {
        "method": variant.label,
        "seed": seed,
        "epochs": args.epochs,
        "batch_size": int(config.train.batch_size),
        "flow_layers": variant.flow_layers,
        "vi_parameters": vi_parameters,
        "auxiliary_parameters": auxiliary_parameters,
        "total_parameters": total_parameters,
        "training_time_sec": training_time,
        "elbo": elbo,
        "elbo_std": elbo_std,
        "elbo_std_q": elbo_std_q,
        "elbo_ci_half": elbo_ci_half,
        "w2": w2,
        "run_dir": str(Path(runner.save_path).resolve()),
        **modes,
    }

    checkpoint_path = Path(runner.save_path) / "final_vi_model.pt"
    torch.save(runner.vi_model.state_dict(), checkpoint_path)
    if getattr(runner, "reverse_model", None) is not None:
        torch.save(
            runner.reverse_model.state_dict(),
            Path(runner.save_path) / "final_auxiliary_model.pt",
        )
    runner.writer.close()
    return record


def read_existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        records = list(csv.DictReader(handle))
    for record in records:
        record["seed"] = int(record["seed"])
        record["mode_masses"] = json.loads(record["mode_masses"])
    return records


def write_run_records(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "seed",
        "epochs",
        "batch_size",
        "flow_layers",
        "vi_parameters",
        "auxiliary_parameters",
        "total_parameters",
        "training_time_sec",
        "elbo",
        "elbo_std",
        "elbo_std_q",
        "elbo_ci_half",
        "w2",
        "modes_covered_1pct",
        "near_mode_mass",
        "mode_entropy",
        "mode_masses",
        "run_dir",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in sorted(records, key=lambda item: (item["method"], item["seed"])):
            serializable = dict(record)
            serializable["mode_masses"] = json.dumps(
                record["mode_masses"],
                separators=(",", ":"),
            )
            writer.writerow(serializable)


def summarize_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    methods = sorted({str(record["method"]) for record in records})
    for method in methods:
        group = [record for record in records if record["method"] == method]
        summary: dict[str, Any] = {
            "method": method,
            "n_seeds": len(group),
            "vi_parameters": int(float(group[0]["vi_parameters"])),
            "auxiliary_parameters": int(float(group[0]["auxiliary_parameters"])),
            "total_parameters": int(float(group[0]["total_parameters"])),
        }
        for field in NUMERIC_SUMMARY_FIELDS:
            values = [float(record[field]) for record in group]
            summary[f"{field}_mean"] = statistics.fmean(values)
            summary[f"{field}_sd"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summaries.append(summary)
    return summaries


def write_summary(path: Path, summaries: list[dict[str, Any]]) -> None:
    if not summaries:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def hardware_metadata(device: torch.device, args: argparse.Namespace) -> dict[str, Any]:
    device_name = platform.processor() or platform.machine()
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(device)
    return {
        "created_at": datetime.now().astimezone().isoformat(),
        "device": str(device),
        "device_name": device_name,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "protocol": {
            "target": "8_gaussians",
            "objective": "reverse_KL_with_matched_linear_annealing",
            "epochs": args.epochs,
            "w2_samples": args.w2_samples,
            "w2_projections": args.w2_projections,
            "elbo_samples": args.elbo_samples,
            "mode_samples": args.mode_samples,
            "timing": (
                "end-to-end optimization including DIVI auxiliary warmup; "
                "excluding evaluation, plotting, sampling, and checkpoints"
            ),
        },
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    report_dir = (PROJECT_ROOT / args.report_dir).resolve()
    run_metrics_path = report_dir / "run_metrics.csv"
    records = read_existing_records(run_metrics_path)
    completed = {
        (str(record["method"]), int(record["seed"]))
        for record in records
    }

    for variant in variants_from_args(args):
        for seed in args.seeds:
            key = (variant.label, seed)
            if key in completed and not args.overwrite:
                print(
                    f"Skipping completed run {variant.label}, seed={seed}",
                    flush=True,
                )
                continue
            print(
                f"Running {variant.label}, seed={seed}, device={device}",
                flush=True,
            )
            record = run_one(variant, seed, args, device)
            records = [
                existing
                for existing in records
                if (existing["method"], int(existing["seed"])) != key
            ]
            records.append(record)
            completed.add(key)
            write_run_records(run_metrics_path, records)
            write_summary(report_dir / "summary.csv", summarize_records(records))
            print(
                f"Completed {variant.label}, seed={seed}: "
                f"ELBO={record['elbo']:.4f}, W2={record['w2']:.4f}, "
                f"time={record['training_time_sec']:.2f}s",
                flush=True,
            )
            if device.type == "cuda":
                torch.cuda.empty_cache()

    metadata = hardware_metadata(device, args)
    with (report_dir / "metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
