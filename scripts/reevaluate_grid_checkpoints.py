from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, REPO_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from grid_benchmark_common import (  # noqa: E402
    BNN_TARGETS,
    CAMPAIGN_DIR,
    CAMPAIGN_SLUG,
    MANIFEST_PATH,
    METHOD_VARIANTS,
    REPO_ROOT,
    TARGETS,
    metric_support,
)
from runner.runners import Runners  # noqa: E402
from utils.logging import get_logger  # noqa: E402


logger = get_logger()

OUTPUT_DIR = CAMPAIGN_DIR / "generated_reports"
SUMMARY_CSV_PATH = OUTPUT_DIR / "official_reevaluation_summary.csv"
RAW_JSONL_PATH = OUTPUT_DIR / "official_reevaluation_raw.jsonl"
SKIPPED_CSV_PATH = OUTPUT_DIR / "official_reevaluation_skipped.csv"
MARKDOWN_PATH = OUTPUT_DIR / "official_reevaluation_by_target.md"

SCRATCH_RESULTS_DIR = f"results/{CAMPAIGN_SLUG}/reeval_scratch"
SCRATCH_TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}/reeval_scratch"

REPORT_METRICS = ["elbo", "kl", "w2", "mmd", "ksd", "rmse", "nll"]
def _resolve_repo_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str)
    if path.exists():
        return path
    for anchor in ("tb_logs", "results", "configs", "campaigns", "baselines"):
        if anchor in path.parts:
            idx = path.parts.index(anchor)
            return REPO_ROOT.joinpath(*path.parts[idx:])
    return REPO_ROOT / path


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest_map() -> tuple[dict[str, dict[str, Any]], dict[str, int]]:
    manifest = _read_json(MANIFEST_PATH)
    by_run_id: dict[str, dict[str, Any]] = {}
    run_order: dict[str, int] = {}
    for idx, entry in enumerate(manifest):
        by_run_id[entry["run_id"]] = entry
        run_order[entry["run_id"]] = idx
    return by_run_id, run_order


def _load_completed_rows() -> list[dict[str, str]]:
    with (OUTPUT_DIR / "official_completed_runs.csv").open(
        "r",
        encoding="utf-8",
        newline="",
    ) as fh:
        rows = list(csv.DictReader(fh))
    return [row for row in rows if row.get("status") == "completed"]


def _load_existing_summary() -> dict[str, dict[str, str]]:
    if not SUMMARY_CSV_PATH.exists():
        return {}
    with SUMMARY_CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        return {row["run_id"]: row for row in csv.DictReader(fh)}


def _load_existing_skips() -> dict[str, dict[str, str]]:
    if not SKIPPED_CSV_PATH.exists():
        return {}
    with SKIPPED_CSV_PATH.open("r", encoding="utf-8", newline="") as fh:
        return {row["run_id"]: row for row in csv.DictReader(fh)}


def _variant_order_key(entry: dict[str, Any]) -> tuple[int, int]:
    try:
        variant_idx = METHOD_VARIANTS.index(entry["variant"])
    except ValueError:
        variant_idx = len(METHOD_VARIANTS)
    anneal_idx = 0 if entry.get("annealing_mode") == "on" else 1
    return variant_idx, anneal_idx


def _find_final_checkpoint(result_dir: Path) -> tuple[Path | None, int | None, str | None]:
    ckpt_root = result_dir / "checkpoints"
    if not ckpt_root.exists():
        return None, None, "checkpoints_dir_missing"

    candidates: list[tuple[int, Path]] = []
    for epoch_dir in ckpt_root.glob("epoch_*"):
        if not epoch_dir.is_dir():
            continue
        try:
            epoch = int(epoch_dir.name.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        vi_ckpt = epoch_dir / "vi_model.pt"
        if vi_ckpt.is_file():
            candidates.append((epoch, epoch_dir))

    if not candidates:
        return None, None, "vi_model_missing"

    epoch, ckpt_dir = max(candidates, key=lambda item: item[0])
    return ckpt_dir, epoch, None


def _set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, str):
        try:
            value = float(value)
        except ValueError:
            return value
    if not math.isfinite(value):
        return "nan"
    abs_value = abs(value)
    if abs_value >= 1.0e4 or (0 < abs_value < 1.0e-3):
        return f"{value:.3e}"
    if abs_value >= 100:
        return f"{value:.3f}"
    if abs_value >= 1:
        return f"{value:.4f}"
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _metric_support_for_target(target: str) -> dict[str, bool]:
    support = metric_support(target)
    is_bnn = target in BNN_TARGETS
    return {
        "elbo": True,
        "kl": bool(support["kl"]),
        "w2": bool(support["w2"]),
        "mmd": bool(support["mmd"]),
        "ksd": bool(support["ksd"]),
        "rmse": is_bnn,
        "nll": is_bnn,
    }


def _budget_for_target(target: str) -> dict[str, int]:
    if target in BNN_TARGETS:
        return {
            "kl_num_samples": 0,
            "w2_num_samples": 0,
            "w2_num_projections": 0,
            "mmd_num_samples": 0,
            "ksd_num_samples": 5000,
            "elbo_num_z_samples": 1024,
            "elbo_batch_size": 256,
            "elbo_num_batches": 16,
            "bnn_num_samples": 2000,
        }
    if target == "Langevin_post":
        return {
            "kl_num_samples": 0,
            "w2_num_samples": 50000,
            "w2_num_projections": 4096,
            "mmd_num_samples": 5000,
            "ksd_num_samples": 10000,
            "elbo_num_z_samples": 4096,
            "elbo_batch_size": 256,
            "elbo_num_batches": 16,
            "bnn_num_samples": 0,
        }
    if target == "LRwaveform":
        return {
            "kl_num_samples": 0,
            "w2_num_samples": 0,
            "w2_num_projections": 0,
            "mmd_num_samples": 0,
            "ksd_num_samples": 10000,
            "elbo_num_z_samples": 4096,
            "elbo_batch_size": 512,
            "elbo_num_batches": 16,
            "bnn_num_samples": 0,
        }
    return {
        "kl_num_samples": 50000,
        "w2_num_samples": 50000,
        "w2_num_projections": 4096,
        "mmd_num_samples": 5000,
        "ksd_num_samples": 10000,
        "elbo_num_z_samples": 4096,
        "elbo_batch_size": 512,
        "elbo_num_batches": 16,
        "bnn_num_samples": 0,
    }


def _remove_file_handlers() -> None:
    root_logger = get_logger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


def _prepare_config(
    config_path: Path,
    run_id: str,
    force_device: str,
) -> DictConfig:
    main_cfg: DictConfig = OmegaConf.load(config_path)  # type: ignore[assignment]
    main_cfg.config_path = config_path.as_posix()

    if force_device == "cpu":
        device = "cpu"
    elif force_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = "cuda"
    else:
        device = "cuda" if main_cfg.get("use_cuda", False) and torch.cuda.is_available() else "cpu"

    main_cfg.device = device
    main_cfg.use_cuda = device == "cuda"
    main_cfg.output = OmegaConf.merge(  # type: ignore[assignment]
        main_cfg.get("output", {}),
        {
            "results_dir": f"{SCRATCH_RESULTS_DIR}/{run_id}",
            "tb_dir": f"{SCRATCH_TB_DIR}/{run_id}",
        },
    )
    return main_cfg


def build_runner_for_evaluation(
    config_path: Path,
    runner_type: str,
    run_id: str,
    checkpoint_dir: Path,
    target: str,
    force_device: str,
):
    main_cfg = _prepare_config(config_path, run_id, force_device)
    base_seed = int(main_cfg.get("seed", 42))
    _set_seed(base_seed, main_cfg.device == "cuda")

    runner = Runners[runner_type](config=main_cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    _remove_file_handlers()

    state = torch.load(checkpoint_dir / "vi_model.pt", map_location=runner.device)
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()

    budget = _budget_for_target(target)
    support = _metric_support_for_target(target)
    runner.n_ite_samples = budget["kl_num_samples"]
    runner.n_w2_samples = budget["w2_num_samples"]
    runner.n_w2_projections = budget["w2_num_projections"]
    runner.n_mmd_samples = budget["mmd_num_samples"]
    runner.n_ksd_samples = budget["ksd_num_samples"]
    runner.n_elbo_z_samples = budget["elbo_num_z_samples"]
    runner.n_elbo_batch_size = budget["elbo_batch_size"]
    runner.n_elbo_batches = budget["elbo_num_batches"]
    runner.n_bnn_samples = budget["bnn_num_samples"]
    runner.metric_mmd_enabled = bool(support["mmd"])
    if getattr(runner, "metric_mmd_enabled", False):
        runner._init_mmd_baseline_samples()

    return runner, base_seed


def _capture_metric(
    metric_name: str,
    fn,
    values: dict[str, float],
    errors: dict[str, str],
    runtimes: dict[str, float],
) -> None:
    start = time.perf_counter()
    try:
        values[metric_name] = float(fn())
    except Exception as exc:  # noqa: BLE001
        errors[metric_name] = f"{type(exc).__name__}: {exc}"
    finally:
        runtimes[f"{metric_name}_runtime_sec"] = time.perf_counter() - start


def evaluate_runner_once(
    runner,
    target: str,
    support: dict[str, bool],
    seed: int,
) -> dict[str, Any]:
    _set_seed(seed, runner.device == "cuda")
    if runner.device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    values: dict[str, float] = {}
    errors: dict[str, str] = {}
    runtimes: dict[str, float] = {}

    if support["elbo"]:
        _capture_metric(
            "elbo",
            lambda: runner.evaluate_elbo()[0],
            values,
            errors,
            runtimes,
        )
    if support["kl"]:
        _capture_metric("kl", runner.evaluate_vi_to_baseline_kl, values, errors, runtimes)
    if support["w2"]:
        _capture_metric("w2", runner.evaluate_vi_to_baseline_w2, values, errors, runtimes)
    if support["mmd"]:
        _capture_metric("mmd", runner.evaluate_mmd, values, errors, runtimes)
    if support["ksd"]:
        _capture_metric("ksd", runner.evaluate_ksd, values, errors, runtimes)
    if support["rmse"] or support["nll"]:
        start = time.perf_counter()
        try:
            rmse, test_llk = runner.evaluate_bnn_metrics()
            values["rmse"] = float(rmse)
            values["nll"] = float(-test_llk)
        except Exception as exc:  # noqa: BLE001
            message = f"{type(exc).__name__}: {exc}"
            if support["rmse"]:
                errors["rmse"] = message
            if support["nll"]:
                errors["nll"] = message
        finally:
            runtimes["bnn_runtime_sec"] = time.perf_counter() - start

    return {
        "target": target,
        "seed": seed,
        "metrics": values,
        "errors": errors,
        "runtimes": runtimes,
    }


def summarize_metric(values: list[float]) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    arr = np.asarray(values, dtype=np.float64)
    mean = float(arr.mean())
    if arr.size == 1:
        return mean, 0.0
    se = float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, se


def summarize_run(
    run_info: dict[str, Any],
    repeats: list[dict[str, Any]],
    support: dict[str, bool],
) -> dict[str, Any]:
    summary: dict[str, Any] = dict(run_info)
    summary["repeat_count"] = len(repeats)

    for metric in REPORT_METRICS:
        metric_values = [
            repeat["metrics"][metric]
            for repeat in repeats
            if metric in repeat["metrics"]
        ]
        metric_mean, metric_se = summarize_metric(metric_values)
        summary[f"{metric}_mean"] = metric_mean
        summary[f"{metric}_se"] = metric_se
        summary[f"{metric}_count"] = len(metric_values)
        summary[f"{metric}_supported"] = support[metric]

    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _metric_cell(row: dict[str, Any], metric: str) -> str:
    supported = row.get(f"{metric}_supported", False)
    if isinstance(supported, str):
        supported = supported.lower() == "true"
    if not supported:
        return "N/A"
    mean = row.get(f"{metric}_mean")
    se = row.get(f"{metric}_se")
    if mean is None:
        return "FAILED"
    return f"{_format_float(mean)} +/- {_format_float(se)}"


def _write_markdown(
    path: Path,
    rows: list[dict[str, Any]],
    skips: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Official Re-Evaluation By Target",
        "",
        f"Runs summarized: {len(rows)}",
        "",
    ]

    for target in TARGETS:
        target_rows = [row for row in rows if row["target"] == target]
        if not target_rows:
            continue
        lines.extend(
            [
                f"## {target}",
                "",
                "| Variant | Anneal | ELBO | KL | W2 | MMD | KSD | RMSE | NLL | Train Time / Ckpt |",
                "|---------|--------|------|----|----|-----|-----|------|-----|-------------------|",
            ]
        )
        for row in target_rows:
            lines.append(
                "| "
                + " | ".join(
                    [
                        str(row["variant_label"]),
                        str(row["annealing_mode"]),
                        _metric_cell(row, "elbo"),
                        _metric_cell(row, "kl"),
                        _metric_cell(row, "w2"),
                        _metric_cell(row, "mmd"),
                        _metric_cell(row, "ksd"),
                        _metric_cell(row, "rmse"),
                        _metric_cell(row, "nll"),
                        f"{_format_float(row['duration_sec'])}s / e{row['checkpoint_epoch']}",
                    ]
                )
                + " |"
            )
        lines.append("")

    if skips:
        lines.extend(["## Skipped Runs", ""])
        lines.extend(
            [
                "| Target | Variant | Anneal | Reason |",
                "|--------|---------|--------|--------|",
            ]
        )
        for row in skips:
            lines.append(
                f"| {row['target']} | {row['variant_label']} | {row['annealing_mode']} | {row['reason']} |"
            )
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def _summary_row_to_csv(row: dict[str, Any]) -> dict[str, Any]:
    csv_row: dict[str, Any] = {}
    for key, value in row.items():
        if isinstance(value, float):
            csv_row[key] = f"{value:.12g}"
        else:
            csv_row[key] = value
    return csv_row


def _append_raw_records(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        for record in records:
            fh.write(json.dumps(record, ensure_ascii=True) + "\n")


def _ordered_summary_rows(summary_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_run_ids = sorted(
        summary_rows,
        key=lambda key: (
            TARGETS.index(summary_rows[key]["target"])
            if summary_rows[key]["target"] in TARGETS
            else len(TARGETS),
            METHOD_VARIANTS.index(summary_rows[key]["variant_key"])
            if summary_rows[key]["variant_key"] in METHOD_VARIANTS
            else len(METHOD_VARIANTS),
            0 if summary_rows[key]["annealing_mode"] == "on" else 1,
            key,
        ),
    )
    return [summary_rows[run_id] for run_id in ordered_run_ids]


def _ordered_skip_rows(skipped_rows: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    ordered_run_ids = sorted(
        skipped_rows,
        key=lambda key: (
            TARGETS.index(skipped_rows[key]["target"])
            if skipped_rows[key]["target"] in TARGETS
            else len(TARGETS),
            skipped_rows[key]["variant_label"],
            skipped_rows[key]["annealing_mode"],
            key,
        ),
    )
    return [skipped_rows[run_id] for run_id in ordered_run_ids]


def _flush_outputs(
    summary_rows: dict[str, dict[str, Any]],
    skipped_rows: dict[str, dict[str, Any]],
) -> None:
    ordered_summary_rows = _ordered_summary_rows(summary_rows)
    ordered_skip_rows = _ordered_skip_rows(skipped_rows)
    _write_csv(
        SUMMARY_CSV_PATH,
        [_summary_row_to_csv(row) for row in ordered_summary_rows],
    )
    _write_csv(SKIPPED_CSV_PATH, ordered_skip_rows)
    _write_markdown(MARKDOWN_PATH, ordered_summary_rows, ordered_skip_rows)


def _selected_run_ids(args: argparse.Namespace) -> set[str] | None:
    selected = set(args.run_id or [])
    return selected or None


def _select_rows(
    completed_rows: list[dict[str, str]],
    manifest_by_run_id: dict[str, dict[str, Any]],
    run_order: dict[str, int],
    args: argparse.Namespace,
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    selected_run_ids = _selected_run_ids(args)
    selected_targets = set(args.target or [])

    selected: list[tuple[dict[str, str], dict[str, Any]]] = []
    for row in completed_rows:
        manifest_entry = manifest_by_run_id.get(row["run_id"])
        if manifest_entry is None:
            continue
        if selected_run_ids is not None and row["run_id"] not in selected_run_ids:
            continue
        if selected_targets and row["target"] not in selected_targets:
            continue
        selected.append((row, manifest_entry))

    selected.sort(
        key=lambda item: (
            TARGETS.index(item[0]["target"]) if item[0]["target"] in TARGETS else len(TARGETS),
            _variant_order_key(item[1]),
            run_order.get(item[0]["run_id"], 10**9),
        )
    )
    if args.max_runs is not None:
        selected = selected[: args.max_runs]
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Re-evaluate final VI checkpoints for the official grid benchmark.")
    parser.add_argument("--repeat-count", type=int, default=50)
    parser.add_argument("--run-id", action="append", default=[], help="Restrict to a specific run_id. May be passed multiple times.")
    parser.add_argument("--target", action="append", default=[], help="Restrict to a target. May be passed multiple times.")
    parser.add_argument("--max-runs", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    manifest_by_run_id, run_order = _load_manifest_map()
    completed_rows = _load_completed_rows()
    selected = _select_rows(completed_rows, manifest_by_run_id, run_order, args)

    if not selected:
        print("No completed runs matched the selection.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    existing_summary = {} if args.overwrite else _load_existing_summary()
    existing_skips = {} if args.overwrite else _load_existing_skips()
    if args.overwrite:
        for path in (SUMMARY_CSV_PATH, RAW_JSONL_PATH, SKIPPED_CSV_PATH, MARKDOWN_PATH):
            path.unlink(missing_ok=True)

    summary_rows: dict[str, dict[str, Any]] = {run_id: dict(row) for run_id, row in existing_summary.items()}
    skipped_rows: dict[str, dict[str, Any]] = {run_id: dict(row) for run_id, row in existing_skips.items()}

    for row, manifest_entry in selected:
        run_id = row["run_id"]
        if run_id in summary_rows or run_id in skipped_rows:
            logger.info(f"Skipping already processed run: {run_id}")
            continue

        config_path = _resolve_repo_path(
            row.get("config_path") or manifest_entry.get("config_path")
        )
        result_dir = _resolve_repo_path(
            row.get("result_path") or manifest_entry.get("result_path")
        )
        if config_path is None or not config_path.exists():
            skipped_rows[run_id] = {
                "run_id": run_id,
                "target": row["target"],
                "variant_label": manifest_entry["variant_label"],
                "annealing_mode": row["annealing_mode"],
                "reason": "config_missing",
            }
            _flush_outputs(summary_rows, skipped_rows)
            continue
        if result_dir is None or not result_dir.exists():
            skipped_rows[run_id] = {
                "run_id": run_id,
                "target": row["target"],
                "variant_label": manifest_entry["variant_label"],
                "annealing_mode": row["annealing_mode"],
                "reason": "result_dir_missing",
            }
            _flush_outputs(summary_rows, skipped_rows)
            continue

        checkpoint_dir, checkpoint_epoch, error_reason = _find_final_checkpoint(result_dir)
        if checkpoint_dir is None or checkpoint_epoch is None:
            skipped_rows[run_id] = {
                "run_id": run_id,
                "target": row["target"],
                "variant_label": manifest_entry["variant_label"],
                "annealing_mode": row["annealing_mode"],
                "reason": error_reason or "checkpoint_missing",
            }
            _flush_outputs(summary_rows, skipped_rows)
            continue

        logger.info(
            f"Evaluating {run_id} from {checkpoint_dir.as_posix()} for {args.repeat_count} repeats."
        )
        support = _metric_support_for_target(row["target"])
        runner, base_seed = build_runner_for_evaluation(
            config_path=config_path,
            runner_type=manifest_entry["runner_type"],
            run_id=run_id,
            checkpoint_dir=checkpoint_dir,
            target=row["target"],
            force_device=args.device,
        )

        raw_records: list[dict[str, Any]] = []
        repeat_outputs: list[dict[str, Any]] = []
        try:
            for repeat_idx in range(args.repeat_count):
                seed = base_seed + repeat_idx
                repeat_start = time.perf_counter()
                repeat_output = evaluate_runner_once(
                    runner=runner,
                    target=row["target"],
                    support=support,
                    seed=seed,
                )
                repeat_elapsed = time.perf_counter() - repeat_start
                repeat_output["repeat_idx"] = repeat_idx
                repeat_output["elapsed_sec"] = repeat_elapsed
                raw_record = {
                    "run_id": run_id,
                    "target": row["target"],
                    "variant_key": manifest_entry["variant"],
                    "variant_label": manifest_entry["variant_label"],
                    "annealing_mode": row["annealing_mode"],
                    "checkpoint_epoch": checkpoint_epoch,
                    "checkpoint_dir": checkpoint_dir.as_posix(),
                    **repeat_output,
                }
                raw_records.append(raw_record)
                repeat_outputs.append(repeat_output)
        finally:
            if hasattr(runner, "writer"):
                runner.writer.close()
            _remove_file_handlers()
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        _append_raw_records(RAW_JSONL_PATH, raw_records)

        run_info = {
            "run_id": run_id,
            "target": row["target"],
            "variant_key": manifest_entry["variant"],
            "variant_label": manifest_entry["variant_label"],
            "annealing_mode": row["annealing_mode"],
            "config_path": config_path.as_posix(),
            "result_path": result_dir.as_posix(),
            "checkpoint_epoch": checkpoint_epoch,
            "checkpoint_dir": checkpoint_dir.as_posix(),
            "duration_sec": float(row["duration_sec"]),
        }
        summary_rows[run_id] = summarize_run(run_info, repeat_outputs, support)
        _flush_outputs(summary_rows, skipped_rows)

    _flush_outputs(summary_rows, skipped_rows)
    print(f"Wrote {SUMMARY_CSV_PATH}")
    print(f"Wrote {RAW_JSONL_PATH}")
    print(f"Wrote {SKIPPED_CSV_PATH}")
    print(f"Wrote {MARKDOWN_PATH}")


if __name__ == "__main__":
    main()
