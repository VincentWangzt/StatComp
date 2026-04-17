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

from runner.runners import Runners  # noqa: E402
from utils.elm import (  # noqa: E402
    estimate_log_q_reverse_is,
    fit_reverse_proposal,
    sample_reference_samples,
    save_reverse_proposal_fit,
    summarize_elm,
)
from utils.logging import get_logger  # noqa: E402


logger = get_logger()


def _set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _remove_file_handlers() -> None:
    root_logger = get_logger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


def _prepare_config(
    config_path: Path,
    output_dir: Path,
    force_device: str,
) -> DictConfig:
    config: DictConfig = OmegaConf.load(config_path)  # type: ignore[assignment]
    config.config_path = config_path.as_posix()

    if force_device == "cpu":
        device = "cpu"
    elif force_device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        device = "cuda"
    else:
        device = "cuda" if config.get("use_cuda", False) and torch.cuda.is_available() else "cpu"

    config.device = device
    config.use_cuda = device == "cuda"
    config.output = OmegaConf.merge(  # type: ignore[assignment]
        config.get("output", {}),
        {
            "results_dir": (output_dir / "scratch_results").as_posix(),
            "tb_dir": (output_dir / "scratch_tb").as_posix(),
        },
    )
    return config


def build_runner(
    config_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    force_device: str,
):
    config = _prepare_config(config_path, output_dir, force_device)
    runner_type = str(config.runner_type)
    seed = int(config.get("seed", 42))
    _set_seed(seed, config.device == "cuda")

    runner = Runners[runner_type](config=config)
    if hasattr(runner, "writer"):
        runner.writer.close()
    _remove_file_handlers()

    vi_checkpoint = checkpoint_dir / "vi_model.pt"
    if not vi_checkpoint.is_file():
        raise FileNotFoundError(f"VI checkpoint not found: {vi_checkpoint}")
    state = torch.load(vi_checkpoint, map_location=runner.device)
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()
    return runner, seed


def _parse_repeat_budget(value: str) -> tuple[int, int]:
    try:
        repeats_str, samples_str = value.split(":", 1)
        repeats = int(repeats_str)
        samples = int(samples_str)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "repeat budgets must be formatted as REPEATS:SAMPLES, e.g. 100:6000"
        ) from exc
    if repeats < 1 or samples < 1:
        raise argparse.ArgumentTypeError("repeat budgets must be positive")
    return repeats, samples


def _format_float(value: float | None) -> str:
    if value is None:
        return "N/A"
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


def _summarize(values: list[float]) -> tuple[float | None, float | None, float | None, float | None]:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return None, None, None, None
    arr = np.asarray(finite, dtype=np.float64)
    mean = float(arr.mean())
    se = 0.0 if arr.size == 1 else float(arr.std(ddof=1) / math.sqrt(arr.size))
    return mean, se, float(arr.min()), float(arr.max())


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summary_rows(raw_records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_budget: dict[int, list[dict[str, Any]]] = {}
    for record in raw_records:
        by_budget.setdefault(int(record["num_is_samples"]), []).append(record)

    rows: list[dict[str, Any]] = []
    for budget, records in sorted(by_budget.items()):
        values = [float(record["value"]) for record in records]
        mean, se, min_value, max_value = _summarize(values)
        ess_median_values = [
            float(record["diagnostics"].get("ess_median", float("nan")))
            for record in records
        ]
        ess_mean, ess_se, ess_min, ess_max = _summarize(ess_median_values)
        runtime_values = [
            float(record["diagnostics"].get("runtime_sec", float("nan")))
            for record in records
        ]
        runtime_mean, runtime_se, runtime_min, runtime_max = _summarize(runtime_values)
        rows.append(
            {
                "num_is_samples": budget,
                "repeat_count": len(records),
                "elm_mean": mean,
                "elm_se": se,
                "elm_min": min_value,
                "elm_max": max_value,
                "ess_median_mean": ess_mean,
                "ess_median_se": ess_se,
                "ess_median_min": ess_min,
                "ess_median_max": ess_max,
                "runtime_sec_mean": runtime_mean,
                "runtime_sec_se": runtime_se,
                "runtime_sec_min": runtime_min,
                "runtime_sec_max": runtime_max,
            }
        )
    return rows


def _write_markdown(path: Path, summary_rows: list[dict[str, Any]], fit_diagnostics: dict[str, Any]) -> None:
    fit_lr = fit_diagnostics.get("fit_lr")
    fit_nll = fit_diagnostics.get("fit_nll")
    fit_runtime = fit_diagnostics.get("fit_runtime_sec")
    fit_summary_lines = [
        f"Proposal: `{fit_diagnostics.get('proposal_type', 'unknown')}`",
        f"Proposal class: `{fit_diagnostics.get('proposal_class', 'unknown')}`",
        f"Fit mode: `{fit_diagnostics.get('fit_mode', 'unknown')}`",
        f"Fit samples: `{fit_diagnostics.get('fit_samples', 'N/A')}`",
        f"Fit batch size: `{fit_diagnostics.get('fit_batch_size', 'N/A')}`",
        f"Fit epochs: `{fit_diagnostics.get('fit_epochs', 'N/A')}`",
        f"Fit LR: `{_format_float(None if fit_lr is None else float(fit_lr))}`",
        f"Fit NLL: `{_format_float(None if fit_nll is None else float(fit_nll))}`",
        f"Fit runtime: `{_format_float(None if fit_runtime is None else float(fit_runtime))}s`",
    ]
    if "num_components" in fit_diagnostics:
        fit_summary_lines.append(f"MoG components: `{fit_diagnostics['num_components']}`")
    if "fit_loss_initial" in fit_diagnostics:
        fit_summary_lines.extend(
            [
                f"Fit loss initial: `{_format_float(float(fit_diagnostics.get('fit_loss_initial', float('nan'))))}`",
                f"Fit loss final: `{_format_float(float(fit_diagnostics.get('fit_loss_final', float('nan'))))}`",
                f"Fit loss best: `{_format_float(float(fit_diagnostics.get('fit_loss_best', float('nan'))))}`",
            ]
        )
    lines = [
        "# Reverse-IS Expected Log Marginal",
        "",
        *fit_summary_lines,
        "",
        "| IS samples | Repeats | ELM mean +/- SE | ELM min | ELM max | Median ESS mean +/- SE | Runtime mean +/- SE |",
        "|------------|---------|-----------------|---------|---------|-------------------------|---------------------|",
    ]
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["num_is_samples"]),
                    str(row["repeat_count"]),
                    f"{_format_float(row['elm_mean'])} +/- {_format_float(row['elm_se'])}",
                    _format_float(row["elm_min"]),
                    _format_float(row["elm_max"]),
                    f"{_format_float(row['ess_median_mean'])} +/- {_format_float(row['ess_median_se'])}",
                    f"{_format_float(row['runtime_sec_mean'])}s +/- {_format_float(row['runtime_sec_se'])}s",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate expected log marginal with a reverse-IS proposal."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat-budget", action="append", type=_parse_repeat_budget, required=True)
    parser.add_argument("--num-ref-samples", type=int, default=1000)
    parser.add_argument("--proposal-type", choices=["gaussian", "mog", "realnvp"], default="gaussian")
    parser.add_argument("--proposal-config", type=Path, default=None)
    parser.add_argument("--proposal-fit-samples", "--fit-samples", dest="proposal_fit_samples", type=int, default=32768)
    parser.add_argument("--proposal-fit-batch-size", type=int, default=8192)
    parser.add_argument("--proposal-fit-epochs", type=int, default=1000)
    parser.add_argument("--proposal-lr", type=float, default=None)
    parser.add_argument("--proposal-log-every", type=int, default=100)
    parser.add_argument("--save-fitted-proposal", action="store_true")
    parser.add_argument("--is-batch-size", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output_dir / "raw.jsonl"
    summary_csv_path = output_dir / "summary.csv"
    summary_md_path = output_dir / "summary.md"
    fit_json_path = output_dir / "proposal_fit.json"
    if args.overwrite:
        for path in (raw_path, summary_csv_path, summary_md_path, fit_json_path):
            path.unlink(missing_ok=True)

    runner, config_seed = build_runner(
        config_path=args.config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        output_dir=output_dir,
        force_device=args.device,
    )
    base_seed = config_seed if args.seed is None else int(args.seed)

    if runner.baseline_samples is None:
        raise RuntimeError("This target has no baseline samples; ELM cannot be evaluated.")

    try:
        _set_seed(base_seed, runner.device == "cuda")
        reference_samples = sample_reference_samples(
            runner.baseline_samples,
            args.num_ref_samples,
            runner.device,
        )
        proposal_fit = fit_reverse_proposal(
            runner.vi_model,
            proposal_type=args.proposal_type,
            proposal_config_path=args.proposal_config,
            num_fit_samples=args.proposal_fit_samples,
            fit_batch_size=args.proposal_fit_batch_size,
            fit_epochs=args.proposal_fit_epochs,
            fit_lr=args.proposal_lr,
            log_every=args.proposal_log_every,
        )
        save_reverse_proposal_fit(
            output_dir,
            proposal_fit,
            save_state=args.save_fitted_proposal,
        )

        raw_records: list[dict[str, Any]] = []
        with raw_path.open("a", encoding="utf-8") as raw_fh:
            for repeat_count, num_is_samples in args.repeat_budget:
                for repeat_idx in range(repeat_count):
                    seed = base_seed + num_is_samples + repeat_idx
                    _set_seed(seed, runner.device == "cuda")
                    repeat_start = time.perf_counter()
                    log_q_estimate = estimate_log_q_reverse_is(
                        runner.vi_model,
                        proposal_fit.reverse_model,
                        reference_samples,
                        num_is_samples=num_is_samples,
                        is_batch_size=args.is_batch_size,
                        proposal_cache=proposal_fit.cache,
                    )
                    result = summarize_elm(log_q_estimate)
                    elapsed = time.perf_counter() - repeat_start
                    record = {
                        "repeat_idx": repeat_idx,
                        "seed": seed,
                        "proposal_type": proposal_fit.proposal_type,
                        "num_is_samples": int(num_is_samples),
                        "num_ref_samples": int(reference_samples.shape[0]),
                        "value": float(result.value),
                        "stderr": float(result.stderr),
                        "elapsed_sec": float(elapsed),
                        "diagnostics": result.diagnostics,
                    }
                    raw_fh.write(json.dumps(record, ensure_ascii=True) + "\n")
                    raw_fh.flush()
                    raw_records.append(record)
                    summary_rows = _summary_rows(raw_records)
                    _write_csv(summary_csv_path, summary_rows)
                    _write_markdown(summary_md_path, summary_rows, proposal_fit.diagnostics)
                    logger.info(
                        "ELM reverse-IS budget=%s repeat=%s/%s value=%.6f stderr=%.6f elapsed=%.3fs",
                        num_is_samples,
                        repeat_idx + 1,
                        repeat_count,
                        record["value"],
                        record["stderr"],
                        elapsed,
                    )

        summary_rows = _summary_rows(raw_records)
        _write_csv(summary_csv_path, summary_rows)
        _write_markdown(summary_md_path, summary_rows, proposal_fit.diagnostics)
        print(f"Wrote {raw_path}")
        print(f"Wrote {summary_csv_path}")
        print(f"Wrote {summary_md_path}")
        print(f"Wrote {fit_json_path}")
    finally:
        if hasattr(runner, "writer"):
            runner.writer.close()
        _remove_file_handlers()
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
