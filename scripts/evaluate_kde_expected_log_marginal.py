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
from utils.elm import kde_expected_log_marginal, sample_reference_samples  # noqa: E402
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


def _resolve_dtype(dtype_name: str) -> torch.dtype:
    normalized = dtype_name.lower()
    if normalized == "float32":
        return torch.float32
    if normalized == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype: {dtype_name!r}.")


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


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
            raise RuntimeError("CUDA requested for runner device but not available.")
        device = "cuda"
    else:
        device = "cuda" if config.get("use_cuda", False) and torch.cuda.is_available() else "cpu"

    config.device = device
    config.use_cuda = device == "cuda"
    if config.get("vi_model") is not None:
        config.vi_model.device = device
    if config.get("reverse_model") is not None:
        config.reverse_model.device = device
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
    return runner


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


def _validate_budgets(sample_budgets: list[int]) -> list[int]:
    if not sample_budgets:
        raise ValueError("At least one sample budget is required.")
    budgets = []
    for budget in sample_budgets:
        if budget < 1:
            raise ValueError(f"Sample budgets must be positive, got {budget}.")
        budgets.append(int(budget))
    return sorted(set(budgets))


@torch.no_grad()
def generate_vi_samples(
    vi_model: torch.nn.Module,
    *,
    num_samples: int,
    sample_batch_size: int,
    cache_device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if num_samples < 1:
        raise ValueError("num_samples must be positive.")
    if sample_batch_size < 1:
        raise ValueError("sample_batch_size must be positive.")
    if cache_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("KDE sample cache device is CUDA, but CUDA is not available.")

    z_dim = int(getattr(vi_model, "z_dim"))
    samples = torch.empty(
        (num_samples, z_dim),
        device=cache_device,
        dtype=dtype,
    )
    was_training = vi_model.training
    vi_model.eval()
    try:
        for start in range(0, num_samples, sample_batch_size):
            stop = min(start + sample_batch_size, num_samples)
            _, z = vi_model.sampling(num=stop - start)
            samples[start:stop].copy_(z.to(device=cache_device, dtype=dtype))
            logger.info("Generated VI samples %s/%s", stop, num_samples)
    except Exception as exc:
        if "out of memory" in str(exc).lower() and "cuda" in str(exc).lower():
            raise RuntimeError(
                "CUDA ran out of memory while generating or caching VI samples. "
                "Retry with a smaller --sample-batch-size or lower maximum --sample-budgets."
            ) from exc
        raise
    finally:
        if was_training:
            vi_model.train()
    return samples


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_per_reference_csv(
    path: Path,
    rows_by_budget: list[tuple[int, torch.Tensor]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["sample_budget", "ref_index", "log_value"],
        )
        writer.writeheader()
        for sample_budget, values in rows_by_budget:
            for ref_index, value in enumerate(values.tolist()):
                writer.writerow(
                    {
                        "sample_budget": int(sample_budget),
                        "ref_index": int(ref_index),
                        "log_value": float(value),
                    }
                )


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = [
        "# KDE Expected Log Marginal",
        "",
        "| Model samples | KDE ELM | StdErr | Std | Min | Max | Clamped dims | Runtime |",
        "|---------------|---------|--------|-----|-----|-----|--------------|---------|",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["sample_budget"]),
                    _format_float(row["kde_expected_log_marginal"]),
                    _format_float(row["stderr_across_refs"]),
                    _format_float(row["std_across_refs"]),
                    _format_float(row["min_per_ref_log"]),
                    _format_float(row["max_per_ref_log"]),
                    str(row["num_bandwidth_clamped_dims"]),
                    f"{_format_float(row['runtime_sec'])}s",
                ]
            )
            + " |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _check_output_paths(paths: list[Path], overwrite: bool) -> None:
    if overwrite:
        for path in paths:
            path.unlink(missing_ok=True)
        return
    existing = [path for path in paths if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"Output files already exist. Pass --overwrite to replace: {joined}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate paper-style coordinate-wise KDE expected log marginal."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--sample-budgets",
        type=int,
        nargs="+",
        default=[10000, 60000, 100000, 200000],
    )
    parser.add_argument("--num-ref-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--kde-device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--dtype", choices=["float32", "float64"], default="float32")
    parser.add_argument("--sample-batch-size", type=int, default=20000)
    parser.add_argument("--dim-chunk", type=int, default=25)
    parser.add_argument("--ref-chunk", type=int, default=500)
    parser.add_argument("--model-chunk", type=int, default=20000)
    parser.add_argument("--min-bandwidth", type=float, default=1.0e-6)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_csv_path = output_dir / "summary.csv"
    summary_md_path = output_dir / "summary.md"
    per_ref_path = output_dir / "per_reference_log_values.csv"
    metadata_path = output_dir / "metadata.json"
    _check_output_paths(
        [summary_csv_path, summary_md_path, per_ref_path, metadata_path],
        overwrite=args.overwrite,
    )

    budgets = _validate_budgets(args.sample_budgets)
    max_budget = max(budgets)
    kde_device = torch.device(args.kde_device)
    dtype = _resolve_dtype(args.dtype)

    runner = build_runner(
        config_path=args.config.resolve(),
        checkpoint_dir=args.checkpoint_dir.resolve(),
        output_dir=output_dir,
        force_device=args.device,
    )
    if runner.baseline_samples is None:
        raise RuntimeError("This target has no baseline samples; KDE ELM cannot be evaluated.")

    try:
        _set_seed(args.seed, runner.device == "cuda" or kde_device.type == "cuda")
        reference_samples = sample_reference_samples(
            runner.baseline_samples,
            args.num_ref_samples,
            kde_device,
        ).to(dtype=dtype)

        generation_start = time.perf_counter()
        model_samples = generate_vi_samples(
            runner.vi_model,
            num_samples=max_budget,
            sample_batch_size=args.sample_batch_size,
            cache_device=kde_device,
            dtype=dtype,
        )
        _sync_if_cuda(kde_device)
        generation_runtime = time.perf_counter() - generation_start

        summary_rows: list[dict[str, Any]] = []
        per_ref_rows: list[tuple[int, torch.Tensor]] = []
        for budget in budgets:
            logger.info("Evaluating KDE ELM with %s generated samples", budget)
            _sync_if_cuda(kde_device)
            start = time.perf_counter()
            estimate = kde_expected_log_marginal(
                reference_samples,
                model_samples[:budget],
                dim_chunk=args.dim_chunk,
                ref_chunk=args.ref_chunk,
                model_chunk=args.model_chunk,
                min_bandwidth=args.min_bandwidth,
                dtype=dtype,
                device=kde_device,
            )
            _sync_if_cuda(kde_device)
            elapsed = time.perf_counter() - start
            diagnostics = dict(estimate.diagnostics)
            diagnostics["runtime_sec"] = float(elapsed)
            row = {
                "sample_budget": int(budget),
                "num_ref_samples": int(reference_samples.shape[0]),
                "z_dim": int(reference_samples.shape[1]),
                "kde_expected_log_marginal": float(estimate.value),
                "stderr_across_refs": float(estimate.stderr),
                "std_across_refs": float(diagnostics["std_across_refs"]),
                "min_per_ref_log": float(diagnostics["min_per_ref_log"]),
                "max_per_ref_log": float(diagnostics["max_per_ref_log"]),
                "runtime_sec": float(elapsed),
                "device": str(kde_device),
                "dtype": args.dtype,
                "dim_chunk": int(args.dim_chunk),
                "ref_chunk": int(args.ref_chunk),
                "model_chunk": int(args.model_chunk),
                "bandwidth_rule": "scott",
                "min_bandwidth": float(args.min_bandwidth),
                "num_bandwidth_clamped_dims": int(
                    diagnostics["num_bandwidth_clamped_dims"]
                ),
            }
            summary_rows.append(row)
            per_ref_rows.append((budget, estimate.per_reference_log_values))
            _write_csv(summary_csv_path, summary_rows)
            _write_markdown(summary_md_path, summary_rows)
            _write_per_reference_csv(per_ref_path, per_ref_rows)
            logger.info(
                "KDE ELM budget=%s value=%.6f stderr=%.6f elapsed=%.3fs",
                budget,
                estimate.value,
                estimate.stderr,
                elapsed,
            )

        metadata = {
            "config": str(args.config.resolve()),
            "checkpoint_dir": str(args.checkpoint_dir.resolve()),
            "seed": int(args.seed),
            "sample_budgets": budgets,
            "num_ref_samples": int(reference_samples.shape[0]),
            "max_sample_budget": int(max_budget),
            "sample_batch_size": int(args.sample_batch_size),
            "bandwidth_rule": "scott",
            "min_bandwidth": float(args.min_bandwidth),
            "dim_chunk": int(args.dim_chunk),
            "ref_chunk": int(args.ref_chunk),
            "model_chunk": int(args.model_chunk),
            "dtype": args.dtype,
            "runner_device": str(runner.device),
            "kde_device": str(kde_device),
            "generation_runtime_sec": float(generation_runtime),
        }
        metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        print(f"Wrote {summary_csv_path}")
        print(f"Wrote {summary_md_path}")
        print(f"Wrote {per_ref_path}")
        print(f"Wrote {metadata_path}")
    finally:
        if hasattr(runner, "writer"):
            runner.writer.close()
        _remove_file_handlers()
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
