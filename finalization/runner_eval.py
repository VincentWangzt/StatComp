from __future__ import annotations

import csv
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf

from runner.runners import Runners
from utils.logging import get_logger
from utils.metrics import compute_sliced_wasserstein

from .artifacts import RunRecord, find_final_checkpoint, load_baseline_samples
from .config import repo_path


logger = get_logger()


def _finite_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _first_finite(*values: Any) -> float | None:
    for value in values:
        parsed = _finite_float(value)
        if parsed is not None:
            return parsed
    return None


def _campaign_summary_path(cfg: Any) -> Path:
    manifest_path = repo_path(str(cfg.campaign.manifest_path))
    assert manifest_path is not None
    return manifest_path.parent / "generated_reports" / "summary.csv"


def load_campaign_timing(cfg: Any) -> dict[str, dict[str, float]]:
    summary_path = _campaign_summary_path(cfg)
    if not summary_path.exists():
        return {}
    timing: dict[str, dict[str, float]] = {}
    with summary_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            run_id = row.get("run_id")
            if not run_id:
                continue
            values: dict[str, float] = {}
            for key in ("wall_clock_sec", "training_time_sec", "iterations"):
                parsed = _finite_float(row.get(key))
                if parsed is not None:
                    values[key] = parsed
            if values:
                timing[run_id] = values
    return timing


def augment_run_rows_with_campaign_timing(rows: list[dict[str, Any]], cfg: Any) -> list[dict[str, Any]]:
    timing = load_campaign_timing(cfg)
    if not timing:
        return rows
    augmented: list[dict[str, Any]] = []
    for row in rows:
        next_row = dict(row)
        run_id = str(next_row.get("run_id", ""))
        values = timing.get(run_id)
        if values is not None:
            wall_clock = values.get("wall_clock_sec")
            if wall_clock is not None:
                next_row["wall_clock_sec"] = wall_clock
                next_row["duration_sec"] = wall_clock
            training_time = values.get("training_time_sec")
            if training_time is not None:
                next_row["training_time_sec"] = training_time
            iterations = values.get("iterations")
            if iterations is not None:
                next_row["summary_iterations"] = iterations
                if next_row.get("checkpoint_epoch") in (None, ""):
                    next_row["checkpoint_epoch"] = iterations
        augmented.append(next_row)
    return augmented


def set_seed(seed: int, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def remove_file_handlers() -> None:
    root_logger = get_logger()
    for handler in list(root_logger.handlers):
        if isinstance(handler, logging.FileHandler):
            handler.close()
            root_logger.removeHandler(handler)


def prepare_config(rec: RunRecord, *, device: str, scratch_results: str, scratch_tb: str):
    cfg = OmegaConf.load(rec.config_path)
    cfg.config_path = rec.config_path.as_posix()
    if device == "cpu":
        resolved_device = "cpu"
    elif device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available.")
        resolved_device = "cuda"
    else:
        resolved_device = "cuda" if cfg.get("use_cuda", False) and torch.cuda.is_available() else "cpu"
    cfg.device = resolved_device
    cfg.use_cuda = resolved_device == "cuda"
    cfg.output = OmegaConf.merge(
        cfg.get("output", {}),
        {
            "results_dir": f"{scratch_results}/{rec.run_id}",
            "tb_dir": f"{scratch_tb}/{rec.run_id}",
        },
    )
    return cfg


def build_runner(rec: RunRecord, cfg: Any):
    ckpt_dir, epoch = find_final_checkpoint(rec.result_path)
    set_seed(rec.seed, cfg.device == "cuda")
    runner = Runners[rec.runner_type](config=cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    state = torch.load(ckpt_dir / "vi_model.pt", map_location=runner.device)
    runner.vi_model.load_state_dict(state)
    runner.vi_model.eval()
    return runner, ckpt_dir, epoch


def _sample_vi(runner, count: int, batch_size: int) -> torch.Tensor:
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, count, batch_size):
            current = min(batch_size, count - start)
            _, z = runner.vi_model.sampling(num=current)
            chunks.append(z.detach().cpu())
    return torch.cat(chunks, dim=0)


def _sample_target(runner, count: int, batch_size: int) -> torch.Tensor:
    if hasattr(runner.target_model, "sample"):
        chunks: list[torch.Tensor] = []
        with torch.no_grad():
            for start in range(0, count, batch_size):
                current = min(batch_size, count - start)
                chunks.append(runner.target_model.sample(current).detach().cpu())
        return torch.cat(chunks, dim=0)
    baseline = load_baseline_samples(runner.target_type)
    if baseline.shape[0] < count:
        indices = torch.randint(0, baseline.shape[0], (count,))
        return baseline[indices]
    return baseline[torch.randperm(baseline.shape[0])[:count]]


def constrained_w2(runner, width: float, cfg: Any) -> float:
    needed = int(cfg.accepted_samples)
    batch_size = int(cfg.sample_batch_size)
    max_draws = int(cfg.max_draws or 0)

    def collect(source: str) -> torch.Tensor:
        accepted: list[torch.Tensor] = []
        total = 0
        while sum(x.shape[0] for x in accepted) < needed:
            if max_draws > 0 and total >= max_draws:
                raise RuntimeError(f"Reached max_draws={max_draws} for {source} width={width}")
            if source == "vi":
                samples = _sample_vi(runner, batch_size, batch_size)
            else:
                samples = _sample_target(runner, batch_size, batch_size)
            total += samples.shape[0]
            mask = (samples.abs() < float(width)).all(dim=1)
            if mask.any():
                accepted.append(samples[mask].cpu())
        return torch.cat(accepted, dim=0)[:needed]

    vi = collect("vi")
    truth = collect("truth")
    return compute_sliced_wasserstein(
        vi,
        truth,
        num_projections=int(cfg.get("num_projections", 5000)),
        device=runner.device,
        p=2,
    )


def evaluate_w2_budgeted(runner, target: str, cfg: Any) -> float:
    num_samples = int(cfg.num_samples)
    vi = _sample_vi(runner, num_samples, num_samples)
    baseline = load_baseline_samples(target)
    if baseline.shape[0] >= num_samples:
        truth = baseline[torch.randperm(baseline.shape[0])[:num_samples]]
    else:
        truth = baseline[torch.randint(0, baseline.shape[0], (num_samples,))]
    return compute_sliced_wasserstein(
        vi,
        truth,
        num_projections=int(cfg.num_projections),
        device=runner.device,
        p=2,
    )


def evaluate_one_run(rec: RunRecord, cfg: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    eval_cfg = cfg.evaluation
    runner_cfg = prepare_config(
        rec,
        device=str(eval_cfg.device),
        scratch_results=str(cfg.campaign.scratch_results_dir),
        scratch_tb=str(cfg.campaign.scratch_tb_dir),
    )
    runner, ckpt_dir, ckpt_epoch = build_runner(rec, runner_cfg)
    metrics: dict[str, float] = {}
    errors: dict[str, str] = {}
    runtimes: dict[str, float] = {}
    raw_rows: list[dict[str, Any]] = []

    try:
        runner.n_elbo_batch_size = int(eval_cfg.elbo.batch_size)
        runner.n_elbo_batches = int(eval_cfg.elbo.num_batches)
        runner.n_elbo_z_samples = int(eval_cfg.elbo.num_z_samples)
        runner.n_w2_samples = int(eval_cfg.w2.num_samples)
        runner.n_w2_projections = int(eval_cfg.w2.num_projections)
        runner.n_bnn_samples = int(eval_cfg.bnn.num_samples)
        runner.n_expected_log_marginal_ref_samples = int(eval_cfg.langevin_kde_elm.num_ref_samples)
        runner.n_expected_log_marginal_model_samples = int(eval_cfg.langevin_kde_elm.num_model_samples)
        runner.n_expected_log_marginal_sample_batch_size = int(eval_cfg.langevin_kde_elm.sample_batch_size)
        runner.n_expected_log_marginal_dim_chunk = int(eval_cfg.langevin_kde_elm.dim_chunk)
        runner.n_expected_log_marginal_ref_chunk = int(eval_cfg.langevin_kde_elm.ref_chunk)
        runner.n_expected_log_marginal_model_chunk = int(eval_cfg.langevin_kde_elm.model_chunk)
        runner.expected_log_marginal_dtype = str(eval_cfg.langevin_kde_elm.dtype)
        runner._expected_log_marginal_reference_samples = None

        metric_plan: list[tuple[str, Any]] = []
        if rec.target in {"banana", "multimodal", "x_shaped", "student_uc", "8_gaussians", "Langevin_post"}:
            metric_plan.append(("elbo", lambda: runner.evaluate_elbo()[0]))
        if rec.target in {"banana", "multimodal", "x_shaped", "student_uc", "8_gaussians", "Langevin_post"}:
            metric_plan.append(("w2", lambda: evaluate_w2_budgeted(runner, rec.target, eval_cfg.w2)))
        if rec.target == "student_uc" and bool(eval_cfg.student_uc_constrained_w2.enabled):
            for width in eval_cfg.student_uc_constrained_w2.widths:
                metric_plan.append((
                    f"w2_edge_{int(width)}",
                    lambda w=float(width): constrained_w2(
                        runner,
                        w,
                        OmegaConf.merge(
                            eval_cfg.student_uc_constrained_w2,
                            {"num_projections": int(eval_cfg.w2.num_projections)},
                        ),
                    ),
                ))
        if rec.target == "Langevin_post" and bool(eval_cfg.langevin_kde_elm.enabled):
            metric_plan.append(("kde_elm", lambda: runner.evaluate_expected_log_marginal().value))
        if rec.target.startswith("Bnn_"):
            metric_plan.append(("rmse_nll", runner.evaluate_bnn_metrics))

        for metric_name, metric_fn in metric_plan:
            start = time.perf_counter()
            try:
                value = metric_fn()
                if metric_name == "rmse_nll":
                    rmse, test_llk = value
                    metrics["rmse"] = float(rmse)
                    metrics["nll"] = float(-test_llk)
                else:
                    metrics[metric_name] = float(value)
            except Exception as exc:
                errors[metric_name] = f"{type(exc).__name__}: {exc}"
                if bool(eval_cfg.fail_fast):
                    raise
            finally:
                runtimes[f"{metric_name}_runtime_sec"] = time.perf_counter() - start

        for name, value in metrics.items():
            raw_rows.append(
                {
                    "run_id": rec.run_id,
                    "seed": rec.seed,
                    "method": rec.method,
                    "target": rec.target,
                    "metric": name,
                    "value": value,
                    "checkpoint_epoch": ckpt_epoch,
                    "duration_sec": rec.duration_sec,
                }
            )

        summary = {
            "run_id": rec.run_id,
            "seed": rec.seed,
            "method": rec.method,
            "target": rec.target,
            "checkpoint_epoch": ckpt_epoch,
            "checkpoint_dir": ckpt_dir.as_posix(),
            "duration_sec": rec.duration_sec,
            "errors": json.dumps(errors, sort_keys=True),
            **metrics,
            **runtimes,
        }
        return summary, raw_rows
    finally:
        if hasattr(runner, "writer"):
            runner.writer.close()
        remove_file_handlers()
        del runner
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["target"], row["method"]), []).append(row)
    out: list[dict[str, Any]] = []
    excluded = {
        "seed",
        "duration_sec",
        "wall_clock_sec",
        "training_time_sec",
        "checkpoint_epoch",
        "summary_iterations",
    }
    metric_names = sorted(
        {
            key
            for row in rows
            for key, value in row.items()
            if key not in excluded and _finite_float(value) is not None
        }
    )
    for (target, method), items in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "target": target,
            "method": method,
            "seed_count": len({int(item["seed"]) for item in items}),
        }
        durations = [
            value
            for item in items
            for value in [_first_finite(item.get("wall_clock_sec"), item.get("duration_sec"))]
            if value is not None
        ]
        training_times = [
            value
            for item in items
            for value in [_finite_float(item.get("training_time_sec"))]
            if value is not None
        ]
        epochs = [
            value
            for item in items
            for value in [_first_finite(item.get("checkpoint_epoch"), item.get("summary_iterations"))]
            if value is not None
        ]
        if durations:
            summary["duration_sec_mean"] = float(np.mean(durations))
            summary["duration_sec_se"] = float(np.std(durations, ddof=1) / math.sqrt(len(durations))) if len(durations) > 1 else 0.0
            summary["wall_clock_sec_mean"] = summary["duration_sec_mean"]
            summary["wall_clock_sec_se"] = summary["duration_sec_se"]
        if training_times:
            summary["training_time_sec_mean"] = float(np.mean(training_times))
            summary["training_time_sec_se"] = float(np.std(training_times, ddof=1) / math.sqrt(len(training_times))) if len(training_times) > 1 else 0.0
        if epochs:
            summary["training_iterations_mean"] = float(np.mean(epochs))
        for metric in metric_names:
            values = [value for item in items for value in [_finite_float(item.get(metric))] if value is not None]
            if not values:
                continue
            summary[f"{metric}_mean"] = float(np.mean(values))
            summary[f"{metric}_se"] = float(np.std(values, ddof=1) / math.sqrt(len(values))) if len(values) > 1 else 0.0
            summary[f"{metric}_count"] = len(values)
        out.append(summary)
    return out


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate_runs(records: list[RunRecord], cfg: Any) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    out_dir = repo_path(str(cfg.campaign.output_dir))
    assert out_dir is not None
    run_csv = out_dir / "reevaluation_runs.csv"
    raw_jsonl = out_dir / "reevaluation_raw.jsonl"
    summary_csv = out_dir / "reevaluation_summary.csv"
    if run_csv.exists() and summary_csv.exists() and not bool(cfg.evaluation.overwrite):
        with run_csv.open("r", encoding="utf-8", newline="") as fh:
            run_rows = list(csv.DictReader(fh))
        run_rows = augment_run_rows_with_campaign_timing(run_rows, cfg)
        summary_rows = summarize(run_rows)
        write_csv(run_csv, run_rows)
        write_csv(summary_csv, summary_rows)
        return run_rows, summary_rows

    run_rows: list[dict[str, Any]] = []
    raw_rows: list[dict[str, Any]] = []
    for rec in records:
        logger.info(f"Re-evaluating {rec.run_id}")
        summary, raw = evaluate_one_run(rec, cfg)
        run_rows.append(summary)
        run_rows = augment_run_rows_with_campaign_timing(run_rows, cfg)
        raw_rows.extend(raw)
        write_csv(run_csv, run_rows)
        write_jsonl(raw_jsonl, raw_rows)
        write_csv(summary_csv, summarize(run_rows))
    summary_rows = summarize(run_rows)
    write_csv(run_csv, run_rows)
    write_jsonl(raw_jsonl, raw_rows)
    write_csv(summary_csv, summary_rows)
    return run_rows, summary_rows
