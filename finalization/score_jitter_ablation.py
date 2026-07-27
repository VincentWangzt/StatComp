"""Initialization-jitter ablation for posterior-HMC score references."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .config import REPO_ROOT, repo_path
from . import score_approximation as score_analysis


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "finalization"
    / "score_jitter_ablation.yaml"
)


def load_jitter_config(
    path: str | Path | None,
    overrides: list[str] | None = None,
) -> DictConfig:
    config_path = DEFAULT_CONFIG if path is None else Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    return cfg  # type: ignore[return-value]


def _read_seed_record(
    path: Path,
    fingerprint: str,
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("analysis_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Ablation fingerprint mismatch in {path}."
        )
    return payload


def seed_record_path(
    run_root: Path,
    spec: score_analysis.CellSpec,
) -> Path:
    return run_root / "seeds" / f"seed_{spec.record.seed}.json"


def pairwise_reference_l2(
    reference_means: dict[float, torch.Tensor],
) -> list[dict[str, float]]:
    """Calculate pairwise L2 gaps between jitter-specific reference means."""
    rows: list[dict[str, float]] = []
    scales = sorted(reference_means)
    for index, jitter_a in enumerate(scales):
        score_a = reference_means[jitter_a]
        energy_a = score_a.square().sum(dim=-1).mean()
        for jitter_b in scales[index + 1:]:
            score_b = reference_means[jitter_b]
            energy_b = score_b.square().sum(dim=-1).mean()
            l2 = (
                (score_a - score_b)
                .square()
                .sum(dim=-1)
                .mean()
            )
            average_energy = 0.5 * (energy_a + energy_b)
            rows.append({
                "jitter_a": float(jitter_a),
                "jitter_b": float(jitter_b),
                "reference_mean_l2": float(l2.item()),
                "reference_mean_rms": float(
                    torch.sqrt(l2).item()
                ),
                "relative_reference_mean_l2": float(
                    (
                        l2
                        / average_energy.clamp_min(
                            torch.finfo(l2.dtype).eps
                        )
                    ).item()
                ),
            })
    return rows


def _reference_kwargs(
    reference_cfg: DictConfig,
    *,
    jitter_scale: float,
) -> dict[str, Any]:
    return {
        "total_samples": int(reference_cfg.total_samples),
        "num_chains": int(reference_cfg.num_chains),
        "burn_in_steps": int(reference_cfg.burn_in_steps),
        "thinning": int(reference_cfg.thinning),
        "step_size": float(reference_cfg.step_size),
        "leapfrog_steps": int(reference_cfg.leapfrog_steps),
        "init_jitter_scale": float(jitter_scale),
        "adapt_step_size": bool(reference_cfg.adapt_step_size),
        "target_acceptance": float(
            reference_cfg.target_acceptance
        ),
        "adaptation_rate": float(reference_cfg.adaptation_rate),
        "min_step_size": float(reference_cfg.min_step_size),
        "max_step_size": float(reference_cfg.max_step_size),
        "divergence_threshold": float(
            reference_cfg.divergence_threshold
        ),
        "accumulator_dtype": score_analysis._accumulator_dtype(
            str(reference_cfg.accumulator_dtype)
        ),
    }


def evaluate_seed(
    runner: Any,
    spec: score_analysis.CellSpec,
    cfg: DictConfig,
    *,
    fingerprint: str,
) -> dict[str, Any]:
    """Evaluate every jitter scale with common forward and HMC randomness."""
    total_started = time.perf_counter()
    score_analysis._load_checkpoint(runner, spec)
    device = torch.device(runner.device)
    use_cuda = device.type == "cuda"
    forward_count = int(cfg.evaluation.forward_batch_size)

    forward_seed = score_analysis.stable_seed(spec.key, "forward")
    score_analysis.seed_everything(
        forward_seed,
        use_cuda=use_cuda,
    )
    generating_epsilon, z = runner.vi_model.sampling(
        num=forward_count
    )
    generating_epsilon = generating_epsilon.detach()
    z = z.detach()

    method_seed = score_analysis.stable_seed(spec.key, "method")
    score_analysis.seed_everything(
        method_seed,
        use_cuda=use_cuda,
    )
    score_analysis._sync(device)
    method_started = time.perf_counter()
    method_score, method_diagnostics = (
        score_analysis.method_native_score(
            runner,
            spec.record.method,
            z,
            generating_epsilon,
            aisivi_z_chunk_size=int(
                cfg.evaluation.aisivi_z_chunk_size
            ),
        )
    )
    score_analysis._sync(device)
    method_runtime = time.perf_counter() - method_started
    with torch.no_grad():
        target_score = runner.target_model.score(z).detach()

    reference_seed = score_analysis.stable_seed(
        spec.key,
        "reference_hmc",
    )
    jitter_metrics: list[dict[str, Any]] = []
    reference_means: dict[float, torch.Tensor] = {}
    for jitter_value in cfg.ablation.jitter_scales:
        jitter_scale = float(jitter_value)
        score_analysis.seed_everything(
            reference_seed,
            use_cuda=use_cuda,
        )
        score_analysis._sync(device)
        reference_started = time.perf_counter()
        reference_scores, diagnostics = (
            score_analysis.posterior_hmc_reference_scores(
                runner.vi_model,
                z,
                generating_epsilon,
                **_reference_kwargs(
                    cfg.evaluation.reference,
                    jitter_scale=jitter_scale,
                ),
            )
        )
        score_analysis._sync(device)
        reference_runtime = (
            time.perf_counter() - reference_started
        )
        quality_status, quality_issues = (
            score_analysis.assess_hmc_reference_quality(
                diagnostics,
                cfg.evaluation.reference.quality,
            )
        )
        metrics = score_analysis.compute_score_metrics(
            method_score,
            reference_scores,
            target_score,
        )
        reference_mean = reference_scores.mean(dim=0)
        reference_means[jitter_scale] = (
            reference_mean.detach().cpu()
        )
        jitter_metrics.append({
            "jitter_scale": jitter_scale,
            "reference_seed": reference_seed,
            "reference_runtime_sec": reference_runtime,
            "reference_quality_status": quality_status,
            "reference_quality_issues": quality_issues,
            "diagnostics": diagnostics,
            **metrics,
        })

    return {
        "analysis_fingerprint": fingerprint,
        "cell_key": spec.key,
        "run_id": spec.record.run_id,
        "method": spec.record.method.upper(),
        "target": spec.record.target,
        "seed": spec.record.seed,
        "progress": spec.progress,
        "epoch": spec.epoch,
        "checkpoint_dir": spec.checkpoint_dir.as_posix(),
        "forward_batch_size": forward_count,
        "forward_seed": forward_seed,
        "method_seed": method_seed,
        "method_runtime_sec": method_runtime,
        "method_diagnostics": method_diagnostics,
        "reference_num_chains": int(
            cfg.evaluation.reference.num_chains
        ),
        "reference_samples_per_chain": (
            int(cfg.evaluation.reference.total_samples)
            // int(cfg.evaluation.reference.num_chains)
        ),
        "reference_seed": reference_seed,
        "jitter_metrics": jitter_metrics,
        "pairwise_reference_l2": pairwise_reference_l2(
            reference_means
        ),
        "total_runtime_sec": time.perf_counter() - total_started,
        "completed_at": score_analysis.utc_now(),
    }


def _mean_sd(
    values: Iterable[float],
) -> tuple[float, float, int]:
    array = np.asarray(list(values), dtype=np.float64)
    if len(array) == 0:
        return float("nan"), float("nan"), 0
    return (
        float(array.mean()),
        float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        int(len(array)),
    )


def summarize_jitter_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = [
        "method_l2",
        "method_relative_l2",
        "method_target_l2",
        "reference_target_l2",
        "reference_internal_l2",
        "reference_mean_mcse_l2",
        "reference_mean_score_sq_norm",
        "reference_runtime_sec",
        "diagnostic_hmc_score_rhat_p95",
        "diagnostic_hmc_epsilon_rhat_p95",
        "diagnostic_hmc_post_burn_acceptance_rate",
        "diagnostic_hmc_post_burn_acceptance_min",
        "diagnostic_hmc_divergence_fraction",
        "diagnostic_hmc_final_step_size_median",
        "diagnostic_hmc_mean_squared_jump_distance",
    ]
    result: list[dict[str, Any]] = []
    for jitter_scale in sorted({
        float(row["jitter_scale"]) for row in rows
    }):
        selected = [
            row
            for row in rows
            if float(row["jitter_scale"]) == jitter_scale
        ]
        summary: dict[str, Any] = {
            "jitter_scale": jitter_scale,
            "n_seeds": len(selected),
            "quality_n_pass": sum(
                row["reference_quality_status"] == "pass"
                for row in selected
            ),
            "quality_n_warning": sum(
                row["reference_quality_status"] != "pass"
                for row in selected
            ),
        }
        for metric in metrics:
            mean, sd, count = _mean_sd(
                float(row[metric])
                for row in selected
                if row.get(metric) is not None
                and math.isfinite(float(row[metric]))
            )
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_n"] = count
        result.append(summary)
    return result


def summarize_pairwise_rows(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics = [
        "reference_mean_l2",
        "reference_mean_rms",
        "relative_reference_mean_l2",
    ]
    pairs = sorted({
        (float(row["jitter_a"]), float(row["jitter_b"]))
        for row in rows
    })
    result: list[dict[str, Any]] = []
    for jitter_a, jitter_b in pairs:
        selected = [
            row
            for row in rows
            if float(row["jitter_a"]) == jitter_a
            and float(row["jitter_b"]) == jitter_b
        ]
        summary: dict[str, Any] = {
            "jitter_a": jitter_a,
            "jitter_b": jitter_b,
            "n_seeds": len(selected),
        }
        for metric in metrics:
            mean, sd, count = _mean_sd(
                float(row[metric]) for row in selected
            )
            summary[f"{metric}_mean"] = mean
            summary[f"{metric}_sd"] = sd
            summary[f"{metric}_n"] = count
        result.append(summary)
    return result


def _flatten_jitter_row(
    seed_record: dict[str, Any],
    jitter_record: dict[str, Any],
) -> dict[str, Any]:
    row = {
        key: seed_record[key]
        for key in (
            "run_id",
            "method",
            "target",
            "seed",
            "epoch",
            "progress",
            "forward_batch_size",
            "reference_num_chains",
            "reference_samples_per_chain",
        )
    }
    row.update({
        key: value
        for key, value in jitter_record.items()
        if key not in {"diagnostics", "reference_quality_issues"}
    })
    row["reference_quality_issues"] = "; ".join(
        jitter_record["reference_quality_issues"]
    )
    for key, value in jitter_record["diagnostics"].items():
        row[f"diagnostic_{key}"] = value
    return row


def _metric_text(mean: float, sd: float) -> str:
    return f"{mean:.4e} ± {sd:.4e}"


def _write_markdown(
    report_dir: Path,
    jitter_summary: list[dict[str, Any]],
    pairwise_summary: list[dict[str, Any]],
) -> None:
    lines = [
        "# HMC Initialization-Jitter Ablation",
        "",
        "DSIVI on `8_gaussians`, epoch 10,000, seeds 42–44. "
        "Values are mean ± sample standard deviation across seeds.",
        "",
        "| Jitter | Method–HMC L2 | Internal L2 | Reference-mean MC MSE | "
        "Method–target L2 | HMC–target L2 | Score R-hat p95 | Quality |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in jitter_summary:
        lines.append(
            "| {jitter:.1e} | {method} | {internal} | {mcse} | "
            "{method_target} | {target} | {rhat} | "
            "{passed}/{total} |".format(
                jitter=float(row["jitter_scale"]),
                method=_metric_text(
                    float(row["method_l2_mean"]),
                    float(row["method_l2_sd"]),
                ),
                internal=_metric_text(
                    float(row["reference_internal_l2_mean"]),
                    float(row["reference_internal_l2_sd"]),
                ),
                mcse=_metric_text(
                    float(row["reference_mean_mcse_l2_mean"]),
                    float(row["reference_mean_mcse_l2_sd"]),
                ),
                method_target=_metric_text(
                    float(row["method_target_l2_mean"]),
                    float(row["method_target_l2_sd"]),
                ),
                target=_metric_text(
                    float(row["reference_target_l2_mean"]),
                    float(row["reference_target_l2_sd"]),
                ),
                rhat=_metric_text(
                    float(
                        row[
                            "diagnostic_hmc_score_rhat_p95_mean"
                        ]
                    ),
                    float(
                        row[
                            "diagnostic_hmc_score_rhat_p95_sd"
                        ]
                    ),
                ),
                passed=int(row["quality_n_pass"]),
                total=int(row["n_seeds"]),
            )
        )
    lines.extend([
        "",
        "`Internal L2` is the average squared disagreement of each "
        "chain-mean score from the ten-chain grand mean. "
        "`Reference-mean MC MSE` is the heuristic `Internal L2 / 9`; "
        "it only has a literal MCSE interpretation for independent, "
        "equally distributed chain means.",
        "",
        "## Pairwise HMC-reference sensitivity",
        "",
        "| Jitter A | Jitter B | Reference-mean L2 | RMS gap | "
        "Relative L2 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in pairwise_summary:
        lines.append(
            "| {a:.1e} | {b:.1e} | {l2} | {rms} | {relative} |".format(
                a=float(row["jitter_a"]),
                b=float(row["jitter_b"]),
                l2=_metric_text(
                    float(row["reference_mean_l2_mean"]),
                    float(row["reference_mean_l2_sd"]),
                ),
                rms=_metric_text(
                    float(row["reference_mean_rms_mean"]),
                    float(row["reference_mean_rms_sd"]),
                ),
                relative=_metric_text(
                    float(
                        row["relative_reference_mean_l2_mean"]
                    ),
                    float(
                        row["relative_reference_mean_l2_sd"]
                    ),
                ),
            )
        )
    (report_dir / "score_jitter_ablation.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _render_plot(
    report_dir: Path,
    jitter_summary: list[dict[str, Any]],
    pairwise_summary: list[dict[str, Any]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [
        f"{float(row['jitter_scale']):.0e}"
        if float(row["jitter_scale"]) > 0
        else "0"
        for row in jitter_summary
    ]
    x = np.arange(len(labels), dtype=np.float64)
    panels = [
        ("method_l2", "Method–HMC L2", True),
        ("reference_internal_l2", "Internal L2", True),
        (
            "diagnostic_hmc_score_rhat_p95",
            "Score R-hat p95",
            False,
        ),
    ]
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(9.0, 6.8),
        squeeze=False,
    )
    for axis, (metric, title, log_scale) in zip(
        axes.reshape(-1)[:3],
        panels,
        strict=True,
    ):
        mean = np.asarray([
            float(row[f"{metric}_mean"])
            for row in jitter_summary
        ])
        sd = np.asarray([
            float(row[f"{metric}_sd"])
            for row in jitter_summary
        ])
        axis.errorbar(
            x,
            mean,
            yerr=sd,
            marker="o",
            linewidth=1.8,
            capsize=3,
            color="#2f6f9f",
        )
        if log_scale:
            axis.set_yscale("log")
        axis.set_title(title)
        axis.set_xticks(x, labels)
        axis.set_xlabel("Initialization jitter scale")
        axis.grid(True, which="both", alpha=0.25)
    axes[1, 0].axhline(
        1.1,
        color="#555555",
        linestyle="--",
        linewidth=1.0,
    )

    zero_pairs = sorted(
        (
            row
            for row in pairwise_summary
            if float(row["jitter_a"]) == 0.0
        ),
        key=lambda row: float(row["jitter_b"]),
    )
    axis = axes[1, 1]
    pair_x = np.arange(len(zero_pairs), dtype=np.float64)
    pair_mean = np.asarray([
        float(row["reference_mean_l2_mean"])
        for row in zero_pairs
    ])
    pair_sd = np.asarray([
        float(row["reference_mean_l2_sd"])
        for row in zero_pairs
    ])
    axis.errorbar(
        pair_x,
        pair_mean,
        yerr=pair_sd,
        marker="o",
        linewidth=1.8,
        capsize=3,
        color="#b45f38",
    )
    axis.set_yscale("log")
    axis.set_xticks(
        pair_x,
        [
            f"{float(row['jitter_b']):.0e}"
            for row in zero_pairs
        ],
    )
    axis.set_xlabel("Compared with zero jitter")
    axis.set_title("Pairwise HMC-reference L2")
    axis.grid(True, which="both", alpha=0.25)
    figure.tight_layout()
    png = report_dir / "score_jitter_ablation.png"
    pdf = report_dir / "score_jitter_ablation.pdf"
    figure.savefig(png, dpi=300, bbox_inches="tight")
    figure.savefig(pdf, bbox_inches="tight")
    plt.close(figure)
    return [png, pdf]


def aggregate_results(
    cfg: DictConfig,
    specs: list[score_analysis.CellSpec],
    *,
    fingerprint: str,
    require_complete: bool = True,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    report_dir = repo_path(str(cfg.output.report_dir))
    assert runtime_root is not None and report_dir is not None
    run_root = runtime_root / fingerprint[:16]
    seed_records: list[dict[str, Any]] = []
    missing: list[int] = []
    for spec in specs:
        path = seed_record_path(run_root, spec)
        if not path.is_file():
            missing.append(spec.record.seed)
            continue
        seed_records.append(_read_seed_record(path, fingerprint))
    if require_complete and missing:
        raise RuntimeError(
            f"Missing jitter-ablation seeds: {missing}"
        )

    jitter_rows: list[dict[str, Any]] = []
    pairwise_rows: list[dict[str, Any]] = []
    for record in seed_records:
        for jitter_record in record["jitter_metrics"]:
            jitter_rows.append(
                _flatten_jitter_row(record, jitter_record)
            )
        for pair_record in record["pairwise_reference_l2"]:
            pairwise_rows.append({
                "run_id": record["run_id"],
                "method": record["method"],
                "target": record["target"],
                "seed": record["seed"],
                "epoch": record["epoch"],
                **pair_record,
            })
    jitter_summary = summarize_jitter_rows(jitter_rows)
    pairwise_summary = summarize_pairwise_rows(pairwise_rows)
    if require_complete:
        expected_jitters = len(cfg.ablation.jitter_scales)
        expected_pairs = (
            expected_jitters * (expected_jitters - 1) // 2
        )
        if len(seed_records) != len(cfg.selection.seeds):
            raise RuntimeError("Unexpected seed-record count.")
        if len(jitter_summary) != expected_jitters:
            raise RuntimeError("Unexpected jitter-summary count.")
        if len(pairwise_summary) != expected_pairs:
            raise RuntimeError("Unexpected pairwise-summary count.")
        if any(
            int(row["n_seeds"]) != len(cfg.selection.seeds)
            for row in jitter_summary + pairwise_summary
        ):
            raise RuntimeError(
                "A summary row has an incomplete seed count."
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    score_analysis._write_csv(
        report_dir / "jitter_metrics.csv",
        jitter_rows,
    )
    score_analysis._write_csv(
        report_dir / "pairwise_reference_l2.csv",
        pairwise_rows,
    )
    score_analysis._write_csv(
        report_dir / "jitter_summary.csv",
        jitter_summary,
    )
    score_analysis._write_csv(
        report_dir / "pairwise_summary.csv",
        pairwise_summary,
    )
    _write_markdown(
        report_dir,
        jitter_summary,
        pairwise_summary,
    )
    figure_paths = _render_plot(
        report_dir,
        jitter_summary,
        pairwise_summary,
    )
    metadata = {
        "analysis_fingerprint": fingerprint,
        "git_commit": score_analysis._git_commit(),
        "generated_at": score_analysis.utc_now(),
        "completed_seeds": len(seed_records),
        "jitter_rows": len(jitter_rows),
        "pairwise_rows": len(pairwise_rows),
        "jitter_summary_rows": len(jitter_summary),
        "pairwise_summary_rows": len(pairwise_summary),
        "quality_warnings": sum(
            row["reference_quality_status"] != "pass"
            for row in jitter_rows
        ),
        "figures": [
            path.relative_to(report_dir).as_posix()
            for path in figure_paths
        ],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    score_analysis.atomic_write_json(
        report_dir / "run_metadata.json",
        metadata,
    )
    return jitter_summary, pairwise_summary


def run_ablation(
    cfg: DictConfig,
    *,
    limit: int | None = None,
    resume: bool = True,
    aggregate_only: bool = False,
) -> tuple[int, int]:
    fingerprint = score_analysis.config_fingerprint(cfg)
    specs = score_analysis.build_cell_specs(cfg)
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    assert runtime_root is not None
    run_root = runtime_root / fingerprint[:16]
    if aggregate_only:
        jitter_summary, pairwise_summary = aggregate_results(
            cfg,
            specs,
            fingerprint=fingerprint,
            require_complete=True,
        )
        return len(jitter_summary), len(pairwise_summary)

    if (
        str(cfg.evaluation.device) == "cuda"
        and not torch.cuda.is_available()
    ):
        raise RuntimeError(
            "CUDA is required by the production ablation."
        )
    pending = []
    for spec in specs:
        path = seed_record_path(run_root, spec)
        if resume and path.is_file():
            payload = _read_seed_record(path, fingerprint)
            if payload.get("cell_key") == spec.key:
                continue
        pending.append(spec)
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        pending = pending[:limit]

    completed_now = 0
    for spec in pending:
        runner: Any | None = None
        try:
            runner = score_analysis._build_runner(spec.record, cfg)
            started = time.perf_counter()
            record = evaluate_seed(
                runner,
                spec,
                cfg,
                fingerprint=fingerprint,
            )
            score_analysis.atomic_write_json(
                seed_record_path(run_root, spec),
                record,
            )
            completed_now += 1
            metrics_text = ", ".join(
                (
                    f"j={float(row['jitter_scale']):.1e}:"
                    f"{float(row['method_l2']):.3e}"
                )
                for row in record["jitter_metrics"]
            )
            print(
                f"[{completed_now}/{len(pending)}] "
                f"seed={spec.record.seed}, epoch={spec.epoch}: "
                f"method_l2=[{metrics_text}], "
                f"runtime={time.perf_counter() - started:.1f}s",
                flush=True,
            )
        finally:
            score_analysis._release_runner(runner)

    if limit is None:
        jitter_summary, pairwise_summary = aggregate_results(
            cfg,
            specs,
            fingerprint=fingerprint,
            require_complete=True,
        )
        return len(jitter_summary), len(pairwise_summary)
    return completed_now, 0
