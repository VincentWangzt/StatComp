"""Inference-only latency benchmark for native SIVI score estimators.

The timed region contains only the work needed to estimate
``nabla_z log q_phi(z)``:

* SIVI samples prior auxiliaries, forms the mixture log density, and
  differentiates it with respect to ``z``.
* UIVI runs its posterior HMC sampler and averages conditional scores.
* AISIVI samples its already-trained reverse flow, forms the
  importance-weighted mixture log density, and differentiates it.
* DSIVI evaluates its already-trained score network once.

Checkpoint loading, input-pair generation, warm-up calls, model fitting, and
optimizer work are deliberately outside the timed region.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch
from omegaconf import DictConfig, OmegaConf

from runner.runners import Runners

from .artifacts import (
    RunRecord,
    completed_runs,
    find_final_checkpoint,
    load_manifest,
    select_runs,
)
from .config import REPO_ROOT, repo_path
from .runner_eval import prepare_config, remove_file_handlers, set_seed


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "finalization"
    / "score_estimator_timing_x_shaped.yaml"
)
SUPPORTED_METHODS = ("SIVI", "UIVI", "AISIVI", "DSIVI")
ScoreEstimator = Callable[
    [Any, torch.Tensor, torch.Tensor],
    torch.Tensor,
]


def load_timing_config(
    path: str | Path | None,
    overrides: list[str] | None = None,
) -> DictConfig:
    config_path = DEFAULT_CONFIG if path is None else Path(path)
    if not config_path.is_absolute():
        config_path = REPO_ROOT / config_path
    cfg = OmegaConf.load(config_path)
    if overrides:
        cfg = OmegaConf.merge(
            cfg,
            OmegaConf.from_dotlist(overrides),
        )
    return cfg  # type: ignore[return-value]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()
    except ValueError:
        return path.resolve().as_posix()


def stable_seed(*parts: object) -> int:
    payload = "::".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(
        hashlib.sha256(payload).digest()[:8],
        byteorder="little",
    ) % (2**31)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _checkpoint_dir(
    record: RunRecord,
    checkpoint_epoch: int | str,
) -> tuple[Path, int]:
    if str(checkpoint_epoch).lower() == "final":
        return find_final_checkpoint(record.result_path)
    epoch = int(checkpoint_epoch)
    checkpoint_dir = (
        record.result_path / "checkpoints" / f"epoch_{epoch}"
    )
    if not (checkpoint_dir / "vi_model.pt").is_file():
        raise FileNotFoundError(
            f"Missing epoch-{epoch} checkpoint for {record.run_id}: "
            f"{checkpoint_dir}"
        )
    return checkpoint_dir, epoch


def select_timing_records(cfg: DictConfig) -> list[RunRecord]:
    methods = [str(value).upper() for value in cfg.selection.methods]
    unsupported = sorted(set(methods) - set(SUPPORTED_METHODS))
    if unsupported:
        raise ValueError(
            "Unsupported timing methods: " + ", ".join(unsupported)
        )
    targets = [str(value) for value in cfg.selection.targets]
    seeds = [int(value) for value in cfg.selection.seeds]
    selected = select_runs(
        completed_runs(load_manifest(cfg.campaign.manifest_path)),
        methods=methods,
        targets=targets,
        seeds=seeds,
    )
    keys: dict[tuple[str, str, int], list[RunRecord]] = {}
    for record in selected:
        key = (record.method.upper(), record.target, record.seed)
        keys.setdefault(key, []).append(record)
    expected = {
        (method, target, seed)
        for method in methods
        for target in targets
        for seed in seeds
    }
    missing = sorted(expected - set(keys))
    duplicated = sorted(key for key, values in keys.items() if len(values) > 1)
    if missing:
        raise RuntimeError(f"Missing manifest timing cells: {missing}")
    if duplicated:
        raise RuntimeError(f"Duplicate manifest timing cells: {duplicated}")
    method_order = {method: index for index, method in enumerate(methods)}
    return sorted(
        (values[0] for values in keys.values()),
        key=lambda record: (
            record.target,
            record.seed,
            method_order[record.method.upper()],
        ),
    )


def build_timing_runner(
    cfg: DictConfig,
    record: RunRecord,
) -> tuple[Any, Path, int, dict[str, str | None]]:
    runner_cfg = prepare_config(
        record,
        device=str(cfg.evaluation.device),
        scratch_results=str(cfg.output.scratch_results_dir),
        scratch_tb=str(cfg.output.scratch_tb_dir),
    )
    set_seed(record.seed, runner_cfg.device == "cuda")
    runner = Runners[record.runner_type](config=runner_cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()

    checkpoint_dir, epoch = _checkpoint_dir(
        record,
        cfg.selection.checkpoint_epoch,
    )
    vi_path = checkpoint_dir / "vi_model.pt"
    vi_state = torch.load(
        vi_path,
        map_location=runner.device,
        weights_only=True,
    )
    runner.vi_model.load_state_dict(vi_state)
    runner.vi_model.eval()
    for parameter in runner.vi_model.parameters():
        parameter.requires_grad_(False)

    hashes: dict[str, str | None] = {
        "vi_model_sha256": file_sha256(vi_path),
        "reverse_model_sha256": None,
    }
    method = record.method.upper()
    if method in {"AISIVI", "DSIVI"}:
        reverse_path = checkpoint_dir / "reverse_model.pt"
        if not reverse_path.is_file():
            raise FileNotFoundError(reverse_path)
        reverse_state = torch.load(
            reverse_path,
            map_location=runner.device,
            weights_only=True,
        )
        runner.reverse_model.load_state_dict(reverse_state)
        runner.reverse_model.eval()
        for parameter in runner.reverse_model.parameters():
            parameter.requires_grad_(False)
        hashes["reverse_model_sha256"] = file_sha256(reverse_path)
    return runner, checkpoint_dir, epoch, hashes


def sivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the SIVI mixture score through autograd."""

    auxiliary_count = int(runner.training_reverse_sample_num)
    auxiliary_epsilon = runner.vi_model.sample_epsilon(
        num=auxiliary_count,
    )
    epsilon_aux = auxiliary_epsilon.unsqueeze(0).repeat(
        z.shape[0],
        1,
        1,
    )
    epsilon_aux = torch.cat(
        [epsilon_aux, generating_epsilon.unsqueeze(1)],
        dim=1,
    )
    z_aux = z.detach().unsqueeze(1).repeat(
        1,
        auxiliary_count + 1,
        1,
    )
    z_aux.requires_grad_(True)
    log_terms = runner.vi_model.logp(z_aux, epsilon_aux)
    log_mixture = torch.logsumexp(log_terms, dim=1) - math.log(
        auxiliary_count + 1
    )
    component_gradient = torch.autograd.grad(
        log_mixture.sum(),
        z_aux,
        create_graph=False,
    )[0]
    return component_gradient.sum(dim=1).detach()


def uivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the UIVI score with its native posterior HMC."""

    z_aux, epsilon_aux, _ = runner.sample_epsilon_hmc(
        z,
        eps_init=generating_epsilon,
        num_samples=int(runner.training_reverse_sample_num),
        burn_in_steps=int(runner.hmc_burn_in_steps),
        step_size=float(runner.hmc_step_size),
        leapfrog_steps=int(runner.hmc_leapfrog_steps),
    )
    with torch.no_grad():
        return runner.vi_model.score(z_aux, epsilon_aux).mean(
            dim=1
        ).detach()


def aisivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the AISIVI score using a loaded reverse-flow checkpoint."""

    del generating_epsilon
    sample_count = int(runner.training_reverse_sample_num)
    with torch.no_grad():
        z_aux, epsilon_aux, log_q_reverse = (
            runner.reverse_model.sample(
                z,
                num_samples=sample_count,
            )
        )
        log_importance = (
            runner.vi_model.log_q_epsilon(epsilon_aux) - log_q_reverse
        ).clamp(max=10.0)
    z_aux.requires_grad_(True)
    log_terms = runner.vi_model.logp(
        z_aux,
        epsilon_aux,
    ) + log_importance
    log_mixture = torch.logsumexp(log_terms, dim=1) - math.log(
        sample_count
    )
    component_gradient = torch.autograd.grad(
        log_mixture.sum(),
        z_aux,
        create_graph=False,
    )[0]
    score = component_gradient.sum(dim=1).detach()
    if bool(runner.normalize_reverse_score):
        score = score - score.mean(dim=0, keepdim=True)
    return score


def dsivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> torch.Tensor:
    """Estimate the DSIVI/DIVI score with one score-network forward pass."""

    del generating_epsilon
    with torch.no_grad():
        score = runner.reverse_model.score(z).detach()
        if bool(runner.normalize_reverse_score):
            score = score - score.mean(dim=0, keepdim=True)
    return score


ESTIMATORS: dict[str, ScoreEstimator] = {
    "SIVI": sivi_score,
    "UIVI": uivi_score,
    "AISIVI": aisivi_score,
    "DSIVI": dsivi_score,
}


def estimator_metadata(runner: Any, method: str) -> dict[str, Any]:
    normalized = method.upper()
    if normalized == "SIVI":
        auxiliaries = int(runner.training_reverse_sample_num) + 1
        return {
            "estimator": (
                "prior sampling + mixture logsumexp + autograd score"
            ),
            "native_auxiliary_samples": auxiliaries,
            "hmc_burn_in_steps": None,
            "hmc_leapfrog_steps": None,
        }
    if normalized == "UIVI":
        return {
            "estimator": "posterior HMC + conditional-score mean",
            "native_auxiliary_samples": int(
                runner.training_reverse_sample_num
            ),
            "hmc_burn_in_steps": int(runner.hmc_burn_in_steps),
            "hmc_leapfrog_steps": int(runner.hmc_leapfrog_steps),
        }
    if normalized == "AISIVI":
        return {
            "estimator": (
                "reverse-flow sampling + importance mixture + "
                "autograd score"
            ),
            "native_auxiliary_samples": int(
                runner.training_reverse_sample_num
            ),
            "hmc_burn_in_steps": None,
            "hmc_leapfrog_steps": None,
        }
    if normalized == "DSIVI":
        return {
            "estimator": "score-network forward pass",
            "native_auxiliary_samples": 0,
            "hmc_burn_in_steps": None,
            "hmc_leapfrog_steps": None,
        }
    raise ValueError(f"Unsupported method: {method}")


def input_bank(
    runner: Any,
    *,
    batch_size: int,
    calls: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        epsilon, z = runner.vi_model.sampling(
            num=batch_size * calls,
        )
    return (
        epsilon.reshape(calls, batch_size, -1),
        z.reshape(calls, batch_size, -1),
    )


def benchmark_estimator(
    runner: Any,
    estimator: ScoreEstimator,
    *,
    batch_size: int,
    warmup_calls: int,
    timed_calls: int,
) -> tuple[list[float], int]:
    if batch_size < 1 or warmup_calls < 0 or timed_calls < 2:
        raise ValueError(
            "batch_size must be positive, warmup_calls non-negative, "
            "and timed_calls at least two."
        )
    device = torch.device(runner.device)
    epsilon_bank, z_bank = input_bank(
        runner,
        batch_size=batch_size,
        calls=warmup_calls + timed_calls,
    )

    for index in range(warmup_calls):
        estimator(runner, z_bank[index], epsilon_bank[index])
    _synchronize(device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    elapsed_ms: list[float] = []
    outputs: list[torch.Tensor] = []
    for offset in range(timed_calls):
        index = warmup_calls + offset
        _synchronize(device)
        started_ns = time.perf_counter_ns()
        score = estimator(
            runner,
            z_bank[index],
            epsilon_bank[index],
        )
        _synchronize(device)
        elapsed_ms.append(
            (time.perf_counter_ns() - started_ns) / 1_000_000.0
        )
        outputs.append(score)

    stacked = torch.stack(outputs)
    if stacked.shape != (
        timed_calls,
        batch_size,
        int(runner.z_dim),
    ):
        raise RuntimeError(
            f"Unexpected score shape {tuple(stacked.shape)}."
        )
    if not bool(torch.isfinite(stacked).all().item()):
        raise FloatingPointError(
            "A timed score estimate contained a non-finite value."
        )
    peak_memory_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else 0
    )
    return elapsed_ms, peak_memory_bytes


def summarize_timings(values: list[float]) -> dict[str, float]:
    ordered = sorted(float(value) for value in values)
    return {
        "latency_ms_mean": statistics.fmean(ordered),
        "latency_ms_sd": statistics.stdev(ordered),
        "latency_ms_median": statistics.median(ordered),
        "latency_ms_min": ordered[0],
        "latency_ms_max": ordered[-1],
    }


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_cuda_runtime": torch.version.cuda,
        "device": str(device),
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        metadata.update({
            "gpu_name": properties.name,
            "gpu_total_memory_bytes": int(
                properties.total_memory
            ),
            "gpu_compute_capability": (
                f"{properties.major}.{properties.minor}"
            ),
        })
    return metadata


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _latency_text(mean: float, sd: float) -> str:
    decimals = 4 if mean < 1.0 else 3
    return f"{mean:.{decimals}f} ± {sd:.{decimals}f}"


def write_markdown_report(
    path: Path,
    *,
    summary_rows: list[dict[str, Any]],
    environment: dict[str, Any],
    warmup_calls: int,
    timed_calls: int,
) -> None:
    lines = [
        "# Native score-estimator latency",
        "",
        (
            "Latency is synchronized end-to-end wall time on "
            f"{environment.get('gpu_name', environment['device'])}. "
            f"Each cell uses {warmup_calls} untimed warm-up calls followed "
            f"by {timed_calls} timed calls and reports mean ± sample "
            "standard deviation."
        ),
        (
            "Checkpoint loading, input `(z, epsilon)` generation, optimizer "
            "work, model fitting, diagnostics, and logging are excluded. "
            "AISIVI loads its trained reverse-flow checkpoint; it is not "
            "trained or refit by this benchmark."
        ),
        "",
    ]
    methods = list(dict.fromkeys(row["method"] for row in summary_rows))
    for batch_size in sorted({
        int(row["batch_size"]) for row in summary_rows
    }):
        lines.extend([
            f"## Batch size {batch_size}",
            "",
            "| Method | Mean ± SD (ms) | ms per z | z / second |",
            "|---|---:|---:|---:|",
        ])
        lookup = {
            row["method"]: row
            for row in summary_rows
            if int(row["batch_size"]) == batch_size
        }
        for method in methods:
            row = lookup[method]
            label = "DIVI (DSIVI)" if method == "DSIVI" else method
            lines.append(
                f"| {label} | "
                f"{_latency_text(row['latency_ms_mean'], row['latency_ms_sd'])} | "
                f"{row['latency_ms_per_z']:.6f} | "
                f"{row['throughput_z_per_sec']:.1f} |"
            )
        lines.append("")

    lines.extend([
        "## Estimator boundaries and checkpoints",
        "",
        (
            "| Method | Timed estimator | Native auxiliaries | "
            "Checkpoint |"
        ),
        "|---|---|---:|---|",
    ])
    first_batch = min(int(row["batch_size"]) for row in summary_rows)
    for row in summary_rows:
        if int(row["batch_size"]) != first_batch:
            continue
        label = "DIVI (DSIVI)" if row["method"] == "DSIVI" else row["method"]
        lines.append(
            f"| {label} | {row['estimator']} | "
            f"{int(row['native_auxiliary_samples'])} | "
            f"`{row['checkpoint_dir']}` |"
        )
    uivi_row = next(
        row for row in summary_rows if row["method"] == "UIVI"
    )
    lines.extend([
        "",
        (
            "UIVI uses "
            f"{int(uivi_row['hmc_burn_in_steps'])} "
            "burn-in transitions, "
            f"{int(uivi_row['native_auxiliary_samples'])} "
            "retained transitions, and "
            f"{int(uivi_row['hmc_leapfrog_steps'])} "
            "leapfrog steps per transition."
        ),
        "",
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def run_benchmark(cfg: DictConfig) -> list[dict[str, Any]]:
    warmup_calls = int(cfg.evaluation.warmup_calls)
    timed_calls = int(cfg.evaluation.timed_calls)
    batch_sizes = [int(value) for value in cfg.evaluation.batch_sizes]
    benchmark_seed = int(cfg.evaluation.seed)
    records = select_timing_records(cfg)

    summary_rows: list[dict[str, Any]] = []
    repetition_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    environment: dict[str, Any] | None = None

    for record in records:
        method = record.method.upper()
        runner, checkpoint_dir, epoch, hashes = build_timing_runner(
            cfg,
            record,
        )
        device = torch.device(runner.device)
        if environment is None:
            environment = environment_metadata(device)
        metadata = estimator_metadata(runner, method)
        checkpoint_rows.append({
            "method": method,
            "target": record.target,
            "seed": record.seed,
            "run_id": record.run_id,
            "epoch": epoch,
            "checkpoint_dir": relative_path(checkpoint_dir),
            **hashes,
        })

        for batch_size in batch_sizes:
            cell_seed = stable_seed(
                benchmark_seed,
                record.run_id,
                epoch,
                batch_size,
            )
            set_seed(cell_seed, device.type == "cuda")
            elapsed_ms, peak_memory_bytes = benchmark_estimator(
                runner,
                ESTIMATORS[method],
                batch_size=batch_size,
                warmup_calls=warmup_calls,
                timed_calls=timed_calls,
            )
            timing_summary = summarize_timings(elapsed_ms)
            mean_ms = timing_summary["latency_ms_mean"]
            row = {
                "method": method,
                "target": record.target,
                "seed": record.seed,
                "run_id": record.run_id,
                "epoch": epoch,
                "checkpoint_dir": relative_path(checkpoint_dir),
                "batch_size": batch_size,
                "warmup_calls": warmup_calls,
                "timed_calls": timed_calls,
                "timing_seed": cell_seed,
                **timing_summary,
                "latency_ms_per_z": mean_ms / batch_size,
                "throughput_z_per_sec": 1000.0 * batch_size / mean_ms,
                "peak_memory_bytes": peak_memory_bytes,
                **metadata,
            }
            summary_rows.append(row)
            for repetition, latency_ms in enumerate(elapsed_ms, start=1):
                repetition_rows.append({
                    "method": method,
                    "target": record.target,
                    "seed": record.seed,
                    "epoch": epoch,
                    "batch_size": batch_size,
                    "repetition": repetition,
                    "latency_ms": latency_ms,
                })
            print(
                f"{method} batch={batch_size}: "
                f"{_latency_text(row['latency_ms_mean'], row['latency_ms_sd'])} "
                "ms",
                flush=True,
            )
        del runner
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if environment is None:
        raise RuntimeError("No timing records were selected.")
    output_dir = repo_path(str(cfg.output.report_dir))
    if output_dir is None:
        raise ValueError("output.report_dir is required.")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "timing_summary.csv", summary_rows)
    _write_csv(output_dir / "timing_repetitions.csv", repetition_rows)
    write_markdown_report(
        output_dir / "score_estimator_timing.md",
        summary_rows=summary_rows,
        environment=environment,
        warmup_calls=warmup_calls,
        timed_calls=timed_calls,
    )
    run_metadata = {
        "generated_at": utc_now(),
        "git_commit": git_commit(),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "environment": environment,
        "checkpoints": checkpoint_rows,
        "summary": summary_rows,
        "training_or_refitting_performed": False,
        "timing_definition": (
            "Synchronized end-to-end wall-clock latency; checkpoint loading, "
            "input generation, warm-up, diagnostics, logging, fitting, and "
            "optimizer work excluded."
        ),
    }
    (output_dir / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"report_dir={output_dir}", flush=True)
    return summary_rows


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark inference-only native score estimators.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="OmegaConf dotlist overrides.",
    )
    args = parser.parse_args(argv)
    run_benchmark(
        load_timing_config(args.config, args.overrides),
    )


if __name__ == "__main__":
    main()
