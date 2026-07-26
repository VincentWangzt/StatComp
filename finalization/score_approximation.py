"""Checkpoint-based analysis of marginal-score approximations.

The analysis compares the score used by each training method with a high-budget
Monte Carlo estimate of the marginal score of the checkpointed variational
distribution.  It is intentionally post-hoc: checkpoints are loaded read-only
and no optimizer, scheduler, or reverse-model updates are performed.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from runner.runners import Runners

from .artifacts import (
    RunRecord,
    completed_runs,
    find_all_checkpoints,
    load_manifest,
    select_runs,
)
from .config import REPO_ROOT, repo_path
from .runner_eval import prepare_config, remove_file_handlers, set_seed


DEFAULT_CONFIG = (
    REPO_ROOT / "configs" / "finalization" / "score_approximation.yaml"
)
NONFINITE_SCORE_MESSAGE = "Score calculation produced non-finite values."


@dataclass(frozen=True)
class CellSpec:
    record: RunRecord
    progress: float
    epoch: int
    checkpoint_dir: Path

    @property
    def key(self) -> str:
        return (
            f"{self.record.run_id}|{self.record.method}|{self.record.target}|"
            f"{self.record.seed}|{self.epoch}"
        )


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_score_config(
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


def config_fingerprint(cfg: DictConfig) -> str:
    payload = OmegaConf.to_container(cfg, resolve=True)
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def stable_seed(*parts: object) -> int:
    encoded = "|".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(encoded).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1)


def seed_everything(seed: int, *, use_cuda: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _progress_epoch(total_epochs: int, progress: float) -> int:
    if not 0.0 < progress <= 1.0:
        raise ValueError(f"Checkpoint progress must be in (0, 1], got {progress}")
    raw_epoch = total_epochs * progress
    epoch = int(round(raw_epoch))
    if not math.isclose(raw_epoch, epoch, abs_tol=1.0e-8):
        raise ValueError(
            f"Progress {progress} does not select an integer epoch from "
            f"training horizon {total_epochs}."
        )
    return epoch


def select_progress_checkpoints(
    checkpoints: Iterable[tuple[int, Path]],
    *,
    total_epochs: int,
    progresses: Iterable[float],
) -> list[tuple[float, int, Path]]:
    by_epoch = {
        int(epoch): Path(model_path).parent
        for epoch, model_path in checkpoints
    }
    selected: list[tuple[float, int, Path]] = []
    for value in progresses:
        progress = float(value)
        epoch = _progress_epoch(total_epochs, progress)
        checkpoint_dir = by_epoch.get(epoch)
        if checkpoint_dir is None:
            raise FileNotFoundError(
                f"Required checkpoint epoch_{epoch} is missing; "
                f"available epochs are {sorted(by_epoch)}"
            )
        selected.append((progress, epoch, checkpoint_dir))
    return selected


def build_cell_specs(cfg: DictConfig) -> list[CellSpec]:
    manifest = load_manifest(str(cfg.campaign.manifest_path))
    records = completed_runs(manifest)
    methods = [str(value).upper() for value in cfg.selection.methods]
    targets = [str(value) for value in cfg.selection.targets]
    seeds = [int(value) for value in cfg.selection.seeds]
    progresses = [float(value) for value in cfg.selection.checkpoint_progress]
    selected = select_runs(
        records,
        methods=methods,
        targets=targets,
        seeds=seeds,
    )

    expected_run_count = len(methods) * len(targets) * len(seeds)
    if len(selected) != expected_run_count:
        discovered = {
            (rec.method.upper(), rec.target, rec.seed)
            for rec in selected
        }
        missing = [
            (method, target, seed)
            for method in methods
            for target in targets
            for seed in seeds
            if (method, target, seed) not in discovered
        ]
        raise RuntimeError(
            f"Expected {expected_run_count} selected runs, found "
            f"{len(selected)}; missing={missing}"
        )

    method_order = {method: index for index, method in enumerate(methods)}
    target_order = {target: index for index, target in enumerate(targets)}
    selected.sort(
        key=lambda rec: (
            target_order[rec.target],
            method_order[rec.method.upper()],
            rec.seed,
        )
    )

    cells: list[CellSpec] = []
    for rec in selected:
        run_cfg = OmegaConf.load(rec.config_path)
        total_epochs = int(run_cfg.train.epochs)
        checkpoints = find_all_checkpoints(rec.result_path)
        stage_checkpoints = select_progress_checkpoints(
            checkpoints,
            total_epochs=total_epochs,
            progresses=progresses,
        )
        for progress, epoch, checkpoint_dir in stage_checkpoints:
            vi_path = checkpoint_dir / "vi_model.pt"
            if not vi_path.is_file():
                raise FileNotFoundError(vi_path)
            if rec.method.upper() in {"AISIVI", "DSIVI"}:
                reverse_path = checkpoint_dir / "reverse_model.pt"
                if not reverse_path.is_file():
                    raise FileNotFoundError(reverse_path)
            cells.append(
                CellSpec(
                    record=rec,
                    progress=progress,
                    epoch=epoch,
                    checkpoint_dir=checkpoint_dir,
                )
            )
    return cells


def _conditional_parameters(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not hasattr(vi_model, "net") or not hasattr(
        vi_model, "_variance_from_raw"
    ):
        raise TypeError(
            "Score analysis requires the ConditionalGaussian VI interface."
        )
    output = vi_model.net(epsilon)
    mu, var_raw = output.chunk(2, dim=-1)
    var, _ = vi_model._variance_from_raw(var_raw)
    return mu, var


def conditional_logp_and_score(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    mu, var = _conditional_parameters(vi_model, epsilon)
    log_var = torch.log(var)
    dimension = z.shape[-1]
    logp = -0.5 * (
        dimension * math.log(2.0 * math.pi)
        + log_var.sum(dim=-1)
        + (((z - mu) ** 2) / var).sum(dim=-1)
    )
    score = -(z - mu) / var
    return logp, score


def diagonal_gaussian_mixture_block(
    z: torch.Tensor,
    mu: torch.Tensor,
    var: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return log component-sum and score for one shared component bank."""
    if z.ndim != 2 or mu.ndim != 2 or var.shape != mu.shape:
        raise ValueError("Expected z=[N,D] and mu,var=[K,D].")
    if z.shape[-1] != mu.shape[-1]:
        raise ValueError("z and component dimensions do not match.")

    inv_var = var.reciprocal()
    mu_inv_var = mu * inv_var
    component_const = -0.5 * (
        torch.log(var).sum(dim=-1)
        + (mu * mu_inv_var).sum(dim=-1)
        + z.shape[-1] * math.log(2.0 * math.pi)
    )
    log_components = (
        z @ mu_inv_var.transpose(0, 1)
        - 0.5 * (z * z) @ inv_var.transpose(0, 1)
        + component_const.unsqueeze(0)
    )
    log_sum = torch.logsumexp(log_components, dim=1)
    weights = torch.softmax(log_components, dim=1)
    weighted_inv_var = weights @ inv_var
    weighted_mu_inv_var = weights @ mu_inv_var
    score = weighted_mu_inv_var - z * weighted_inv_var
    return log_sum, score


def mixture_block_summary(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        mu, var = _conditional_parameters(vi_model, epsilon)
        return diagonal_gaussian_mixture_block(z, mu, var)


def merge_mixture_summaries(
    left_log_sum: torch.Tensor,
    left_score: torch.Tensor,
    right_log_sum: torch.Tensor,
    right_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Merge scores for two disjoint, unnormalised mixture component sets."""
    combined_log_sum = torch.logaddexp(left_log_sum, right_log_sum)
    left_weight = torch.exp(left_log_sum - combined_log_sum).unsqueeze(-1)
    right_weight = torch.exp(right_log_sum - combined_log_sum).unsqueeze(-1)
    combined_score = (
        left_weight * left_score + right_weight * right_score
    )
    return combined_log_sum, combined_score


def streamed_reference_score(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    *,
    reverse_batch_size: int,
    num_batches: int,
    accumulator_dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    if reverse_batch_size < 1 or num_batches < 1:
        raise ValueError("Reference batch size and count must be positive.")

    cumulative_log_sum: torch.Tensor | None = None
    cumulative_score: torch.Tensor | None = None
    with torch.no_grad():
        for _ in range(num_batches):
            epsilon = vi_model.sample_epsilon(num=reverse_batch_size)
            block_log_sum, block_score = mixture_block_summary(
                vi_model,
                z,
                epsilon,
            )
            block_log_sum = block_log_sum.to(accumulator_dtype)
            block_score = block_score.to(accumulator_dtype)
            if cumulative_log_sum is None:
                cumulative_log_sum = block_log_sum
                cumulative_score = block_score
            else:
                assert cumulative_score is not None
                cumulative_log_sum, cumulative_score = (
                    merge_mixture_summaries(
                        cumulative_log_sum,
                        cumulative_score,
                        block_log_sum,
                        block_score,
                    )
                )
    assert cumulative_score is not None
    return cumulative_score


def autograd_mixture_score(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    epsilon: torch.Tensor,
) -> torch.Tensor:
    """Small-batch reference used to validate the analytic identity."""
    z_grad = z.detach().clone().requires_grad_(True)
    z_expanded = z_grad.unsqueeze(1).expand(-1, epsilon.shape[0], -1)
    epsilon_expanded = epsilon.unsqueeze(0).expand(
        z.shape[0], -1, -1
    )
    log_components = vi_model.logp(z_expanded, epsilon_expanded)
    log_mixture = torch.logsumexp(log_components, dim=1)
    return torch.autograd.grad(log_mixture.sum(), z_grad)[0].detach()


def native_sivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    auxiliary_count = int(runner.training_reverse_sample_num)
    auxiliary_epsilon = runner.vi_model.sample_epsilon(num=auxiliary_count)
    auxiliary_log_sum, auxiliary_score = mixture_block_summary(
        runner.vi_model,
        z,
        auxiliary_epsilon,
    )
    with torch.no_grad():
        generating_logp, generating_score = conditional_logp_and_score(
            runner.vi_model,
            z,
            generating_epsilon,
        )
        _, score = merge_mixture_summaries(
            auxiliary_log_sum,
            auxiliary_score,
            generating_logp,
            generating_score,
        )
    return score, {"native_auxiliary_samples": auxiliary_count + 1}


def native_uivi_score(
    runner: Any,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    z_aux, epsilon_aux, acceptance_rate = runner.sample_epsilon_hmc(
        z,
        eps_init=generating_epsilon,
        num_samples=int(runner.training_reverse_sample_num),
        burn_in_steps=int(runner.hmc_burn_in_steps),
        step_size=float(runner.hmc_step_size),
        leapfrog_steps=int(runner.hmc_leapfrog_steps),
    )
    with torch.no_grad():
        score = runner.vi_model.score(z_aux, epsilon_aux).mean(dim=1)
    return score, {
        "native_auxiliary_samples": int(runner.training_reverse_sample_num),
        "hmc_acceptance_rate": float(acceptance_rate),
    }


def _native_aisivi_score_chunk(
    runner: Any,
    z: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    sample_count = int(runner.training_reverse_sample_num)
    with torch.no_grad():
        z_aux, epsilon_aux, log_q_reverse = runner.reverse_model.sample(
            z,
            num_samples=sample_count,
        )
        raw_importance = (
            runner.vi_model.log_q_epsilon(epsilon_aux) - log_q_reverse
        )
        if not torch.isfinite(raw_importance).all():
            raise FloatingPointError(
                "AISIVI produced non-finite importance weights."
            )
        clipped_fraction = (raw_importance > 10.0).float().mean()
        importance = raw_importance.clamp(max=10.0)
        conditional_logp, conditional_score = conditional_logp_and_score(
            runner.vi_model,
            z_aux,
            epsilon_aux,
        )
        log_terms = conditional_logp + importance
        finite = torch.isfinite(log_terms)
        if (~finite).all(dim=1).any():
            raise FloatingPointError(
                "AISIVI produced a row without any finite score terms."
            )
        safe_log_terms = torch.where(
            finite,
            log_terms,
            torch.full_like(log_terms, -torch.inf),
        )
        weights = torch.softmax(safe_log_terms, dim=1)
        safe_conditional_score = torch.where(
            finite.unsqueeze(-1),
            conditional_score,
            torch.zeros_like(conditional_score),
        )
        score = (
            weights.unsqueeze(-1) * safe_conditional_score
        ).sum(dim=1)
        if bool(runner.normalize_reverse_score):
            score = score - score.mean(dim=0, keepdim=True)
        ess = weights.square().sum(dim=1).reciprocal()
    return score, {
        "native_auxiliary_samples": sample_count,
        "importance_clipped_fraction": float(clipped_fraction.item()),
        "importance_ess_mean": float(ess.mean().item()),
        "importance_ess_min": float(ess.min().item()),
    }


def native_aisivi_score(
    runner: Any,
    z: torch.Tensor,
    *,
    z_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    if z_chunk_size is None or z_chunk_size >= z.shape[0]:
        return _native_aisivi_score_chunk(runner, z)
    if z_chunk_size < 1:
        raise ValueError("AISIVI z chunk size must be positive.")

    scores: list[torch.Tensor] = []
    diagnostics: list[tuple[int, dict[str, float]]] = []
    for start in range(0, z.shape[0], z_chunk_size):
        chunk = z[start:start + z_chunk_size]
        chunk_score, chunk_diagnostics = _native_aisivi_score_chunk(
            runner,
            chunk,
        )
        scores.append(chunk_score)
        diagnostics.append((chunk.shape[0], chunk_diagnostics))

    total = sum(size for size, _ in diagnostics)
    merged = {
        "native_auxiliary_samples": int(
            diagnostics[0][1]["native_auxiliary_samples"]
        ),
        "importance_clipped_fraction": sum(
            size * values["importance_clipped_fraction"]
            for size, values in diagnostics
        ) / total,
        "importance_ess_mean": sum(
            size * values["importance_ess_mean"]
            for size, values in diagnostics
        ) / total,
        "importance_ess_min": min(
            values["importance_ess_min"]
            for _, values in diagnostics
        ),
        "aisivi_z_chunk_size": z_chunk_size,
    }
    return torch.cat(scores, dim=0), merged


def native_dsivi_score(
    runner: Any,
    z: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    with torch.no_grad():
        score = runner.reverse_model.score(z).detach()
        if bool(runner.normalize_reverse_score):
            score = score - score.mean(dim=0, keepdim=True)
    return score, {"native_auxiliary_samples": 0}


def method_native_score(
    runner: Any,
    method: str,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    aisivi_z_chunk_size: int | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    method_upper = method.upper()
    if method_upper == "SIVI":
        return native_sivi_score(runner, z, generating_epsilon)
    if method_upper == "UIVI":
        return native_uivi_score(runner, z, generating_epsilon)
    if method_upper == "AISIVI":
        return native_aisivi_score(
            runner,
            z,
            z_chunk_size=aisivi_z_chunk_size,
        )
    if method_upper == "DSIVI":
        return native_dsivi_score(runner, z)
    raise ValueError(f"Unsupported score-analysis method: {method}")


def compute_score_metrics(
    method_score: torch.Tensor | None,
    reference_scores: torch.Tensor,
) -> dict[str, Any]:
    if reference_scores.ndim != 3:
        raise ValueError("reference_scores must have shape [R,N,D].")
    if (
        method_score is not None
        and method_score.shape != reference_scores.shape[1:]
    ):
        raise ValueError("Method and reference score shapes do not match.")
    if not torch.isfinite(reference_scores).all():
        raise FloatingPointError(NONFINITE_SCORE_MESSAGE)
    if method_score is not None and not torch.isfinite(method_score).all():
        raise FloatingPointError(NONFINITE_SCORE_MESSAGE)

    reference_mean = reference_scores.mean(dim=0)
    repeat_internal = (
        (reference_scores - reference_mean.unsqueeze(0))
        .square()
        .sum(dim=-1)
        .mean(dim=1)
    )
    metrics: dict[str, Any] = {
        "reference_internal_l2": float(repeat_internal.mean().item()),
        "reference_repeat_internal_l2": [
            float(value) for value in repeat_internal.tolist()
        ],
        "reference_mean_score_sq_norm": float(
            reference_mean.square().sum(dim=-1).mean().item()
        ),
    }
    if method_score is None:
        metrics.update({
            "method_l2": None,
            "method_l2_z_sd": None,
            "method_score_sq_norm": None,
        })
        return metrics

    method_point_l2 = (
        (method_score.to(reference_mean.dtype) - reference_mean)
        .square()
        .sum(dim=-1)
    )
    metrics.update({
        "method_l2": float(method_point_l2.mean().item()),
        "method_l2_z_sd": float(
            method_point_l2.std(unbiased=True).item()
            if method_point_l2.numel() > 1
            else 0.0
        ),
        "method_score_sq_norm": float(
            method_score.to(reference_mean.dtype)
            .square()
            .sum(dim=-1)
            .mean()
            .item()
        ),
    })
    return metrics


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _load_checkpoint(runner: Any, spec: CellSpec) -> None:
    vi_state = torch.load(
        spec.checkpoint_dir / "vi_model.pt",
        map_location=runner.device,
    )
    runner.vi_model.load_state_dict(vi_state)
    runner.vi_model.eval()
    for parameter in runner.vi_model.parameters():
        parameter.requires_grad_(False)

    if spec.record.method.upper() in {"AISIVI", "DSIVI"}:
        reverse_state = torch.load(
            spec.checkpoint_dir / "reverse_model.pt",
            map_location=runner.device,
        )
        runner.reverse_model.load_state_dict(reverse_state)
        runner.reverse_model.eval()
        for parameter in runner.reverse_model.parameters():
            parameter.requires_grad_(False)
    runner.curr_epoch = spec.epoch


def _accumulator_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float64", "double", "torch.float64"}:
        return torch.float64
    if normalized in {"float32", "single", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported accumulator dtype: {name}")


def evaluate_cell(
    runner: Any,
    spec: CellSpec,
    cfg: DictConfig,
    *,
    fingerprint: str,
) -> dict[str, Any]:
    _load_checkpoint(runner, spec)
    device = torch.device(runner.device)
    use_cuda = device.type == "cuda"
    forward_count = int(cfg.evaluation.forward_batch_size)
    reference_cfg = cfg.evaluation.reference
    reverse_batch_size = int(reference_cfg.reverse_batch_size)
    reference_batches = int(reference_cfg.num_batches)
    reference_repeats = int(reference_cfg.repeats)
    accumulator_dtype = _accumulator_dtype(
        str(reference_cfg.accumulator_dtype)
    )

    forward_seed = stable_seed(spec.key, "forward")
    seed_everything(forward_seed, use_cuda=use_cuda)
    generating_epsilon, z = runner.vi_model.sampling(num=forward_count)

    method_seed = stable_seed(spec.key, "method")
    seed_everything(method_seed, use_cuda=use_cuda)
    _sync(device)
    method_started = time.perf_counter()
    method_score: torch.Tensor | None
    method_status = "ok"
    method_error = ""
    try:
        method_score, diagnostics = method_native_score(
            runner,
            spec.record.method,
            z,
            generating_epsilon,
            aisivi_z_chunk_size=int(
                cfg.evaluation.get("aisivi_z_chunk_size", forward_count)
            ),
        )
    except RuntimeError as exc:
        if (
            spec.record.method.upper() != "AISIVI"
            or "Failed to obtain finite samples from RealNVP" not in str(exc)
        ):
            raise
        # This is the method's native training-time failure mode. Preserve
        # the cell and reference diagnostics rather than silently replacing
        # the score with a different estimator.
        method_score = None
        diagnostics = {"native_auxiliary_samples": int(
            runner.training_reverse_sample_num
        )}
        method_status = "unavailable"
        method_error = f"{type(exc).__name__}: {exc}"
    _sync(device)
    method_runtime = time.perf_counter() - method_started

    reference_scores: list[torch.Tensor] = []
    reference_runtimes: list[float] = []
    reference_seeds: list[int] = []
    for repeat in range(reference_repeats):
        repeat_seed = stable_seed(spec.key, "reference", repeat)
        reference_seeds.append(repeat_seed)
        seed_everything(repeat_seed, use_cuda=use_cuda)
        _sync(device)
        reference_started = time.perf_counter()
        reference_score = streamed_reference_score(
            runner.vi_model,
            z,
            reverse_batch_size=reverse_batch_size,
            num_batches=reference_batches,
            accumulator_dtype=accumulator_dtype,
        )
        _sync(device)
        reference_runtimes.append(
            time.perf_counter() - reference_started
        )
        reference_scores.append(reference_score)

    stacked_reference = torch.stack(reference_scores, dim=0)
    metrics = compute_score_metrics(method_score, stacked_reference)
    total_runtime = method_runtime + sum(reference_runtimes)

    gpu_name = ""
    if use_cuda:
        gpu_name = torch.cuda.get_device_name(device)

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
        "reference_reverse_batch_size": reverse_batch_size,
        "reference_num_batches": reference_batches,
        "reference_total_auxiliaries": (
            reverse_batch_size * reference_batches
        ),
        "reference_repeats": reference_repeats,
        "accumulator_dtype": str(accumulator_dtype),
        "forward_seed": forward_seed,
        "method_seed": method_seed,
        "reference_seeds": reference_seeds,
        "method_runtime_sec": method_runtime,
        "method_status": method_status,
        "method_error": method_error,
        "reference_runtime_sec": sum(reference_runtimes),
        "reference_repeat_runtime_sec": reference_runtimes,
        "total_runtime_sec": total_runtime,
        "device": str(device),
        "gpu_name": gpu_name,
        "diagnostics": diagnostics,
        **metrics,
        "completed_at": utc_now(),
    }


def cell_record_path(
    run_root: Path,
    spec: CellSpec,
) -> Path:
    return (
        run_root
        / "cells"
        / spec.record.target
        / spec.record.method.upper()
        / f"seed_{spec.record.seed}"
        / f"epoch_{spec.epoch}.json"
    )


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temp_path, path)


def _read_cell(path: Path, fingerprint: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("analysis_fingerprint") != fingerprint:
        raise RuntimeError(
            f"Cell fingerprint mismatch in {path}: "
            f"{payload.get('analysis_fingerprint')} != {fingerprint}"
        )
    return payload


def pending_cell_specs(
    specs: Iterable[CellSpec],
    *,
    run_root: Path,
    fingerprint: str,
    resume: bool,
) -> list[CellSpec]:
    pending: list[CellSpec] = []
    for spec in specs:
        path = cell_record_path(run_root, spec)
        if resume and path.is_file():
            payload = _read_cell(path, fingerprint)
            if payload.get("cell_key") == spec.key:
                continue
        pending.append(spec)
    return pending


def _build_runner(rec: RunRecord, cfg: DictConfig) -> Any:
    runner_cfg = prepare_config(
        rec,
        device=str(cfg.evaluation.device),
        scratch_results=str(cfg.output.scratch_results_dir),
        scratch_tb=str(cfg.output.scratch_tb_dir),
    )
    set_seed(rec.seed, runner_cfg.device == "cuda")
    runner = Runners[rec.runner_type](config=runner_cfg)
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    return runner


def _release_runner(runner: Any | None) -> None:
    if runner is None:
        return
    if hasattr(runner, "writer"):
        runner.writer.close()
    remove_file_handlers()
    del runner
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _flatten_cell(record: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "reference_repeat_internal_l2",
        "reference_repeat_runtime_sec",
        "reference_seeds",
        "diagnostics",
    }
    row = {
        key: value
        for key, value in record.items()
        if key not in excluded
    }
    for key, value in record.get("diagnostics", {}).items():
        row[f"diagnostic_{key}"] = value
    return row


def _summary_rows(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, float, int],
        list[dict[str, Any]],
    ] = {}
    for record in records:
        key = (
            str(record["target"]),
            str(record["method"]),
            float(record["progress"]),
            int(record["epoch"]),
        )
        groups.setdefault(key, []).append(record)

    rows: list[dict[str, Any]] = []
    for (target, method, progress, epoch), items in sorted(groups.items()):
        method_values = np.asarray([
            float(item["method_l2"])
            for item in items
            if item.get("method_l2") is not None
            and math.isfinite(float(item["method_l2"]))
        ], dtype=np.float64)
        internal_values = np.asarray(
            [float(item["reference_internal_l2"]) for item in items],
            dtype=np.float64,
        )
        rows.append({
            "target": target,
            "method": method,
            "progress": progress,
            "epoch": epoch,
            "n_seeds": len(items),
            "method_n_valid": int(len(method_values)),
            "method_n_failed": int(len(items) - len(method_values)),
            "method_l2_mean": (
                float(method_values.mean())
                if len(method_values)
                else None
            ),
            "method_l2_sd": (
                float(method_values.std(ddof=1))
                if len(method_values) > 1
                else (0.0 if len(method_values) == 1 else None)
            ),
            "reference_internal_l2_mean": float(
                internal_values.mean()
            ),
            "reference_internal_l2_sd": float(
                internal_values.std(ddof=1)
                if len(internal_values) > 1
                else 0.0
            ),
            "method_runtime_sec_mean": float(
                np.mean(
                    [float(item["method_runtime_sec"]) for item in items]
                )
            ),
            "reference_runtime_sec_mean": float(
                np.mean(
                    [
                        float(item["reference_runtime_sec"])
                        for item in items
                    ]
                )
            ),
        })
    return rows


def _metric_text(mean: float | None, sd: float | None) -> str:
    if mean is None or sd is None:
        return "NA"
    return f"{mean:.4e} ± {sd:.4e}"


def _method_metric_text(row: dict[str, Any]) -> str:
    metric = _metric_text(
        row.get("method_l2_mean"),
        row.get("method_l2_sd"),
    )
    n_valid = int(row["method_n_valid"])
    n_seeds = int(row["n_seeds"])
    if n_valid != n_seeds:
        metric += f" (n={n_valid}/{n_seeds})"
    return metric


def _write_markdown_table(
    path: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# Score-Approximation Analysis",
        "",
        "All values are mean ± sample standard deviation across seeds 42–46.",
        "",
        "| Target | Method | Stage | Epoch | Method L2 | Reference internal L2 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {target} | {method} | {stage:.0f}% | {epoch} | {method_l2} | "
            "{internal_l2} |".format(
                target=row["target"],
                method=row["method"],
                stage=100.0 * float(row["progress"]),
                epoch=row["epoch"],
                method_l2=_method_metric_text(row),
                internal_l2=_metric_text(
                    float(row["reference_internal_l2_mean"]),
                    float(row["reference_internal_l2_sd"]),
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _write_latex_table(
    path: Path,
    summary_rows: list[dict[str, Any]],
) -> None:
    lines = [
        r"\begin{tabular}{llrrcc}",
        r"\toprule",
        r"Target & Method & Stage & Epoch & Method L2 & Internal L2 \\",
        r"\midrule",
    ]
    for row in summary_rows:
        if row["method_l2_mean"] is None:
            method_metric = "NA"
        else:
            method_metric = (
                f"{float(row['method_l2_mean']):.4e} "
                rf"$\pm$ {float(row['method_l2_sd']):.4e}"
            )
        if int(row["method_n_valid"]) != int(row["n_seeds"]):
            method_metric += (
                f" ($n={int(row['method_n_valid'])}/"
                f"{int(row['n_seeds'])}$)"
            )
        internal_metric = (
            f"{float(row['reference_internal_l2_mean']):.4e} "
            rf"$\pm$ {float(row['reference_internal_l2_sd']):.4e}"
        )
        lines.append(
            f"{_latex_escape(str(row['target']))} & "
            f"{_latex_escape(str(row['method']))} & "
            f"{100.0 * float(row['progress']):.0f}\\% & "
            f"{int(row['epoch'])} & {method_metric} & "
            f"{internal_metric} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def aggregate_results(
    cfg: DictConfig,
    specs: list[CellSpec],
    *,
    fingerprint: str,
    require_complete: bool = True,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    report_dir = repo_path(str(cfg.output.report_dir))
    assert runtime_root is not None and report_dir is not None
    run_root = runtime_root / fingerprint[:16]

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        path = cell_record_path(run_root, spec)
        if not path.is_file():
            missing.append(spec.key)
            continue
        records.append(_read_cell(path, fingerprint))
    if require_complete and missing:
        raise RuntimeError(
            f"Cannot aggregate incomplete analysis: {len(missing)} cells "
            f"are missing."
        )

    checkpoint_rows = [_flatten_cell(record) for record in records]
    repeat_rows: list[dict[str, Any]] = []
    for record in records:
        repeat_l2 = record["reference_repeat_internal_l2"]
        repeat_runtime = record["reference_repeat_runtime_sec"]
        repeat_seeds = record["reference_seeds"]
        for index, (value, runtime, seed) in enumerate(
            zip(repeat_l2, repeat_runtime, repeat_seeds, strict=True)
        ):
            repeat_rows.append({
                "run_id": record["run_id"],
                "target": record["target"],
                "method": record["method"],
                "seed": record["seed"],
                "progress": record["progress"],
                "epoch": record["epoch"],
                "repeat": index,
                "reference_seed": seed,
                "reference_internal_l2": value,
                "runtime_sec": runtime,
            })
    summary_rows = _summary_rows(records)
    if require_complete:
        expected_summary_rows = (
            len(cfg.selection.targets)
            * len(cfg.selection.methods)
            * len(cfg.selection.checkpoint_progress)
        )
        expected_seed_count = len(cfg.selection.seeds)
        if len(summary_rows) != expected_summary_rows:
            raise RuntimeError(
                f"Expected {expected_summary_rows} summary rows, found "
                f"{len(summary_rows)}."
            )
        invalid_seed_rows = [
            row
            for row in summary_rows
            if int(row["n_seeds"]) != expected_seed_count
        ]
        if invalid_seed_rows:
            raise RuntimeError(
                "Some summary groups do not contain the expected "
                f"{expected_seed_count} seeds."
            )

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "checkpoint_metrics.csv", checkpoint_rows)
    _write_csv(report_dir / "reference_repeat_metrics.csv", repeat_rows)
    _write_csv(report_dir / "seed_summary.csv", summary_rows)
    _write_markdown_table(
        report_dir / "score_approximation_table.md",
        summary_rows,
    )
    _write_latex_table(
        report_dir / "score_approximation_table.tex",
        summary_rows,
    )
    metadata = {
        "analysis_fingerprint": fingerprint,
        "git_commit": _git_commit(),
        "generated_at": utc_now(),
        "expected_cells": len(specs),
        "completed_cells": len(records),
        "summary_rows": len(summary_rows),
        "native_score_failures": sum(
            record.get("method_status", "ok") != "ok"
            for record in records
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    atomic_write_json(report_dir / "run_metadata.json", metadata)
    return records, summary_rows


def run_analysis(
    cfg: DictConfig,
    *,
    limit: int | None = None,
    resume: bool = True,
    aggregate_only: bool = False,
) -> tuple[int, int]:
    fingerprint = config_fingerprint(cfg)
    specs = build_cell_specs(cfg)
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    assert runtime_root is not None
    run_root = runtime_root / fingerprint[:16]

    if aggregate_only:
        records, summary = aggregate_results(
            cfg,
            specs,
            fingerprint=fingerprint,
            require_complete=True,
        )
        return len(records), len(summary)

    device_name = str(cfg.evaluation.device)
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the production configuration.")

    pending = pending_cell_specs(
        specs,
        run_root=run_root,
        fingerprint=fingerprint,
        resume=resume,
    )
    if limit is not None:
        if limit < 1:
            raise ValueError("--limit must be positive.")
        pending = pending[:limit]

    initial_pending_count = len(pending)
    runner: Any | None = None
    active_run_id: str | None = None
    completed_now = 0
    first_runtime: float | None = None
    try:
        for spec in pending:
            if spec.record.run_id != active_run_id:
                _release_runner(runner)
                runner = _build_runner(spec.record, cfg)
                active_run_id = spec.record.run_id

            started = time.perf_counter()
            assert runner is not None
            record = evaluate_cell(
                runner,
                spec,
                cfg,
                fingerprint=fingerprint,
            )
            atomic_write_json(
                cell_record_path(run_root, spec),
                record,
            )
            elapsed = time.perf_counter() - started
            completed_now += 1
            method_l2_text = (
                f"{record['method_l2']:.6e}"
                if record["method_l2"] is not None
                else "NA"
            )
            print(
                f"[{completed_now}/{initial_pending_count}] "
                f"{spec.record.method} {spec.record.target} "
                f"seed={spec.record.seed} epoch={spec.epoch}: "
                f"method_l2={method_l2_text}, "
                f"internal_l2={record['reference_internal_l2']:.6e}, "
                f"runtime={elapsed:.1f}s",
                flush=True,
            )
            if first_runtime is None:
                first_runtime = elapsed
                remaining = len(specs) - 1
                print(
                    "First full-budget cell completed; rough serial "
                    f"remaining estimate={first_runtime * remaining / 3600:.2f}h",
                    flush=True,
                )
    finally:
        _release_runner(runner)

    if limit is None:
        records, summary = aggregate_results(
            cfg,
            specs,
            fingerprint=fingerprint,
            require_complete=True,
        )
        return len(records), len(summary)
    return completed_now, 0
