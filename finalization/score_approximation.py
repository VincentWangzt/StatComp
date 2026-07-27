"""Checkpoint-based analysis of marginal-score approximations.

The analysis compares the score used by each training method with a multi-chain
posterior-HMC estimate of the marginal score of the checkpointed variational
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


def posterior_log_prob(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
    z: torch.Tensor,
) -> torch.Tensor:
    """Unnormalised ``log q_phi(epsilon | z)`` for posterior HMC."""
    return vi_model.log_q_epsilon(epsilon) + vi_model.logp(z, epsilon)


def posterior_log_prob_and_grad(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
    z: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Evaluate the posterior log density and its epsilon gradient."""
    with torch.enable_grad():
        epsilon_grad = epsilon.detach().requires_grad_(True)
        log_prob = posterior_log_prob(vi_model, epsilon_grad, z)
        gradient = torch.autograd.grad(
            log_prob.sum(),
            epsilon_grad,
            create_graph=False,
            retain_graph=False,
        )[0]
    return log_prob.detach(), gradient.detach()


def gelman_rubin_rhat(samples: torch.Tensor) -> torch.Tensor:
    """Classical Gelman--Rubin R-hat over ``[N,C,S,D]`` samples."""
    if samples.ndim != 4:
        raise ValueError("samples must have shape [N,C,S,D].")
    _, chains, draws, _ = samples.shape
    if chains < 2 or draws < 2:
        raise ValueError("R-hat requires at least two chains and two draws.")

    chain_means = samples.mean(dim=2)
    between = draws * chain_means.var(dim=1, unbiased=True)
    within = samples.var(dim=2, unbiased=True).mean(dim=1)
    variance_hat = ((draws - 1.0) / draws) * within + between / draws
    positive_within = within > 0
    rhat = torch.empty_like(within)
    rhat[positive_within] = torch.sqrt(
        (variance_hat[positive_within] / within[positive_within]).clamp_min(0)
    )
    both_constant = (~positive_within) & (between == 0)
    rhat[both_constant] = 1.0
    rhat[(~positive_within) & (~both_constant)] = float("inf")
    return rhat


def _finite_tensor_summary(
    values: torch.Tensor,
    *,
    prefix: str,
) -> dict[str, float | None]:
    flat = values.detach().reshape(-1).to(dtype=torch.float64, device="cpu")
    finite = flat[torch.isfinite(flat)]
    result: dict[str, float | None] = {
        f"{prefix}_nonfinite_fraction": float(
            1.0 - finite.numel() / max(1, flat.numel())
        ),
    }
    if finite.numel() == 0:
        result.update({
            f"{prefix}_median": None,
            f"{prefix}_p95": None,
            f"{prefix}_max": None,
        })
        return result
    result.update({
        f"{prefix}_median": float(finite.median().item()),
        f"{prefix}_p95": float(
            torch.quantile(finite, 0.95).item()
        ),
        f"{prefix}_max": float(finite.max().item()),
    })
    return result


def assess_hmc_reference_quality(
    diagnostics: dict[str, Any],
    quality_cfg: DictConfig,
) -> tuple[str, list[str]]:
    """Apply configured sampler-quality checks without discarding a cell."""
    checks = [
        (
            "hmc_divergence_fraction",
            "<=",
            float(quality_cfg.max_divergence_fraction),
        ),
        (
            "hmc_score_rhat_p95",
            "<=",
            float(quality_cfg.max_score_rhat_p95),
        ),
        (
            "hmc_epsilon_rhat_p95",
            "<=",
            float(quality_cfg.max_epsilon_rhat_p95),
        ),
        (
            "hmc_post_burn_acceptance_rate",
            ">=",
            float(quality_cfg.min_post_burn_acceptance_rate),
        ),
        (
            "hmc_post_burn_acceptance_min",
            ">=",
            float(quality_cfg.min_worst_chain_acceptance_rate),
        ),
    ]
    issues: list[str] = []
    for key, operator, threshold in checks:
        value = diagnostics.get(key)
        if value is None or not math.isfinite(float(value)):
            issues.append(f"{key}=nonfinite")
            continue
        numeric = float(value)
        passed = (
            numeric <= threshold
            if operator == "<="
            else numeric >= threshold
        )
        if not passed:
            issues.append(
                f"{key}={numeric:.6g} {operator} {threshold:.6g} failed"
            )
    return ("pass" if not issues else "warning"), issues


def posterior_hmc_reference_scores(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    total_samples: int,
    num_chains: int,
    burn_in_steps: int,
    thinning: int,
    step_size: float,
    leapfrog_steps: int,
    init_jitter_scale: float,
    adapt_step_size: bool,
    target_acceptance: float,
    adaptation_rate: float,
    min_step_size: float,
    max_step_size: float,
    divergence_threshold: float,
    accumulator_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Estimate the marginal score with batched posterior HMC.

    The returned tensor has shape ``[C,N,Dz]``.  Each entry along the first
    axis is a chain-mean score and therefore serves as one independent
    reference replicate in the internal-L2 calculation.
    """
    if z.ndim != 2 or generating_epsilon.ndim != 2:
        raise ValueError("z and generating_epsilon must both be rank two.")
    if z.shape[0] != generating_epsilon.shape[0]:
        raise ValueError("z and generating_epsilon batch sizes must match.")
    if total_samples < 1 or num_chains < 2:
        raise ValueError(
            "HMC requires positive total_samples and at least two chains."
        )
    if total_samples % num_chains != 0:
        raise ValueError("total_samples must be divisible by num_chains.")
    if burn_in_steps < 0 or thinning < 1:
        raise ValueError("Invalid HMC burn-in or thinning.")
    if step_size <= 0 or leapfrog_steps < 1:
        raise ValueError("Invalid HMC step size or leapfrog count.")
    if init_jitter_scale < 0:
        raise ValueError("init_jitter_scale must be non-negative.")
    if not 0 < target_acceptance < 1:
        raise ValueError("target_acceptance must be in (0, 1).")
    if adaptation_rate < 0:
        raise ValueError("adaptation_rate must be non-negative.")
    if not 0 < min_step_size <= step_size <= max_step_size:
        raise ValueError(
            "Require min_step_size <= step_size <= max_step_size."
        )
    if divergence_threshold <= 0:
        raise ValueError("divergence_threshold must be positive.")

    draws_per_chain = total_samples // num_chains
    batch_size, z_dim = z.shape
    epsilon_dim = generating_epsilon.shape[-1]
    device = z.device
    dtype = generating_epsilon.dtype

    z_chains = z.detach().unsqueeze(1).expand(
        batch_size,
        num_chains,
        z_dim,
    )
    epsilon_current = generating_epsilon.detach().unsqueeze(1).expand(
        batch_size,
        num_chains,
        epsilon_dim,
    ).clone()
    if init_jitter_scale:
        jitter = torch.randn_like(epsilon_current) * init_jitter_scale
        jitter[:, 0, :].zero_()
        epsilon_current = epsilon_current + jitter

    log_step = torch.full(
        (batch_size, num_chains, 1),
        math.log(step_size),
        device=device,
        dtype=dtype,
    )
    min_log_step = math.log(min_step_size)
    max_log_step = math.log(max_step_size)
    accepted_sum = torch.zeros(
        batch_size,
        num_chains,
        device=device,
        dtype=torch.float64,
    )
    retained_accept_sum = torch.zeros_like(accepted_sum)
    retained_transitions = 0
    divergence_count = torch.zeros_like(accepted_sum)
    squared_jump_sum = torch.zeros_like(accepted_sum)
    epsilon_samples: list[torch.Tensor] = []
    score_samples: list[torch.Tensor] = []

    total_transitions = burn_in_steps + draws_per_chain * thinning
    for transition in range(total_transitions):
        transition_step = log_step.exp()
        epsilon_before = epsilon_current
        momentum_initial = torch.randn_like(epsilon_current)
        log_prob_initial, gradient = posterior_log_prob_and_grad(
            vi_model,
            epsilon_current,
            z_chains,
        )
        kinetic_initial = 0.5 * momentum_initial.square().sum(dim=-1)

        momentum = (
            momentum_initial
            + 0.5 * transition_step * gradient
        )
        epsilon_proposed = epsilon_current
        log_prob_proposed = log_prob_initial
        for leapfrog_index in range(leapfrog_steps):
            epsilon_proposed = (
                epsilon_proposed + transition_step * momentum
            )
            log_prob_proposed, gradient = posterior_log_prob_and_grad(
                vi_model,
                epsilon_proposed,
                z_chains,
            )
            if leapfrog_index != leapfrog_steps - 1:
                momentum = momentum + transition_step * gradient
        momentum = momentum + 0.5 * transition_step * gradient
        kinetic_proposed = 0.5 * momentum.square().sum(dim=-1)

        delta_h = (
            kinetic_proposed - log_prob_proposed
            - kinetic_initial + log_prob_initial
        )
        finite_transition = (
            torch.isfinite(delta_h)
            & torch.isfinite(log_prob_initial)
            & torch.isfinite(log_prob_proposed)
        )
        log_acceptance = torch.where(
            finite_transition,
            (-delta_h).clamp(max=0),
            torch.full_like(delta_h, -torch.inf),
        )
        acceptance_probability = torch.exp(log_acceptance)
        accept = (
            torch.log(torch.rand_like(log_acceptance))
            < log_acceptance
        )
        epsilon_current = torch.where(
            accept.unsqueeze(-1),
            epsilon_proposed,
            epsilon_current,
        ).detach()

        accepted_sum += accept.to(torch.float64)
        divergence_count += (
            (~finite_transition) | (delta_h.abs() > divergence_threshold)
        ).to(torch.float64)
        squared_jump_sum += (
            epsilon_current - epsilon_before
        ).square().sum(dim=-1).to(torch.float64)

        if adapt_step_size and transition < burn_in_steps:
            gain = adaptation_rate / math.sqrt(transition + 1.0)
            log_step = (
                log_step
                + gain
                * (
                    acceptance_probability.detach().unsqueeze(-1)
                    - target_acceptance
                )
            ).clamp(min=min_log_step, max=max_log_step)

        if transition >= burn_in_steps:
            retained_accept_sum += accept.to(torch.float64)
            retained_transitions += 1
            retained_index = transition - burn_in_steps
            if retained_index % thinning == 0:
                epsilon_samples.append(epsilon_current)
                with torch.no_grad():
                    score_samples.append(
                        vi_model.score(z_chains, epsilon_current).detach()
                    )

    if len(score_samples) != draws_per_chain:
        raise RuntimeError(
            "Posterior HMC retained an unexpected number of samples."
        )
    stacked_epsilon = torch.stack(epsilon_samples, dim=2)
    stacked_score = torch.stack(score_samples, dim=2)
    chain_score_means = stacked_score.mean(
        dim=2,
        dtype=accumulator_dtype,
    ).permute(1, 0, 2).contiguous()

    epsilon_rhat = gelman_rubin_rhat(stacked_epsilon)
    score_rhat = gelman_rubin_rhat(stacked_score)
    post_burn_acceptance = retained_accept_sum / max(
        1,
        retained_transitions,
    )
    total_acceptance = accepted_sum / total_transitions
    final_step_size = log_step.exp().squeeze(-1)
    diagnostics: dict[str, Any] = {
        "hmc_num_chains": num_chains,
        "hmc_samples_per_chain": draws_per_chain,
        "hmc_total_samples": total_samples,
        "hmc_burn_in_steps": burn_in_steps,
        "hmc_thinning": thinning,
        "hmc_leapfrog_steps": leapfrog_steps,
        "hmc_acceptance_rate": float(total_acceptance.mean().item()),
        "hmc_post_burn_acceptance_rate": float(
            post_burn_acceptance.mean().item()
        ),
        "hmc_post_burn_acceptance_min": float(
            post_burn_acceptance.min().item()
        ),
        "hmc_divergence_fraction": float(
            divergence_count.sum().item()
            / (total_transitions * batch_size * num_chains)
        ),
        "hmc_mean_squared_jump_distance": float(
            (
                squared_jump_sum
                / total_transitions
            ).mean().item()
        ),
        "hmc_final_step_size_median": float(
            final_step_size.median().item()
        ),
        "hmc_final_step_size_p05": float(
            torch.quantile(final_step_size, 0.05).item()
        ),
        "hmc_final_step_size_p95": float(
            torch.quantile(final_step_size, 0.95).item()
        ),
        **_finite_tensor_summary(
            epsilon_rhat,
            prefix="hmc_epsilon_rhat",
        ),
        **_finite_tensor_summary(
            score_rhat,
            prefix="hmc_score_rhat",
        ),
    }
    return chain_score_means, diagnostics


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
    target_score: torch.Tensor | None = None,
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
    if target_score is not None:
        if target_score.shape != reference_scores.shape[1:]:
            raise ValueError("Target and reference score shapes do not match.")
        if not torch.isfinite(target_score).all():
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
        "reference_mean_mcse_l2": float(
            repeat_internal.mean().item()
            / max(1, reference_scores.shape[0] - 1)
        ),
    }
    if target_score is None:
        metrics.update({
            "target_score_sq_norm": None,
            "reference_target_l2": None,
        })
    else:
        target_score_acc = target_score.to(reference_mean.dtype)
        metrics.update({
            "target_score_sq_norm": float(
                target_score_acc.square().sum(dim=-1).mean().item()
            ),
            "reference_target_l2": float(
                (
                    (reference_mean - target_score_acc)
                    .square()
                    .sum(dim=-1)
                    .mean()
                ).item()
            ),
        })
    if method_score is None:
        metrics.update({
            "method_l2": None,
            "method_relative_l2": None,
            "method_target_l2": None,
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
        "method_relative_l2": float(
            method_point_l2.mean().item()
            / max(
                float(
                    reference_mean.square().sum(dim=-1).mean().item()
                ),
                torch.finfo(reference_mean.dtype).eps,
            )
        ),
        "method_target_l2": (
            float(
                (
                    (
                        method_score.to(reference_mean.dtype)
                        - target_score.to(reference_mean.dtype)
                    )
                    .square()
                    .sum(dim=-1)
                    .mean()
                ).item()
            )
            if target_score is not None
            else None
        ),
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
    reference_estimator = str(reference_cfg.estimator).lower()
    if reference_estimator != "posterior_hmc":
        raise ValueError(
            "The production score reference must use posterior_hmc."
        )
    reference_total_samples = int(reference_cfg.total_samples)
    reference_num_chains = int(reference_cfg.num_chains)
    if (
        reference_num_chains < 1
        or reference_total_samples % reference_num_chains != 0
    ):
        raise ValueError(
            "reference.total_samples must be divisible by a positive "
            "reference.num_chains."
        )
    reference_samples_per_chain = (
        reference_total_samples // reference_num_chains
    )
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
        method_score, method_diagnostics = method_native_score(
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
        method_diagnostics = {"native_auxiliary_samples": int(
            runner.training_reverse_sample_num
        )}
        method_status = "unavailable"
        method_error = f"{type(exc).__name__}: {exc}"
    _sync(device)
    method_runtime = time.perf_counter() - method_started

    reference_seed = stable_seed(spec.key, "reference_hmc")
    seed_everything(reference_seed, use_cuda=use_cuda)
    _sync(device)
    reference_started = time.perf_counter()
    reference_scores, reference_diagnostics = (
        posterior_hmc_reference_scores(
            runner.vi_model,
            z,
            generating_epsilon,
            total_samples=reference_total_samples,
            num_chains=reference_num_chains,
            burn_in_steps=int(reference_cfg.burn_in_steps),
            thinning=int(reference_cfg.thinning),
            step_size=float(reference_cfg.step_size),
            leapfrog_steps=int(reference_cfg.leapfrog_steps),
            init_jitter_scale=float(reference_cfg.init_jitter_scale),
            adapt_step_size=bool(reference_cfg.adapt_step_size),
            target_acceptance=float(reference_cfg.target_acceptance),
            adaptation_rate=float(reference_cfg.adaptation_rate),
            min_step_size=float(reference_cfg.min_step_size),
            max_step_size=float(reference_cfg.max_step_size),
            divergence_threshold=float(
                reference_cfg.divergence_threshold
            ),
            accumulator_dtype=accumulator_dtype,
        )
    )
    _sync(device)
    reference_runtime = time.perf_counter() - reference_started
    reference_replicate_runtimes = [
        reference_runtime / reference_num_chains
    ] * reference_num_chains
    reference_quality_status, reference_quality_issues = (
        assess_hmc_reference_quality(
            reference_diagnostics,
            reference_cfg.quality,
        )
    )

    with torch.no_grad():
        target_score = runner.target_model.score(z).detach()
    metrics = compute_score_metrics(
        method_score,
        reference_scores,
        target_score,
    )
    total_runtime = method_runtime + reference_runtime

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
        "reference_estimator": reference_estimator,
        "reference_total_samples": reference_total_samples,
        "reference_num_chains": reference_num_chains,
        "reference_samples_per_chain": reference_samples_per_chain,
        "reference_repeats": reference_num_chains,
        "reference_replication_unit": "hmc_chain",
        "reference_quality_status": reference_quality_status,
        "reference_quality_issues": reference_quality_issues,
        "accumulator_dtype": str(accumulator_dtype),
        "forward_seed": forward_seed,
        "method_seed": method_seed,
        "reference_seed": reference_seed,
        "method_runtime_sec": method_runtime,
        "method_status": method_status,
        "method_error": method_error,
        "reference_runtime_sec": reference_runtime,
        "reference_repeat_runtime_sec": reference_replicate_runtimes,
        "total_runtime_sec": total_runtime,
        "device": str(device),
        "gpu_name": gpu_name,
        "diagnostics": {
            **method_diagnostics,
            **reference_diagnostics,
        },
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
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _flatten_cell(record: dict[str, Any]) -> dict[str, Any]:
    excluded = {
        "reference_repeat_internal_l2",
        "reference_repeat_runtime_sec",
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
        def finite_values(key: str) -> np.ndarray:
            return np.asarray([
                float(item[key])
                for item in items
                if item.get(key) is not None
                and math.isfinite(float(item[key]))
            ], dtype=np.float64)

        method_values = finite_values("method_l2")
        method_relative_values = finite_values("method_relative_l2")
        method_target_values = finite_values("method_target_l2")
        reference_target_values = finite_values("reference_target_l2")
        internal_values = finite_values("reference_internal_l2")
        mcse_values = finite_values("reference_mean_mcse_l2")

        def mean_sd(
            values: np.ndarray,
        ) -> tuple[float | None, float | None]:
            if len(values) == 0:
                return None, None
            return (
                float(values.mean()),
                float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            )

        method_mean, method_sd = mean_sd(method_values)
        relative_mean, relative_sd = mean_sd(method_relative_values)
        method_target_mean, method_target_sd = mean_sd(
            method_target_values
        )
        reference_target_mean, reference_target_sd = mean_sd(
            reference_target_values
        )
        internal_mean, internal_sd = mean_sd(internal_values)
        mcse_mean, mcse_sd = mean_sd(mcse_values)
        rows.append({
            "target": target,
            "method": method,
            "progress": progress,
            "epoch": epoch,
            "n_seeds": len(items),
            "method_n_valid": int(len(method_values)),
            "method_n_failed": int(len(items) - len(method_values)),
            "method_l2_mean": method_mean,
            "method_l2_sd": method_sd,
            "method_relative_l2_mean": relative_mean,
            "method_relative_l2_sd": relative_sd,
            "method_target_l2_mean": method_target_mean,
            "method_target_l2_sd": method_target_sd,
            "reference_target_l2_mean": reference_target_mean,
            "reference_target_l2_sd": reference_target_sd,
            "reference_internal_l2_mean": internal_mean,
            "reference_internal_l2_sd": internal_sd,
            "reference_mean_mcse_l2_mean": mcse_mean,
            "reference_mean_mcse_l2_sd": mcse_sd,
            "reference_quality_n_pass": sum(
                item.get("reference_quality_status") == "pass"
                for item in items
            ),
            "reference_quality_n_warning": sum(
                item.get("reference_quality_status") != "pass"
                for item in items
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


def _named_method_metric_text(
    row: dict[str, Any],
    name: str,
) -> str:
    metric = _metric_text(
        row.get(f"{name}_mean"),
        row.get(f"{name}_sd"),
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
        "The reference internal L2 is calculated across posterior-HMC chain "
        "means.",
        "",
        "| Target | Method | Stage | Epoch | Method vs HMC q | "
        "Method vs target p | HMC q vs target p | HMC-chain internal L2 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {target} | {method} | {stage:.0f}% | {epoch} | "
            "{method_l2} | {method_target_l2} | {reference_target_l2} | "
            "{internal_l2} |".format(
                target=row["target"],
                method=row["method"],
                stage=100.0 * float(row["progress"]),
                epoch=row["epoch"],
                method_l2=_method_metric_text(row),
                method_target_l2=_named_method_metric_text(
                    row,
                    "method_target_l2",
                ),
                reference_target_l2=_metric_text(
                    row.get("reference_target_l2_mean"),
                    row.get("reference_target_l2_sd"),
                ),
                internal_l2=_metric_text(
                    row.get("reference_internal_l2_mean"),
                    row.get("reference_internal_l2_sd"),
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
        r"\begin{tabular}{llrrcccc}",
        r"\toprule",
        r"Target & Method & Stage & Epoch & Method--HMC & Method--target "
        r"& HMC--target & HMC-chain L2 \\",
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
        method_target_metric = _named_method_metric_text(
            row,
            "method_target_l2",
        ).replace("±", r"$\pm$")
        reference_target_metric = _metric_text(
            row.get("reference_target_l2_mean"),
            row.get("reference_target_l2_sd"),
        ).replace("±", r"$\pm$")
        internal_metric = (
            f"{float(row['reference_internal_l2_mean']):.4e} "
            rf"$\pm$ {float(row['reference_internal_l2_sd']):.4e}"
        )
        lines.append(
            f"{_latex_escape(str(row['target']))} & "
            f"{_latex_escape(str(row['method']))} & "
            f"{100.0 * float(row['progress']):.0f}\\% & "
            f"{int(row['epoch'])} & {method_metric} & "
            f"{method_target_metric} & {reference_target_metric} & "
            f"{internal_metric} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


_SCORE_METHOD_COLORS = {
    "SIVI": "#9467bd",
    "UIVI": "#1f77b4",
    "AISIVI": "#ff7f0e",
    "DSIVI": "#2ca02c",
}


def _score_display_method(method: str) -> str:
    return "DIVI" if method.upper() == "DSIVI" else method.upper()


def _save_score_figure(
    figure: Any,
    path_stem: Path,
) -> list[Path]:
    png_path = path_stem.with_suffix(".png")
    pdf_path = path_stem.with_suffix(".pdf")
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    return [png_path, pdf_path]


def _plot_summary_metrics(
    cfg: DictConfig,
    summary_rows: list[dict[str, Any]],
    *,
    report_dir: Path,
    filename: str,
    metric_rows: list[tuple[str, str]],
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [str(value) for value in cfg.selection.targets]
    methods = [str(value).upper() for value in cfg.selection.methods]
    figure, axes = plt.subplots(
        len(metric_rows),
        len(targets),
        figsize=(5.0 * len(targets), 3.25 * len(metric_rows)),
        squeeze=False,
    )
    for column, target in enumerate(targets):
        for row_index, (metric, label) in enumerate(metric_rows):
            axis = axes[row_index, column]
            for method_index, method in enumerate(methods):
                selected = sorted(
                    (
                        row
                        for row in summary_rows
                        if str(row["target"]) == target
                        and str(row["method"]).upper() == method
                        and row.get(f"{metric}_mean") is not None
                    ),
                    key=lambda row: float(row["progress"]),
                )
                if not selected:
                    continue
                x = np.asarray(
                    [int(row["epoch"]) for row in selected],
                    dtype=np.float64,
                )
                mean = np.asarray(
                    [float(row[f"{metric}_mean"]) for row in selected],
                    dtype=np.float64,
                )
                sd = np.asarray(
                    [float(row[f"{metric}_sd"]) for row in selected],
                    dtype=np.float64,
                )
                finite_positive = (
                    np.isfinite(mean)
                    & np.isfinite(sd)
                    & (mean > 0)
                )
                if not finite_positive.any():
                    continue
                x = x[finite_positive]
                mean = mean[finite_positive]
                sd = sd[finite_positive]
                color = _SCORE_METHOD_COLORS.get(
                    method,
                    f"C{method_index}",
                )
                axis.plot(
                    x,
                    mean,
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    label=_score_display_method(method),
                )
                lower = np.maximum(
                    mean - sd,
                    np.maximum(mean * 1.0e-3, 1.0e-12),
                )
                upper = mean + sd
                axis.fill_between(
                    x,
                    lower,
                    upper,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            axis.set_yscale("log")
            axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
            axis.set_xlabel("Checkpoint epoch")
            if column == 0:
                axis.set_ylabel(label)
            if row_index == 0:
                axis.set_title(target.replace("_", " "))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.01),
        )
    figure.tight_layout(rect=(0, 0.05, 1, 1))
    paths = _save_score_figure(figure, report_dir / filename)
    plt.close(figure)
    return paths


def _diagnostic_summary(
    records: list[dict[str, Any]],
    *,
    diagnostic: str,
) -> dict[tuple[str, str, int], tuple[float, float]]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for record in records:
        value = record.get("diagnostics", {}).get(diagnostic)
        if value is None or not math.isfinite(float(value)):
            continue
        key = (
            str(record["target"]),
            str(record["method"]).upper(),
            int(record["epoch"]),
        )
        grouped.setdefault(key, []).append(float(value))
    result: dict[tuple[str, str, int], tuple[float, float]] = {}
    for key, values in grouped.items():
        array = np.asarray(values, dtype=np.float64)
        result[key] = (
            float(array.mean()),
            float(array.std(ddof=1)) if len(array) > 1 else 0.0,
        )
    return result


def _plot_hmc_diagnostics(
    cfg: DictConfig,
    records: list[dict[str, Any]],
    *,
    report_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    targets = [str(value) for value in cfg.selection.targets]
    methods = [str(value).upper() for value in cfg.selection.methods]
    diagnostic_rows = [
        (
            "hmc_post_burn_acceptance_rate",
            "Post-warmup acceptance",
            "linear",
            0.8,
        ),
        (
            "hmc_score_rhat_p95",
            "Score R-hat (95th percentile)",
            "linear",
            1.1,
        ),
        (
            "hmc_final_step_size_median",
            "Final step size (median)",
            "log",
            None,
        ),
    ]
    summaries = {
        diagnostic: _diagnostic_summary(
            records,
            diagnostic=diagnostic,
        )
        for diagnostic, _, _, _ in diagnostic_rows
    }
    figure, axes = plt.subplots(
        len(diagnostic_rows),
        len(targets),
        figsize=(5.0 * len(targets), 3.0 * len(diagnostic_rows)),
        squeeze=False,
    )
    for column, target in enumerate(targets):
        for row_index, (
            diagnostic,
            label,
            scale,
            guide,
        ) in enumerate(diagnostic_rows):
            axis = axes[row_index, column]
            summary = summaries[diagnostic]
            for method_index, method in enumerate(methods):
                values = sorted(
                    (
                        (epoch, mean, sd)
                        for (
                            value_target,
                            value_method,
                            epoch,
                        ), (mean, sd) in summary.items()
                        if value_target == target
                        and value_method == method
                    ),
                    key=lambda item: item[0],
                )
                if not values:
                    continue
                x = np.asarray(
                    [value[0] for value in values],
                    dtype=np.float64,
                )
                mean = np.asarray(
                    [value[1] for value in values],
                    dtype=np.float64,
                )
                sd = np.asarray(
                    [value[2] for value in values],
                    dtype=np.float64,
                )
                color = _SCORE_METHOD_COLORS.get(
                    method,
                    f"C{method_index}",
                )
                axis.plot(
                    x,
                    mean,
                    color=color,
                    marker="o",
                    linewidth=1.8,
                    markersize=4,
                    label=_score_display_method(method),
                )
                lower = mean - sd
                if scale == "log":
                    lower = np.maximum(
                        lower,
                        np.maximum(mean * 1.0e-3, 1.0e-12),
                    )
                axis.fill_between(
                    x,
                    lower,
                    mean + sd,
                    color=color,
                    alpha=0.16,
                    linewidth=0,
                )
            if guide is not None:
                axis.axhline(
                    guide,
                    color="#555555",
                    linestyle="--",
                    linewidth=1.0,
                )
            if scale == "log":
                axis.set_yscale("log")
            if diagnostic == "hmc_post_burn_acceptance_rate":
                axis.set_ylim(0, 1.02)
            axis.grid(True, which="both", alpha=0.22, linewidth=0.6)
            axis.set_xlabel("Checkpoint epoch")
            if column == 0:
                axis.set_ylabel(label)
            if row_index == 0:
                axis.set_title(target.replace("_", " "))

    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        figure.legend(
            handles,
            labels,
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.005),
        )
    figure.tight_layout(rect=(0, 0.04, 1, 1))
    paths = _save_score_figure(
        figure,
        report_dir / "score_hmc_diagnostics",
    )
    plt.close(figure)
    return paths


def render_score_approximation_figures(
    cfg: DictConfig,
    records: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    *,
    report_dir: Path,
) -> list[Path]:
    """Render error decomposition and sampler-quality figures."""
    paths = _plot_summary_metrics(
        cfg,
        summary_rows,
        report_dir=report_dir,
        filename="score_error_comparison",
        metric_rows=[
            ("method_l2", r"Method vs HMC $q_\phi$ score L2"),
            ("method_target_l2", "Method vs target score L2"),
        ],
    )
    paths.extend(
        _plot_summary_metrics(
            cfg,
            summary_rows,
            report_dir=report_dir,
            filename="score_reference_quality",
            metric_rows=[
                (
                    "reference_target_l2",
                    r"HMC $q_\phi$ vs target score L2",
                ),
                (
                    "reference_internal_l2",
                    "HMC-chain internal L2",
                ),
            ],
        )
    )
    paths.extend(
        _plot_hmc_diagnostics(
            cfg,
            records,
            report_dir=report_dir,
        )
    )
    return paths


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
        reference_seed = int(record["reference_seed"])
        replication_unit = str(record["reference_replication_unit"])
        for index, (value, runtime) in enumerate(
            zip(repeat_l2, repeat_runtime, strict=True)
        ):
            repeat_rows.append({
                "run_id": record["run_id"],
                "target": record["target"],
                "method": record["method"],
                "seed": record["seed"],
                "progress": record["progress"],
                "epoch": record["epoch"],
                "repeat": index,
                "chain": index,
                "replication_unit": replication_unit,
                "reference_seed": reference_seed,
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
    figure_paths = render_score_approximation_figures(
        cfg,
        records,
        summary_rows,
        report_dir=report_dir,
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
        "reference_quality_warnings": sum(
            record.get("reference_quality_status") != "pass"
            for record in records
        ),
        "figures": [
            path.relative_to(report_dir).as_posix()
            for path in figure_paths
        ],
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
