"""Curvature-standardized local quadrature for checkpoint score analysis.

This module is intentionally separate from ``score_approximation``.  It reuses
only checkpoint selection and native-method score helpers; the reference,
runtime records, aggregation, and reports have independent implementations and
output roots.
"""

from __future__ import annotations

import csv
import json
import math
import re
import subprocess
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .config import REPO_ROOT, repo_path
from .score_approximation import (
    CellSpec,
    NONFINITE_SCORE_MESSAGE,
    _build_runner,
    _load_checkpoint,
    _release_runner,
    build_cell_specs,
    config_fingerprint,
    method_native_score,
    seed_everything,
    stable_seed,
    utc_now,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "finalization"
    / "score_local_quadrature.yaml"
)
REPORT_LABEL_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def load_local_quadrature_config(
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


def filter_cell_specs(
    specs: Iterable[CellSpec],
    *,
    seeds: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    epochs: Sequence[int] | None = None,
) -> list[CellSpec]:
    seed_set = None if seeds is None else {int(value) for value in seeds}
    method_set = (
        None
        if methods is None
        else {str(value).upper() for value in methods}
    )
    target_set = (
        None if targets is None else {str(value) for value in targets}
    )
    epoch_set = None if epochs is None else {int(value) for value in epochs}
    selected = [
        spec
        for spec in specs
        if (seed_set is None or spec.record.seed in seed_set)
        and (
            method_set is None
            or spec.record.method.upper() in method_set
        )
        and (target_set is None or spec.record.target in target_set)
        and (epoch_set is None or spec.epoch in epoch_set)
    ]
    if not selected:
        raise RuntimeError("The runtime filters selected no analysis cells.")
    return selected


def _accumulator_dtype(name: str) -> torch.dtype:
    normalized = str(name).lower()
    if normalized in {"float64", "double", "torch.float64"}:
        return torch.float64
    if normalized in {"float32", "single", "torch.float32"}:
        return torch.float32
    raise ValueError(f"Unsupported accumulator dtype: {name}")


def gauss_legendre_tensor_rule(
    *,
    dimension: int,
    order: int,
    half_width: float,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return tensor-product nodes and log weights on ``[-a,a]^d``."""
    if dimension < 1 or order < 1 or half_width <= 0:
        raise ValueError("Invalid Gauss-Legendre rule parameters.")
    nodes_1d, weights_1d = np.polynomial.legendre.leggauss(order)
    node_mesh = np.meshgrid(
        *([nodes_1d] * dimension),
        indexing="ij",
    )
    weight_mesh = np.meshgrid(
        *([weights_1d] * dimension),
        indexing="ij",
    )
    nodes = np.stack(node_mesh, axis=-1).reshape(-1, dimension)
    log_weights = np.stack(
        [np.log(values) for values in weight_mesh],
        axis=-1,
    ).sum(axis=-1).reshape(-1)
    nodes = nodes * half_width
    log_weights = log_weights + dimension * math.log(half_width)
    return (
        torch.as_tensor(nodes, dtype=dtype, device=device),
        torch.as_tensor(log_weights, dtype=dtype, device=device),
    )


def _conditional_parameters(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not hasattr(vi_model, "net") or not hasattr(
        vi_model,
        "_variance_from_raw",
    ):
        raise TypeError(
            "Local quadrature requires the ConditionalGaussian interface."
        )
    output = vi_model.net(epsilon)
    mu, var_raw = output.chunk(2, dim=-1)
    var, log_var = vi_model._variance_from_raw(var_raw)
    return mu, var, log_var


def _validate_vi_model(
    vi_model: torch.nn.Module,
    *,
    epsilon_dim: int,
    z_dim: int,
) -> None:
    if vi_model.__class__.__name__ != "ConditionalGaussian":
        raise TypeError(
            "The local-box pilot supports the standard-normal "
            "ConditionalGaussian VI model only; found "
            f"{vi_model.__class__.__name__}."
        )
    if int(getattr(vi_model, "epsilon_dim", -1)) != epsilon_dim:
        raise ValueError("Unexpected epsilon dimension for local quadrature.")
    if int(getattr(vi_model, "z_dim", -1)) != z_dim:
        raise ValueError("Unexpected z dimension for local quadrature.")


def resolve_quadrature_epsilon_dim(
    vi_model: torch.nn.Module,
    configured_dimension: Any,
) -> int:
    """Resolve the integration dimension from the checkpoint-built VI model."""
    actual_dimension = int(getattr(vi_model, "epsilon_dim", -1))
    if actual_dimension < 1:
        raise ValueError("The VI model does not expose a valid epsilon dimension.")
    normalized = str(configured_dimension).strip().lower()
    if normalized in {
        "auto",
        "checkpoint",
        "auto_from_checkpoint",
    }:
        return actual_dimension
    expected_dimension = int(configured_dimension)
    if expected_dimension != actual_dimension:
        raise ValueError(
            "Configured quadrature epsilon dimension "
            f"{expected_dimension} does not match checkpoint dimension "
            f"{actual_dimension}."
        )
    return actual_dimension


def fisher_gauss_newton_scales(
    vi_model: torch.nn.Module,
    epsilon: torch.Tensor,
    *,
    batch_size: int,
    max_eigenvalue: float,
    accumulator_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Construct positive-definite local scales centered at epsilon samples.

    For a diagonal Gaussian conditional, the expected Fisher contribution is
    ``J_mu^T diag(1 / var) J_mu + 0.5 J_logvar^T J_logvar``.  The identity is
    the precision contribution from the standard-normal epsilon prior.
    """
    if epsilon.ndim != 2:
        raise ValueError("epsilon must have shape [N,D_epsilon].")
    if batch_size < 1 or max_eigenvalue < 1.0:
        raise ValueError("Invalid Fisher scaling configuration.")

    transforms: list[torch.Tensor] = []
    log_determinants: list[torch.Tensor] = []
    fisher_eigenvalues: list[torch.Tensor] = []
    scale_eigenvalues: list[torch.Tensor] = []
    epsilon_dim = epsilon.shape[-1]

    for start in range(0, epsilon.shape[0], batch_size):
        epsilon_block = (
            epsilon[start : start + batch_size]
            .detach()
            .clone()
            .requires_grad_(True)
        )
        with torch.enable_grad():
            mu, var, log_var = _conditional_parameters(
                vi_model,
                epsilon_block,
            )
            statistics = torch.cat([mu, log_var], dim=-1)
            jacobian_columns: list[torch.Tensor] = []
            for index in range(statistics.shape[-1]):
                gradient = torch.autograd.grad(
                    statistics[:, index].sum(),
                    epsilon_block,
                    retain_graph=index + 1 < statistics.shape[-1],
                    create_graph=False,
                )[0]
                jacobian_columns.append(gradient)
        jacobian = torch.stack(jacobian_columns, dim=1).to(
            accumulator_dtype
        )
        z_dim = mu.shape[-1]
        jacobian_mu = jacobian[:, :z_dim]
        jacobian_log_var = jacobian[:, z_dim:]
        inverse_var = var.detach().to(accumulator_dtype).reciprocal()

        fisher = torch.eye(
            epsilon_dim,
            device=epsilon.device,
            dtype=accumulator_dtype,
        ).expand(epsilon_block.shape[0], -1, -1).clone()
        fisher += torch.einsum(
            "bze,bz,bzf->bef",
            jacobian_mu,
            inverse_var,
            jacobian_mu,
        )
        fisher += 0.5 * torch.einsum(
            "bze,bzf->bef",
            jacobian_log_var,
            jacobian_log_var,
        )
        fisher = 0.5 * (fisher + fisher.transpose(-1, -2))
        eigenvalues, eigenvectors = torch.linalg.eigh(fisher)
        eigenvalues = eigenvalues.clamp(
            min=1.0,
            max=max_eigenvalue,
        )
        inverse_sqrt = eigenvalues.rsqrt()
        transform = (
            eigenvectors
            @ torch.diag_embed(inverse_sqrt)
            @ eigenvectors.transpose(-1, -2)
        )
        transforms.append(transform.detach())
        log_determinants.append(
            (-0.5 * eigenvalues.log().sum(dim=-1)).detach()
        )
        fisher_eigenvalues.append(eigenvalues.detach())
        scale_eigenvalues.append(inverse_sqrt.detach())

    transform_all = torch.cat(transforms, dim=0)
    log_det_all = torch.cat(log_determinants, dim=0)
    fisher_all = torch.cat(fisher_eigenvalues, dim=0)
    scale_all = torch.cat(scale_eigenvalues, dim=0)
    diagnostics = {
        "fisher_eigenvalue_min": float(fisher_all.min().item()),
        "fisher_eigenvalue_median": float(fisher_all.median().item()),
        "fisher_eigenvalue_max": float(fisher_all.max().item()),
        "physical_scale_min": float(scale_all.min().item()),
        "physical_scale_median": float(scale_all.median().item()),
        "physical_scale_max": float(scale_all.max().item()),
        "log_abs_det_scale_mean": float(log_det_all.mean().item()),
    }
    return transform_all, log_det_all, diagnostics


def _merge_weighted_score_blocks(
    left_log_sum: torch.Tensor,
    left_score: torch.Tensor,
    right_log_sum: torch.Tensor,
    right_score: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    combined_log_sum = torch.logaddexp(left_log_sum, right_log_sum)
    left_weight = torch.exp(left_log_sum - combined_log_sum).unsqueeze(-1)
    right_weight = torch.exp(
        right_log_sum - combined_log_sum
    ).unsqueeze(-1)
    return (
        combined_log_sum,
        left_weight * left_score + right_weight * right_score,
    )


def local_box_quadrature_score(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    transforms: torch.Tensor,
    log_abs_det_transforms: torch.Tensor,
    *,
    nodes: torch.Tensor,
    log_weights: torch.Tensor,
    boundary_inner_half_width: float,
    z_chunk_size: int,
    node_chunk_size: int,
    accumulator_dtype: torch.dtype = torch.float64,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Evaluate local-box density and score with streamed positive weights."""
    if z.ndim != 2 or generating_epsilon.ndim != 2:
        raise ValueError("z and generating_epsilon must be rank two.")
    if z.shape[0] != generating_epsilon.shape[0]:
        raise ValueError("z and generating_epsilon batch sizes differ.")
    if transforms.shape != (
        z.shape[0],
        generating_epsilon.shape[-1],
        generating_epsilon.shape[-1],
    ):
        raise ValueError("Unexpected local transform shape.")
    if log_abs_det_transforms.shape != (z.shape[0],):
        raise ValueError("Unexpected transform determinant shape.")
    if nodes.ndim != 2 or nodes.shape[-1] != generating_epsilon.shape[-1]:
        raise ValueError("Unexpected quadrature node shape.")
    if log_weights.shape != (nodes.shape[0],):
        raise ValueError("Unexpected quadrature weight shape.")
    if z_chunk_size < 1 or node_chunk_size < 1:
        raise ValueError("Quadrature chunk sizes must be positive.")

    score_blocks: list[torch.Tensor] = []
    log_density_blocks: list[torch.Tensor] = []
    ess_blocks: list[torch.Tensor] = []
    max_weight_blocks: list[torch.Tensor] = []
    boundary_mass_blocks: list[torch.Tensor] = []
    dimension = generating_epsilon.shape[-1]
    log_prior_constant = -0.5 * dimension * math.log(2.0 * math.pi)
    conditional_constant = -0.5 * z.shape[-1] * math.log(
        2.0 * math.pi
    )

    with torch.no_grad():
        for z_start in range(0, z.shape[0], z_chunk_size):
            z_stop = min(z.shape[0], z_start + z_chunk_size)
            z_block = z[z_start:z_stop].to(accumulator_dtype)
            center = generating_epsilon[z_start:z_stop].to(
                accumulator_dtype
            )
            transform = transforms[z_start:z_stop].to(accumulator_dtype)
            log_det = log_abs_det_transforms[z_start:z_stop].to(
                accumulator_dtype
            )

            cumulative_log_sum: torch.Tensor | None = None
            cumulative_score: torch.Tensor | None = None
            cumulative_log_square_sum: torch.Tensor | None = None
            cumulative_shell_log_sum: torch.Tensor | None = None
            cumulative_max_log_term: torch.Tensor | None = None

            for node_start in range(0, nodes.shape[0], node_chunk_size):
                node_stop = min(nodes.shape[0], node_start + node_chunk_size)
                node_block = nodes[node_start:node_stop].to(
                    device=z.device,
                    dtype=accumulator_dtype,
                )
                block_log_weights = log_weights[node_start:node_stop].to(
                    device=z.device,
                    dtype=accumulator_dtype,
                )
                epsilon_block = center.unsqueeze(1) + torch.einsum(
                    "bij,kj->bki",
                    transform,
                    node_block,
                )
                flat_epsilon = epsilon_block.reshape(
                    -1,
                    epsilon_block.shape[-1],
                ).to(dtype=generating_epsilon.dtype)
                mu, var, log_var = _conditional_parameters(
                    vi_model,
                    flat_epsilon,
                )
                mu = mu.reshape(
                    epsilon_block.shape[0],
                    epsilon_block.shape[1],
                    -1,
                ).to(accumulator_dtype)
                var = var.reshape_as(mu).to(accumulator_dtype)
                log_var = log_var.reshape_as(mu).to(accumulator_dtype)

                z_expanded = z_block.unsqueeze(1)
                conditional_log_prob = conditional_constant - 0.5 * (
                    log_var.sum(dim=-1)
                    + (
                        (z_expanded - mu).square() / var
                    ).sum(dim=-1)
                )
                conditional_score = -(z_expanded - mu) / var
                prior_log_prob = log_prior_constant - 0.5 * (
                    epsilon_block.square().sum(dim=-1)
                )
                log_terms = (
                    conditional_log_prob
                    + prior_log_prob
                    + block_log_weights.unsqueeze(0)
                    + log_det.unsqueeze(1)
                )

                block_log_sum = torch.logsumexp(log_terms, dim=1)
                block_weights = torch.softmax(log_terms, dim=1)
                block_score = (
                    block_weights.unsqueeze(-1) * conditional_score
                ).sum(dim=1)
                if cumulative_log_sum is None:
                    cumulative_log_sum = block_log_sum
                    cumulative_score = block_score
                else:
                    assert cumulative_score is not None
                    cumulative_log_sum, cumulative_score = (
                        _merge_weighted_score_blocks(
                            cumulative_log_sum,
                            cumulative_score,
                            block_log_sum,
                            block_score,
                        )
                    )

                block_log_square_sum = torch.logsumexp(
                    2.0 * log_terms,
                    dim=1,
                )
                cumulative_log_square_sum = (
                    block_log_square_sum
                    if cumulative_log_square_sum is None
                    else torch.logaddexp(
                        cumulative_log_square_sum,
                        block_log_square_sum,
                    )
                )
                block_max = log_terms.max(dim=1).values
                cumulative_max_log_term = (
                    block_max
                    if cumulative_max_log_term is None
                    else torch.maximum(cumulative_max_log_term, block_max)
                )

                shell_mask = (
                    node_block.abs().amax(dim=-1)
                    >= boundary_inner_half_width
                )
                if bool(shell_mask.any()):
                    shell_block_log_sum = torch.logsumexp(
                        log_terms[:, shell_mask],
                        dim=1,
                    )
                    cumulative_shell_log_sum = (
                        shell_block_log_sum
                        if cumulative_shell_log_sum is None
                        else torch.logaddexp(
                            cumulative_shell_log_sum,
                            shell_block_log_sum,
                        )
                    )

            assert cumulative_log_sum is not None
            assert cumulative_score is not None
            assert cumulative_log_square_sum is not None
            assert cumulative_max_log_term is not None
            score_blocks.append(cumulative_score)
            log_density_blocks.append(cumulative_log_sum)
            ess_blocks.append(
                torch.exp(
                    2.0 * cumulative_log_sum
                    - cumulative_log_square_sum
                )
            )
            max_weight_blocks.append(
                torch.exp(
                    cumulative_max_log_term - cumulative_log_sum
                )
            )
            if cumulative_shell_log_sum is None:
                boundary_mass_blocks.append(
                    torch.zeros_like(cumulative_log_sum)
                )
            else:
                boundary_mass_blocks.append(
                    torch.exp(
                        cumulative_shell_log_sum - cumulative_log_sum
                    )
                )

    score = torch.cat(score_blocks, dim=0)
    log_density = torch.cat(log_density_blocks, dim=0)
    ess = torch.cat(ess_blocks, dim=0)
    max_weight = torch.cat(max_weight_blocks, dim=0)
    boundary_mass = torch.cat(boundary_mass_blocks, dim=0)
    raw_gradient = torch.exp(log_density).unsqueeze(-1) * score

    finite_values = (
        torch.isfinite(score).all()
        and torch.isfinite(log_density).all()
        and torch.isfinite(raw_gradient).all()
        and torch.isfinite(ess).all()
        and torch.isfinite(max_weight).all()
        and torch.isfinite(boundary_mass).all()
    )
    if not bool(finite_values):
        raise FloatingPointError(NONFINITE_SCORE_MESSAGE)

    diagnostics = {
        "local_log_q_mean": float(log_density.mean().item()),
        "local_log_q_min": float(log_density.min().item()),
        "local_log_q_max": float(log_density.max().item()),
        "local_score_sq_norm": float(
            score.square().sum(dim=-1).mean().item()
        ),
        "local_raw_gradient_sq_norm": float(
            raw_gradient.square().sum(dim=-1).mean().item()
        ),
        "quadrature_ess_mean": float(ess.mean().item()),
        "quadrature_ess_p05": float(torch.quantile(ess, 0.05).item()),
        "quadrature_ess_min": float(ess.min().item()),
        "quadrature_max_weight_mean": float(max_weight.mean().item()),
        "quadrature_max_weight_p95": float(
            torch.quantile(max_weight, 0.95).item()
        ),
        "quadrature_max_weight_max": float(max_weight.max().item()),
        "quadrature_boundary_mass_mean": float(
            boundary_mass.mean().item()
        ),
        "quadrature_boundary_mass_p95": float(
            torch.quantile(boundary_mass, 0.95).item()
        ),
        "quadrature_boundary_mass_max": float(
            boundary_mass.max().item()
        ),
        "quadrature_nodes_per_z": int(nodes.shape[0]),
        "quadrature_conditional_evaluations": int(
            nodes.shape[0] * z.shape[0]
        ),
    }
    return score, log_density, diagnostics


def autograd_local_box_score(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    transforms: torch.Tensor,
    log_abs_det_transforms: torch.Tensor,
    *,
    nodes: torch.Tensor,
    log_weights: torch.Tensor,
) -> torch.Tensor:
    """Small monolithic autograd reference used only in tests."""
    z_grad = z.detach().clone().requires_grad_(True)
    accumulator_dtype = z_grad.dtype
    epsilon = (
        generating_epsilon.to(accumulator_dtype).unsqueeze(1)
        + torch.einsum(
            "bij,kj->bki",
            transforms.to(accumulator_dtype),
            nodes.to(device=z.device, dtype=accumulator_dtype),
        )
    )
    flat_epsilon = epsilon.reshape(-1, epsilon.shape[-1]).to(z.dtype)
    mu, var, log_var = _conditional_parameters(vi_model, flat_epsilon)
    mu = mu.reshape(epsilon.shape[0], epsilon.shape[1], -1).to(
        accumulator_dtype
    )
    var = var.reshape_as(mu).to(accumulator_dtype)
    log_var = log_var.reshape_as(mu).to(accumulator_dtype)
    conditional_log_prob = -0.5 * (
        z.shape[-1] * math.log(2.0 * math.pi)
        + log_var.sum(dim=-1)
        + (
            (z_grad.unsqueeze(1) - mu).square() / var
        ).sum(dim=-1)
    )
    prior_log_prob = -0.5 * (
        epsilon.shape[-1] * math.log(2.0 * math.pi)
        + epsilon.square().sum(dim=-1)
    )
    log_terms = (
        conditional_log_prob
        + prior_log_prob
        + log_weights.to(z.device, accumulator_dtype).unsqueeze(0)
        + log_abs_det_transforms.to(
            z.device,
            accumulator_dtype,
        ).unsqueeze(1)
    )
    log_density = torch.logsumexp(log_terms, dim=1)
    return torch.autograd.grad(log_density.sum(), z_grad)[0].detach()


def compute_local_score_metrics(
    method_score: torch.Tensor | None,
    local_score: torch.Tensor,
    target_score: torch.Tensor,
) -> dict[str, float | None]:
    if local_score.shape != target_score.shape:
        raise ValueError("Local and target score shapes differ.")
    if method_score is not None and method_score.shape != local_score.shape:
        raise ValueError("Method and local score shapes differ.")
    if not torch.isfinite(local_score).all():
        raise FloatingPointError(NONFINITE_SCORE_MESSAGE)
    if not torch.isfinite(target_score).all():
        raise FloatingPointError(NONFINITE_SCORE_MESSAGE)
    if method_score is not None and not torch.isfinite(method_score).all():
        raise FloatingPointError(NONFINITE_SCORE_MESSAGE)

    target_acc = target_score.to(local_score.dtype)
    local_target_l2 = (
        (local_score - target_acc).square().sum(dim=-1)
    )
    result: dict[str, float | None] = {
        "local_score_sq_norm": float(
            local_score.square().sum(dim=-1).mean().item()
        ),
        "target_score_sq_norm": float(
            target_acc.square().sum(dim=-1).mean().item()
        ),
        "local_target_l2": float(local_target_l2.mean().item()),
    }
    if method_score is None:
        result.update({
            "method_l2": None,
            "method_relative_l2": None,
            "method_target_l2": None,
            "method_l2_z_sd": None,
            "method_score_sq_norm": None,
        })
        return result

    method_acc = method_score.to(local_score.dtype)
    point_l2 = (
        (method_acc - local_score).square().sum(dim=-1)
    )
    method_target_l2 = (
        (method_acc - target_acc).square().sum(dim=-1)
    )
    result.update({
        "method_l2": float(point_l2.mean().item()),
        "method_relative_l2": float(
            point_l2.mean().item()
            / max(
                float(
                    local_score.square().sum(dim=-1).mean().item()
                ),
                torch.finfo(local_score.dtype).eps,
            )
        ),
        "method_target_l2": float(method_target_l2.mean().item()),
        "method_l2_z_sd": float(
            point_l2.std(unbiased=True).item()
            if point_l2.numel() > 1
            else 0.0
        ),
        "method_score_sq_norm": float(
            method_acc.square().sum(dim=-1).mean().item()
        ),
    })
    return result


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_local_quadrature_cell(
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
    quad = cfg.evaluation.quadrature
    epsilon_dim = resolve_quadrature_epsilon_dim(
        runner.vi_model,
        quad.epsilon_dim,
    )
    z_dim = int(quad.z_dim)
    order = int(quad.order)
    half_width = float(quad.standardized_half_width)
    accumulator_dtype = _accumulator_dtype(str(quad.accumulator_dtype))
    _validate_vi_model(
        runner.vi_model,
        epsilon_dim=epsilon_dim,
        z_dim=z_dim,
    )

    if use_cuda:
        torch.cuda.reset_peak_memory_stats(device)

    forward_seed = stable_seed(spec.key, "forward")
    seed_everything(forward_seed, use_cuda=use_cuda)
    generating_epsilon, z = runner.vi_model.sampling(num=forward_count)

    method_seed = stable_seed(spec.key, "method")
    seed_everything(method_seed, use_cuda=use_cuda)
    _sync(device)
    method_started = time.perf_counter()
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
        method_score = None
        method_status = "unavailable"
        method_error = f"{type(exc).__name__}: {exc}"
        method_diagnostics = {
            "native_auxiliary_samples": int(
                runner.training_reverse_sample_num
            )
        }
    _sync(device)
    method_runtime = time.perf_counter() - method_started

    _sync(device)
    scaling_started = time.perf_counter()
    transforms, log_determinants, scaling_diagnostics = (
        fisher_gauss_newton_scales(
            runner.vi_model,
            generating_epsilon,
            batch_size=int(quad.scaling_batch_size),
            max_eigenvalue=float(quad.max_fisher_eigenvalue),
            accumulator_dtype=accumulator_dtype,
        )
    )
    _sync(device)
    scaling_runtime = time.perf_counter() - scaling_started

    nodes, log_weights = gauss_legendre_tensor_rule(
        dimension=epsilon_dim,
        order=order,
        half_width=half_width,
        dtype=accumulator_dtype,
        device=device,
    )
    _sync(device)
    quadrature_started = time.perf_counter()
    local_score, _, quadrature_diagnostics = local_box_quadrature_score(
        runner.vi_model,
        z,
        generating_epsilon,
        transforms,
        log_determinants,
        nodes=nodes,
        log_weights=log_weights,
        boundary_inner_half_width=float(
            quad.boundary_inner_half_width
        ),
        z_chunk_size=int(quad.z_chunk_size),
        node_chunk_size=int(quad.node_chunk_size),
        accumulator_dtype=accumulator_dtype,
    )
    _sync(device)
    quadrature_runtime = time.perf_counter() - quadrature_started
    conditional_evaluations = int(
        quadrature_diagnostics["quadrature_conditional_evaluations"]
    )
    quadrature_diagnostics["quadrature_nodes_per_second"] = float(
        conditional_evaluations / max(quadrature_runtime, 1.0e-12)
    )

    with torch.no_grad():
        target_score = runner.target_model.score(z).detach()
    metrics = compute_local_score_metrics(
        method_score,
        local_score,
        target_score,
    )

    peak_allocated = 0
    peak_reserved = 0
    total_gpu_memory = 0
    headroom_gib = 0.0
    gpu_name = ""
    if use_cuda:
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        total_gpu_memory = int(
            torch.cuda.get_device_properties(device).total_memory
        )
        headroom_gib = (
            total_gpu_memory - peak_reserved
        ) / (1024.0**3)
        gpu_name = torch.cuda.get_device_name(device)
        required_headroom = float(quad.min_gpu_headroom_gib)
        if headroom_gib < required_headroom:
            raise RuntimeError(
                "Local quadrature left only "
                f"{headroom_gib:.2f} GiB GPU headroom; require "
                f"{required_headroom:.2f} GiB."
            )

    total_runtime = (
        method_runtime + scaling_runtime + quadrature_runtime
    )
    return {
        "analysis_kind": "local_box_score",
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
        "method_status": method_status,
        "method_error": method_error,
        "quadrature_estimator": str(quad.estimator),
        "quadrature_epsilon_dim": epsilon_dim,
        "quadrature_order": order,
        "quadrature_standardized_half_width": half_width,
        "quadrature_boundary_inner_half_width": float(
            quad.boundary_inner_half_width
        ),
        "quadrature_nodes_per_z": int(nodes.shape[0]),
        "accumulator_dtype": str(accumulator_dtype),
        "method_runtime_sec": method_runtime,
        "scaling_runtime_sec": scaling_runtime,
        "quadrature_runtime_sec": quadrature_runtime,
        "total_runtime_sec": total_runtime,
        "peak_gpu_allocated_bytes": peak_allocated,
        "peak_gpu_reserved_bytes": peak_reserved,
        "total_gpu_memory_bytes": total_gpu_memory,
        "gpu_headroom_gib": headroom_gib,
        "device": str(device),
        "gpu_name": gpu_name,
        "diagnostics": {
            **method_diagnostics,
            **scaling_diagnostics,
            **quadrature_diagnostics,
        },
        **metrics,
        "completed_at": utc_now(),
    }


def cell_record_path(run_root: Path, spec: CellSpec) -> Path:
    return (
        run_root
        / "cells"
        / spec.record.target
        / spec.record.method.upper()
        / f"seed_{spec.record.seed}"
        / f"epoch_{spec.epoch}.json"
    )


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    temp_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temp_path.replace(path)


def _read_cell(path: Path, fingerprint: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("analysis_fingerprint") != fingerprint:
        raise RuntimeError(f"Fingerprint mismatch in {path}.")
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


def _flatten_cell(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: value
        for key, value in record.items()
        if key != "diagnostics"
    }
    for key, value in record.get("diagnostics", {}).items():
        row[f"diagnostic_{key}"] = value
    return row


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _mean_sd(values: Sequence[float]) -> tuple[float | None, float | None]:
    finite = np.asarray(
        [value for value in values if math.isfinite(float(value))],
        dtype=np.float64,
    )
    if len(finite) == 0:
        return None, None
    return (
        float(finite.mean()),
        float(finite.std(ddof=1)) if len(finite) > 1 else None,
    )


def _summary_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
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
    metric_keys = [
        "method_l2",
        "method_relative_l2",
        "method_target_l2",
        "local_target_l2",
        "method_runtime_sec",
        "scaling_runtime_sec",
        "quadrature_runtime_sec",
        "total_runtime_sec",
        "peak_gpu_reserved_bytes",
        "gpu_headroom_gib",
    ]
    diagnostic_keys = [
        "quadrature_nodes_per_second",
        "quadrature_ess_mean",
        "quadrature_max_weight_p95",
        "quadrature_boundary_mass_p95",
        "physical_scale_min",
        "physical_scale_median",
        "physical_scale_max",
    ]
    for (target, method, progress, epoch), items in sorted(groups.items()):
        row: dict[str, Any] = {
            "target": target,
            "method": method,
            "progress": progress,
            "epoch": epoch,
            "n_seeds": len(items),
            "method_n_valid": sum(
                item.get("method_l2") is not None for item in items
            ),
        }
        epsilon_dimensions = {
            int(item["quadrature_epsilon_dim"]) for item in items
        }
        if len(epsilon_dimensions) != 1:
            raise RuntimeError(
                "A summary group contains mixed epsilon dimensions."
            )
        row["quadrature_epsilon_dim"] = epsilon_dimensions.pop()
        for key in metric_keys:
            values = [
                float(item[key])
                for item in items
                if item.get(key) is not None
            ]
            mean, sd = _mean_sd(values)
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
        for key in diagnostic_keys:
            values = [
                float(item["diagnostics"][key])
                for item in items
                if item.get("diagnostics", {}).get(key) is not None
            ]
            mean, sd = _mean_sd(values)
            row[f"{key}_mean"] = mean
            row[f"{key}_sd"] = sd
        rows.append(row)
    return rows


def _metric_text(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "NA"
    if sd is None:
        return f"{mean:.4e}"
    return f"{mean:.4e} ± {sd:.4e}"


def _write_markdown_table(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    report_label: str | None,
) -> None:
    title = "# Local-Box Quadrature Score Analysis"
    if report_label:
        title += f" — {report_label}"
    lines = [
        title,
        "",
        "Reference: Fisher-standardized `[-4,4]^{d_epsilon}`, order-13 "
        "tensor Gauss–Legendre quadrature; `d_epsilon` is read from each "
        "checkpoint and listed below.",
        "",
        "| Target | Method | Stage | Epoch | dε | Method L2 | "
        "Local–target L2 | Runtime (s) | ESS | Boundary p95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {target} | {method} | {stage:.0f}% | {epoch} | {epsilon_dim} | "
            "{method_l2} | {target_l2} | {runtime} | {ess} | "
            "{boundary} |".format(
                target=row["target"],
                method=row["method"],
                stage=100.0 * float(row["progress"]),
                epoch=int(row["epoch"]),
                epsilon_dim=int(row["quadrature_epsilon_dim"]),
                method_l2=_metric_text(
                    row["method_l2_mean"],
                    row["method_l2_sd"],
                ),
                target_l2=_metric_text(
                    row["local_target_l2_mean"],
                    row["local_target_l2_sd"],
                ),
                runtime=_metric_text(
                    row["total_runtime_sec_mean"],
                    row["total_runtime_sec_sd"],
                ),
                ess=_metric_text(
                    row["quadrature_ess_mean_mean"],
                    row["quadrature_ess_mean_sd"],
                ),
                boundary=_metric_text(
                    row["quadrature_boundary_mass_p95_mean"],
                    row["quadrature_boundary_mass_p95_sd"],
                ),
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_escape(value: str) -> str:
    return value.replace("_", r"\_")


def _latex_metric_text(mean: float | None, sd: float | None) -> str:
    if mean is None:
        return "NA"
    if sd is None:
        return f"{mean:.4e}"
    return f"{mean:.4e} $\\pm$ {sd:.4e}"


def _write_latex_table(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    lines = [
        r"\begin{tabular}{llrrrccc}",
        r"\toprule",
        r"Target & Method & Stage & Epoch & $d_\epsilon$ & Method L2 "
        r"& Local--target L2 "
        r"& Runtime (s) \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{_latex_escape(str(row['target']))} & "
            f"{_latex_escape(str(row['method']))} & "
            f"{100.0 * float(row['progress']):.0f}\\% & "
            f"{int(row['epoch'])} & "
            f"{int(row['quadrature_epsilon_dim'])} & "
            f"{_latex_metric_text(row['method_l2_mean'], row['method_l2_sd'])} & "
            f"{_latex_metric_text(row['local_target_l2_mean'], row['local_target_l2_sd'])} & "
            f"{_latex_metric_text(row['total_runtime_sec_mean'], row['total_runtime_sec_sd'])} \\\\"
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


def _report_directory(
    cfg: DictConfig,
    report_label: str | None,
) -> Path:
    base = repo_path(str(cfg.output.report_dir))
    assert base is not None
    if report_label is None:
        return base
    if not REPORT_LABEL_PATTERN.fullmatch(report_label):
        raise ValueError(
            "report_label may contain letters, digits, dot, dash, and "
            "underscore only."
        )
    return base / report_label


def aggregate_local_quadrature_results(
    cfg: DictConfig,
    specs: list[CellSpec],
    *,
    fingerprint: str,
    report_label: str | None,
    require_complete: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    assert runtime_root is not None
    run_root = runtime_root / fingerprint[:16]
    report_dir = _report_directory(cfg, report_label)

    records: list[dict[str, Any]] = []
    missing: list[str] = []
    for spec in specs:
        path = cell_record_path(run_root, spec)
        if path.is_file():
            records.append(_read_cell(path, fingerprint))
        else:
            missing.append(spec.key)
    if require_complete and missing:
        raise RuntimeError(
            f"Cannot aggregate: {len(missing)} selected cells are missing."
        )
    if not records:
        raise RuntimeError("No local-quadrature cells are available.")

    checkpoint_rows = [_flatten_cell(record) for record in records]
    summary_rows = _summary_rows(records)
    total_runtimes = np.asarray(
        [float(record["total_runtime_sec"]) for record in records],
        dtype=np.float64,
    )
    quadrature_runtimes = np.asarray(
        [float(record["quadrature_runtime_sec"]) for record in records],
        dtype=np.float64,
    )
    conditional_evaluations = sum(
        int(record["diagnostics"]["quadrature_conditional_evaluations"])
        for record in records
    )
    peak_reserved = max(
        int(record["peak_gpu_reserved_bytes"]) for record in records
    )
    runtime_summary = {
        "cell_count": len(records),
        "total_runtime_sec": float(total_runtimes.sum()),
        "cell_runtime_median_sec": float(np.median(total_runtimes)),
        "cell_runtime_p95_sec": float(
            np.quantile(total_runtimes, 0.95)
        ),
        "cell_runtime_max_sec": float(total_runtimes.max()),
        "quadrature_runtime_total_sec": float(
            quadrature_runtimes.sum()
        ),
        "quadrature_conditional_evaluations": conditional_evaluations,
        "quadrature_nodes_per_second": float(
            conditional_evaluations
            / max(float(quadrature_runtimes.sum()), 1.0e-12)
        ),
        "observed_seed_runtime_sec": float(total_runtimes.sum()),
        "estimated_five_seed_runtime_sec": float(
            total_runtimes.sum() * 5.0
        ),
        "estimated_five_seed_runtime_hours": float(
            total_runtimes.sum() * 5.0 / 3600.0
        ),
        "projected_200_cell_runtime_sec": float(
            total_runtimes.mean() * 200
        ),
        "projected_200_cell_runtime_hours": float(
            total_runtimes.mean() * 200 / 3600.0
        ),
        "peak_gpu_reserved_bytes": peak_reserved,
        "peak_gpu_reserved_gib": peak_reserved / (1024.0**3),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(report_dir / "checkpoint_metrics.csv", checkpoint_rows)
    _write_csv(report_dir / "seed_summary.csv", summary_rows)
    _write_markdown_table(
        report_dir / "score_local_quadrature_table.md",
        summary_rows,
        report_label=report_label,
    )
    _write_latex_table(
        report_dir / "score_local_quadrature_table.tex",
        summary_rows,
    )
    _atomic_write_json(
        report_dir / "runtime_summary.json",
        runtime_summary,
    )
    metadata = {
        "analysis_kind": "local_box_score",
        "analysis_fingerprint": fingerprint,
        "git_commit": _git_commit(),
        "generated_at": utc_now(),
        "selected_cells": len(specs),
        "completed_cells": len(records),
        "summary_rows": len(summary_rows),
        "report_label": report_label,
        "native_score_failures": sum(
            record.get("method_status") != "ok" for record in records
        ),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    _atomic_write_json(report_dir / "run_metadata.json", metadata)
    return records, summary_rows


def run_local_quadrature_analysis(
    cfg: DictConfig,
    *,
    seeds: Sequence[int] | None = None,
    methods: Sequence[str] | None = None,
    targets: Sequence[str] | None = None,
    epochs: Sequence[int] | None = None,
    limit: int | None = None,
    resume: bool = True,
    aggregate_only: bool = False,
    report_label: str | None = None,
) -> tuple[int, int]:
    fingerprint = config_fingerprint(cfg)
    full_specs = build_cell_specs(cfg)
    specs = filter_cell_specs(
        full_specs,
        seeds=seeds,
        methods=methods,
        targets=targets,
        epochs=epochs,
    )
    runtime_root = repo_path(str(cfg.output.runtime_dir))
    assert runtime_root is not None
    run_root = runtime_root / fingerprint[:16]

    if aggregate_only:
        records, summaries = aggregate_local_quadrature_results(
            cfg,
            specs,
            fingerprint=fingerprint,
            report_label=report_label,
            require_complete=True,
        )
        return len(records), len(summaries)

    if str(cfg.evaluation.device) == "cuda" and not torch.cuda.is_available():
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
            assert runner is not None
            started = time.perf_counter()
            record = evaluate_local_quadrature_cell(
                runner,
                spec,
                cfg,
                fingerprint=fingerprint,
            )
            _atomic_write_json(
                cell_record_path(run_root, spec),
                record,
            )
            elapsed = time.perf_counter() - started
            completed_now += 1
            method_l2 = (
                f"{float(record['method_l2']):.6e}"
                if record["method_l2"] is not None
                else "NA"
            )
            print(
                f"[{completed_now}/{initial_pending_count}] "
                f"{spec.record.method} {spec.record.target} "
                f"seed={spec.record.seed} epoch={spec.epoch}: "
                f"method_l2={method_l2}, "
                f"quadrature={record['quadrature_runtime_sec']:.2f}s, "
                f"total={elapsed:.2f}s, "
                f"headroom={record['gpu_headroom_gib']:.2f}GiB",
                flush=True,
            )
            if first_runtime is None:
                first_runtime = elapsed
                print(
                    "First-cell projections: "
                    f"40_cells={40 * first_runtime / 3600.0:.2f}h, "
                    f"200_cells={200 * first_runtime / 3600.0:.2f}h",
                    flush=True,
                )
    finally:
        _release_runner(runner)

    if limit is None:
        records, summaries = aggregate_local_quadrature_results(
            cfg,
            specs,
            fingerprint=fingerprint,
            report_label=report_label,
            require_complete=True,
        )
        return len(records), len(summaries)
    return completed_now, 0
