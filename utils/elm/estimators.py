from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import torch

from models.reverse_model import BaseReverseConditionalModel

from .proposal import (
    gaussian_reverse_cache,
    gaussian_reverse_mean,
    module_dtype,
    reverse_model_log_prob,
)
from .types import ELMEstimate, LogQEstimate


LOG_ZERO = float("-inf")


def sample_reference_samples(
    baseline_samples: np.ndarray | torch.Tensor,
    num_ref_samples: int,
    device: torch.device | str,
    *,
    replace: bool = False,
) -> torch.Tensor:
    if baseline_samples is None:
        raise RuntimeError("Baseline samples are required for expected log marginal.")
    if num_ref_samples < 1:
        raise ValueError("num_ref_samples must be at least 1.")

    total = int(baseline_samples.shape[0])
    if total > num_ref_samples:
        indices = np.random.choice(total, num_ref_samples, replace=replace)
        reference_samples = baseline_samples[indices]
    else:
        reference_samples = baseline_samples
    return torch.as_tensor(reference_samples, device=device)


def summarize_tensor(values: torch.Tensor, prefix: str) -> dict[str, float]:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return {
            f"{prefix}_min": float("nan"),
            f"{prefix}_p10": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_mean": float("nan"),
            f"{prefix}_p90": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    quantiles = torch.quantile(
        finite,
        torch.tensor([0.1, 0.5, 0.9], device=finite.device, dtype=finite.dtype),
    )
    return {
        f"{prefix}_min": float(finite.min().item()),
        f"{prefix}_p10": float(quantiles[0].item()),
        f"{prefix}_median": float(quantiles[1].item()),
        f"{prefix}_mean": float(finite.mean().item()),
        f"{prefix}_p90": float(quantiles[2].item()),
        f"{prefix}_max": float(finite.max().item()),
    }


def elm_stats(log_q_values: torch.Tensor) -> tuple[float, float, float, int]:
    finite = log_q_values[torch.isfinite(log_q_values)]
    count = int(finite.numel())
    if count == 0:
        return float("nan"), float("nan"), float("nan"), 0
    mean = float(finite.mean().item())
    std = float(finite.std(unbiased=True).item()) if count > 1 else 0.0
    stderr = std / math.sqrt(count)
    return mean, stderr, std, count


@torch.no_grad()
def estimate_log_q_prior(
    vi_model: torch.nn.Module,
    reference_samples: torch.Tensor,
    *,
    num_samples: int,
    epsilon_batch_size: int,
) -> LogQEstimate:
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1.")
    if epsilon_batch_size < 1:
        raise ValueError("epsilon_batch_size must be at least 1.")

    was_training = vi_model.training
    vi_model.eval()
    start = time.perf_counter()
    reference_samples = reference_samples.to(
        device=getattr(vi_model, "device", reference_samples.device),
        dtype=module_dtype(vi_model),
    )
    n_ref = int(reference_samples.shape[0])
    chunk_log_means: list[torch.Tensor] = []
    chunk_sizes: list[int] = []
    remaining = num_samples

    while remaining > 0:
        current = min(epsilon_batch_size, remaining)
        epsilon = vi_model.sample_epsilon(num=current)
        z_expanded = reference_samples.unsqueeze(1).expand(-1, current, -1)
        eps_expanded = epsilon.unsqueeze(0).expand(n_ref, -1, -1)
        log_q = vi_model.logp(z_expanded, eps_expanded)
        chunk_log_means.append(torch.logsumexp(log_q, dim=1) - math.log(current))
        chunk_sizes.append(current)
        remaining -= current

    stacked = torch.stack(chunk_log_means, dim=1)
    weights = torch.tensor(
        chunk_sizes,
        device=stacked.device,
        dtype=stacked.dtype,
    )
    log_q_values = (
        torch.logsumexp(stacked + torch.log(weights).unsqueeze(0), dim=1)
        - math.log(num_samples)
    )
    mean, stderr, log_q_std, finite_count = elm_stats(log_q_values)
    runtime_sec = time.perf_counter() - start
    if was_training:
        vi_model.train()

    return LogQEstimate(
        log_q_values=log_q_values.detach().cpu(),
        stderr=stderr,
        diagnostics={
            "estimator": "prior",
            "num_ref_samples": int(n_ref),
            "num_finite_ref_log_q": int(finite_count),
            "num_samples": int(num_samples),
            "epsilon_batch_size": int(epsilon_batch_size),
            "runtime_sec": float(runtime_sec),
            "log_q_mean": float(mean),
            "log_q_std": float(log_q_std),
        },
        ess_values=None,
    )


@torch.no_grad()
def estimate_log_q_reverse_is(
    vi_model: torch.nn.Module,
    reverse_model: BaseReverseConditionalModel,
    reference_samples: torch.Tensor,
    *,
    num_is_samples: int,
    is_batch_size: int = 1024,
    proposal_cache: dict[str, torch.Tensor] | None = None,
) -> LogQEstimate:
    if num_is_samples < 1:
        raise ValueError("num_is_samples must be at least 1.")
    if is_batch_size < 1:
        raise ValueError("is_batch_size must be at least 1.")

    device = getattr(vi_model, "device", reference_samples.device)
    vi_dtype = module_dtype(vi_model)
    was_vi_training = vi_model.training
    was_rev_training = reverse_model.training
    vi_model.eval()
    reverse_model.eval()

    reference_samples = reference_samples.to(device=device, dtype=vi_dtype)
    n_ref = int(reference_samples.shape[0])
    dtype = reference_samples.dtype
    start = time.perf_counter()

    finite_weight_count = 0
    total_weight_count = 0

    gaussian_cache = proposal_cache or gaussian_reverse_cache(reverse_model, dtype=dtype)
    log_weight_sum = torch.full((n_ref,), LOG_ZERO, device=device, dtype=dtype)
    log_weight_square_sum = torch.full((n_ref,), LOG_ZERO, device=device, dtype=dtype)

    for is_start in range(0, num_is_samples, is_batch_size):
        chunk_n = min(is_batch_size, num_is_samples - is_start)
        z_aux = reference_samples.unsqueeze(1).expand(n_ref, chunk_n, -1)

        if gaussian_cache is None:
            _, epsilon, _ = reverse_model.sample(
                reference_samples,
                num_samples=chunk_n,
            )
        else:
            mean_reverse = gaussian_reverse_mean(z_aux, gaussian_cache)
            noise = torch.randn_like(mean_reverse)
            epsilon = mean_reverse + torch.matmul(
                noise,
                gaussian_cache["chol"].transpose(0, 1),
            )

        log_prior = vi_model.log_q_epsilon(epsilon)
        log_proposal = reverse_model_log_prob(
            reverse_model,
            epsilon,
            z_aux,
            proposal_cache=gaussian_cache,
        )
        log_conditional = vi_model.logp(z_aux, epsilon)
        log_weight = log_conditional + log_prior - log_proposal
        finite_mask = torch.isfinite(log_weight)
        finite_weight_count += int(finite_mask.sum().item())
        total_weight_count += int(log_weight.numel())
        safe_log_weight = torch.where(
            finite_mask,
            log_weight,
            torch.full_like(log_weight, LOG_ZERO),
        )
        log_weight_sum = torch.logaddexp(
            log_weight_sum,
            torch.logsumexp(safe_log_weight, dim=1),
        )
        log_weight_square_sum = torch.logaddexp(
            log_weight_square_sum,
            torch.logsumexp(2.0 * safe_log_weight, dim=1),
        )

    log_q_values = log_weight_sum - math.log(num_is_samples)
    ess_values = torch.exp(2.0 * log_weight_sum - log_weight_square_sum)
    ess_values = torch.where(
        torch.isfinite(ess_values),
        ess_values.clamp(min=0.0, max=float(num_is_samples)),
        torch.zeros_like(ess_values),
    )
    mean, stderr, log_q_std, finite_count = elm_stats(log_q_values)
    runtime_sec = time.perf_counter() - start
    if was_vi_training:
        vi_model.train()
    if was_rev_training:
        reverse_model.train()

    diagnostics: dict[str, Any] = {
        "estimator": "reverse_is",
        "proposal_type": type(reverse_model).__name__,
        "num_ref_samples": int(n_ref),
        "num_finite_ref_log_q": int(finite_count),
        "num_is_samples": int(num_is_samples),
        "is_batch_size": int(is_batch_size),
        "finite_weight_count": int(finite_weight_count),
        "total_weight_count": int(total_weight_count),
        "finite_weight_fraction": (
            float(finite_weight_count / total_weight_count)
            if total_weight_count
            else float("nan")
        ),
        "runtime_sec": float(runtime_sec),
        "log_q_mean": float(mean),
        "log_q_std": float(log_q_std),
        "used_gaussian_fast_path": bool(gaussian_cache is not None),
    }
    diagnostics.update(summarize_tensor(ess_values, "ess"))

    return LogQEstimate(
        log_q_values=log_q_values.detach().cpu(),
        stderr=stderr,
        diagnostics=diagnostics,
        ess_values=ess_values.detach().cpu(),
    )


def summarize_elm(log_q: LogQEstimate) -> ELMEstimate:
    value, stderr, _, _ = elm_stats(log_q.log_q_values)
    diagnostics = dict(log_q.diagnostics)
    diagnostics["stderr"] = float(stderr)
    diagnostics["value"] = float(value)
    return ELMEstimate(
        value=value,
        stderr=stderr,
        log_q_values=log_q.log_q_values,
        diagnostics=diagnostics,
        ess_values=log_q.ess_values,
    )
