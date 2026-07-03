"""
Differentiable MMD² (Maximum Mean Discrepancy squared) estimators for training.

This module provides MMD² loss functions that retain gradient information,
suitable for use as training objectives in KDVI. Unlike the evaluation-only
``compute_mmd`` in ``utils/metrics.py`` (which returns a detached scalar),
these functions return a differentiable tensor and diagnostic information.

The key design choice for KDVI:
- ``x`` (variational samples) carries gradients via the reparameterization trick.
- ``y`` (MCMC-refined samples) is detached — no gradient flows through it.
- Bandwidth is fitted on detached samples and treated as a constant.

Typical usage::

    from utils.mmd import mmd2_v_statistic
    from utils.kernels import GaussianKernel

    kernel = GaussianKernel()
    loss, info = mmd2_v_statistic(z, z_refined, kernel, fit_bandwidth_on="x")
    loss.backward()  # gradient flows through z only
"""

import math
import torch
from torch import Tensor
from typing import Tuple

from utils.kernels import (
    BaseKernel,
    GaussianKernel,
    GaussianKernelMMD,
    LaplaceL2Kernel,
)


def configure_kernel_bandwidth(
    kernel: BaseKernel,
    fit_bandwidth_on: str = "x",
    kernel_bandwidth: float | None = None,
) -> str | None:
    """Resolve adaptive versus fixed bandwidth configuration.

    A fixed positive bandwidth takes precedence over the adaptive source.
    Otherwise only variational (``"x"``) and pooled (``"xy"``) fitting are
    supported.
    """
    if kernel_bandwidth is not None:
        fixed = float(kernel_bandwidth)
        if fixed <= 0:
            raise ValueError(
                f"kernel_bandwidth must be positive, got {fixed}"
            )
        kernel.h = fixed
        return None

    fit_source = str(fit_bandwidth_on).lower()
    if fit_source not in ("x", "xy"):
        raise ValueError(
            "fit_bandwidth_on must be 'x' or 'xy', "
            f"got '{fit_source}'"
        )
    return fit_source


def mmd2_v_statistic(
    x: Tensor,
    y: Tensor,
    kernel: BaseKernel,
    fit_bandwidth_on: str | None = "x",
) -> Tuple[Tensor, dict]:
    """Compute the biased V-statistic estimator of MMD².

    Computes::

        MMD²_V(x, y) = E[k(x, x')] + E[k(y, y')] - 2 * E[k(x, y')]

    where expectations are approximated by sample means (including diagonal
    terms i=j). This is the biased estimator — always non-negative, lower
    variance than the U-statistic.

    **Gradient flow**: Only ``x`` participates in backpropagation. ``y`` must
    be detached before calling this function. The kernel bandwidth is fitted
    on detached samples and does not carry gradients.

    Args:
        x: Samples from the variational model q_phi, shape ``[N, D]``.
            These carry gradients from the reparameterization trick.
        y: MCMC-refined target samples, shape ``[N, D]`` (or ``[M, D]``).
            Must be detached from the computation graph.
        kernel: Kernel object implementing ``pair_eval(x, y)`` and
            ``fit_h(samples)``. Any kernel from ``utils/kernels.py`` works
            (GaussianKernel, IMQKernel, LaplaceKernel, RieszKernel).
        fit_bandwidth_on: Strategy for fitting the kernel bandwidth:
            - ``"x"``: Fit on q_phi samples (detached). Tracks the current
              scale of the variational distribution. **Recommended default.**
            - ``"xy"``: Fit on the pooled set of both x and y.
            - ``None``: Use a positive bandwidth already set on the kernel.

    Returns:
        A tuple ``(mmd2, info)`` where:
            - ``mmd2`` (Tensor): Scalar MMD² value, differentiable w.r.t. ``x``.
            - ``info`` (dict): Diagnostic values for logging:
                - ``'k_xx_mean'``: Mean of k(x, x) matrix (float).
                - ``'k_yy_mean'``: Mean of k(y, y) matrix (float).
                - ``'k_xy_mean'``: Mean of k(x, y) matrix (float).

    Example::

        >>> x = vi_model(epsilon)        # [128, 2], has grad
        >>> y = sgld_transition(...).z   # [128, 2], detached
        >>> kernel = GaussianKernel()
        >>> loss, info = mmd2_v_statistic(x, y, kernel)
        >>> loss.backward()
    """
    # 1. Fit bandwidth (detached — no gradient through bandwidth selection)
    if fit_bandwidth_on == "x":
        kernel.fit_h(x.detach())
    elif fit_bandwidth_on == "xy":
        kernel.fit_h(torch.cat([x.detach(), y.detach()], dim=0))
    elif fit_bandwidth_on is None:
        if kernel.h <= 0:
            raise ValueError(
                "A positive kernel bandwidth must be set when adaptive "
                "bandwidth fitting is disabled."
            )
    else:
        raise ValueError(
            f"fit_bandwidth_on must be 'x', 'xy', or None, "
            f"got '{fit_bandwidth_on}'"
        )

    # 2. Compute pairwise kernel matrices
    # K_xx: gradient flows through both x arguments (symmetric)
    # K_yy: no gradient (both arguments detached)
    # K_xy: gradient flows through x (first argument)
    K_xx = kernel.pair_eval(x, x, fit_h=False, detach_h=True)  # [N, N]
    K_yy = kernel.pair_eval(y, y, fit_h=False, detach_h=True)  # [N, N]
    K_xy = kernel.pair_eval(x, y, fit_h=False, detach_h=True)  # [N, M]

    # 3. V-statistic (biased, includes diagonal)
    k_xx_mean = K_xx.mean()
    k_yy_mean = K_yy.mean()
    k_xy_mean = K_xy.mean()

    mmd2 = k_xx_mean + k_yy_mean - 2.0 * k_xy_mean

    info = {
        'k_xx_mean': k_xx_mean.item(),
        'k_yy_mean': k_yy_mean.item(),
        'k_xy_mean': k_xy_mean.item(),
    }

    return mmd2, info


def paired_l2_loss(x: Tensor, y: Tensor) -> Tuple[Tensor, dict]:
    """Compute the paired mean squared L2 displacement loss.

    ``x`` carries gradients and ``y`` is treated as a fixed MCMC-refined
    target. The two batches must have identical shape because samples are
    compared by index, not as distributions.
    """
    if x.shape != y.shape:
        raise ValueError(
            f"paired_l2_loss requires matching shapes, got {tuple(x.shape)} "
            f"and {tuple(y.shape)}"
        )

    squared_l2 = (x - y.detach()).square().sum(dim=-1)
    loss = squared_l2.mean()
    info = {
        'paired_l2_mean': loss.item(),
        'paired_l2_root_mean': squared_l2.sqrt().mean().item(),
    }
    return loss, info


def _per_dim_pairwise_differences(samples: Tensor) -> Tensor:
    return samples[:, None, :] - samples[None, :, :]


def _fit_per_dim_bandwidth(
    samples: Tensor,
    kernel: BaseKernel,
    min_bandwidth: float = 1e-12,
) -> Tensor:
    """Fit a diagonal bandwidth vector for supported KDVI debug kernels."""
    diffs = _per_dim_pairwise_differences(samples)
    if isinstance(kernel, GaussianKernel):
        d2 = diffs.square()
        h2 = torch.median(d2.reshape(-1, d2.shape[-1]), dim=0).values
        h2 = torch.clamp_min(h2, min_bandwidth)
        h = torch.sqrt(0.5 * h2 / math.log(samples.shape[0] + 1))
    elif isinstance(kernel, GaussianKernelMMD):
        d2 = diffs.square()
        h2 = torch.median(d2.reshape(-1, d2.shape[-1]), dim=0).values
        h2 = torch.clamp_min(h2, min_bandwidth)
        h = torch.sqrt(h2)
    elif isinstance(kernel, LaplaceL2Kernel):
        d_abs = diffs.abs()
        h = torch.median(d_abs.reshape(-1, d_abs.shape[-1]), dim=0).values
        h = torch.clamp_min(h, min_bandwidth)
    else:
        raise ValueError(
            "mmd_per_dim supports only 'gaussian', 'gaussian_mmd', and "
            f"'laplace_l2' kernels, got {kernel.name}"
        )

    return torch.clamp_min(h, min_bandwidth)


def _broadcast_fixed_per_dim_bandwidth(
    x: Tensor,
    kernel: BaseKernel,
) -> Tensor:
    if kernel.h <= 0:
        raise ValueError(
            "A positive kernel bandwidth must be set when adaptive "
            "per-dim bandwidth fitting is disabled."
        )
    return torch.full(
        (x.shape[-1],),
        float(kernel.h),
        device=x.device,
        dtype=x.dtype,
    )


def _per_dim_pair_eval(
    samples_x: Tensor,
    samples_y: Tensor,
    kernel: BaseKernel,
    bandwidths: Tensor,
) -> Tensor:
    if isinstance(kernel, (GaussianKernel, GaussianKernelMMD)):
        scaled = (
            (samples_x[:, None, :] - samples_y[None, :, :]) /
            bandwidths.view(1, 1, -1)
        )
        return torch.exp(-0.5 * scaled.square().sum(dim=-1))
    if isinstance(kernel, LaplaceL2Kernel):
        scaled_x = samples_x / bandwidths.view(1, -1)
        scaled_y = samples_y / bandwidths.view(1, -1)
        return torch.exp(-0.5 * torch.cdist(scaled_x, scaled_y, p=2))
    raise ValueError(
        "mmd_per_dim supports only 'gaussian', 'gaussian_mmd', and "
        f"'laplace_l2' kernels, got {kernel.name}"
    )


def mmd2_v_statistic_per_dim(
    x: Tensor,
    y: Tensor,
    kernel: BaseKernel,
    fit_bandwidth_on: str | None = "x",
) -> Tuple[Tensor, dict]:
    """Compute MMD² with a diagonal per-dimension bandwidth vector.

    This KDVI debug objective mirrors :func:`mmd2_v_statistic`, but replaces
    the scalar kernel bandwidth with a coordinate-wise median heuristic.
    Supported kernels are Gaussian, Gaussian MMD, and Laplace-on-L2.
    """
    if fit_bandwidth_on == "x":
        bandwidths = _fit_per_dim_bandwidth(x.detach(), kernel)
    elif fit_bandwidth_on == "xy":
        bandwidths = _fit_per_dim_bandwidth(
            torch.cat([x.detach(), y.detach()], dim=0),
            kernel,
        )
    elif fit_bandwidth_on is None:
        bandwidths = _broadcast_fixed_per_dim_bandwidth(x, kernel)
    else:
        raise ValueError(
            f"fit_bandwidth_on must be 'x', 'xy', or None, "
            f"got '{fit_bandwidth_on}'"
        )

    K_xx = _per_dim_pair_eval(x, x, kernel, bandwidths)
    K_yy = _per_dim_pair_eval(y, y, kernel, bandwidths)
    K_xy = _per_dim_pair_eval(x, y, kernel, bandwidths)

    k_xx_mean = K_xx.mean()
    k_yy_mean = K_yy.mean()
    k_xy_mean = K_xy.mean()
    mmd2 = k_xx_mean + k_yy_mean - 2.0 * k_xy_mean

    bandwidths_detached = bandwidths.detach()
    info = {
        'k_xx_mean': k_xx_mean.item(),
        'k_yy_mean': k_yy_mean.item(),
        'k_xy_mean': k_xy_mean.item(),
        'kernel_bandwidth_mean': bandwidths_detached.mean().item(),
        'kernel_bandwidth_min': bandwidths_detached.min().item(),
        'kernel_bandwidth_max': bandwidths_detached.max().item(),
    }
    return mmd2, info
