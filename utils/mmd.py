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

import torch
from torch import Tensor
from typing import Tuple

from utils.kernels import BaseKernel


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
