"""
Batched MCMC transition kernels for short-run refinement chains.

This module provides lightweight, batched MCMC transition functions designed
for use in KDVI (Kernel Distillation Variational Inference). Unlike the full
samplers in ``utils/mcmc.py`` (which handle burn-in, thinning, and long-chain
sampling), these functions perform exactly K transition steps on a batch of N
particles in parallel and return the final positions.

Three kernels are provided:

- **SGLD** (Stochastic Gradient Langevin Dynamics): No accept/reject. Same
  update formula as ``SGLDSampler``. Always moves particles — biased for
  finite step sizes but provides consistent gradient signal for KDVI.
- **HMC** (Hamiltonian Monte Carlo): Leapfrog integration with Metropolis-
  Hastings correction. Same algorithm as ``HMCSampler._leapfrog`` but
  batched over N independent particles.
- **MALA** (Metropolis-Adjusted Langevin Algorithm): Langevin proposal with
  Metropolis-Hastings accept/reject. Same proposal as SGLD but corrects for
  finite step size bias via the asymmetric proposal density ratio.

The standard transition helpers operate under ``torch.no_grad()`` (except for
internal gradient computation via ``torch.autograd.grad``). The caller is
responsible for detaching inputs before passing them in. The
``sgld_transition_differentiable`` debug helper is the exception and preserves
the graph through the SGLD transition.

Typical usage in KDVI::

    z_init = z.detach()  # stop gradient from VI model
    out = sgld_transition(z_init, score_fn, step_size=0.05, n_steps=5)
    z_refined = out.z    # [N, D], detached
"""

import math
from typing import Callable, NamedTuple, Union

import torch
from torch import Tensor


class MCMCTransitionOutput(NamedTuple):
    """Output of a batched MCMC transition.

    Attributes:
        z: Final particle positions after K transition steps. Shape ``[N, D]``.
        accept_rate: Mean acceptance rate across the batch and all K steps.
            Always 1.0 for SGLD (no accept/reject).
        mean_disp: Mean Euclidean displacement ``||z_final - z_init||_2``
            averaged over the batch. Useful for diagnosing whether MCMC is
            actually moving particles.
    """

    z: Tensor
    accept_rate: float
    mean_disp: float


def _batched_grad_logp(
    z: Tensor,
    log_prob_fn: Callable[[Tensor], Tensor],
) -> Tensor:
    """Compute the gradient of log p(z) for a batch of particles.

    Uses ``torch.autograd.grad`` with a local ``enable_grad`` context.
    The input ``z`` is detached and re-attached internally to ensure clean
    gradient computation.

    Args:
        z: Particle positions, shape ``[N, D]``. Need not require grad.
        log_prob_fn: Function mapping ``z: [N, D] -> [N]`` log-densities.

    Returns:
        Gradient tensor of shape ``[N, D]``, detached from the computation
        graph.
    """
    with torch.enable_grad():
        z_req = z.detach().requires_grad_(True)
        log_p = log_prob_fn(z_req)  # [N] or [N, 1]
        grad = torch.autograd.grad(log_p.sum(), z_req)[0]  # [N, D]
    return grad.detach()


@torch.no_grad()
def sgld_transition(
    z_init: Tensor,
    score_fn_or_log_prob_fn: Union[
        Callable[[Tensor], Tensor],  # score: z -> [N, D]
        Callable[[Tensor], Tensor],  # logp: z -> [N]
    ],
    step_size: float,
    n_steps: int,
    use_score_fn: bool = False,
) -> MCMCTransitionOutput:
    """Run K steps of SGLD (Stochastic Gradient Langevin Dynamics) on a batch.

    Implements the same update as ``SGLDSampler``::

        z_{k+1} = z_k + (step_size / 2) * score(z_k) + sqrt(step_size) * noise

    No accept/reject step — always moves particles. Biased for finite step
    sizes but provides consistent, non-zero gradient signal for MMD-based
    training.

    Args:
        z_init: Starting particle positions, shape ``[N, D]``. Must be
            detached from any computation graph.
        score_fn_or_log_prob_fn: Either:
            - If ``use_score_fn=True``: a function ``z: [N, D] -> [N, D]``
              returning the score (gradient of log-density) directly.
            - If ``use_score_fn=False``: a function ``z: [N, D] -> [N]``
              returning log-densities, from which the score is computed via
              ``torch.autograd.grad``.
        step_size: SGLD step size (epsilon). Controls the magnitude of both
            the deterministic drift and stochastic noise.
        n_steps: Number of SGLD transition steps (K).
        use_score_fn: If True, treat the callable as a direct score function
            ``z -> grad_z log p(z)``. This avoids the overhead of
            ``torch.autograd.grad`` when the target already exposes a
            ``.score()`` method. Default: False.

    Returns:
        MCMCTransitionOutput with:
            - z: final positions ``[N, D]``
            - accept_rate: always 1.0 (SGLD has no rejection)
            - mean_disp: mean ``||z_final - z_init||_2`` over the batch
    """
    z = z_init.clone()
    noise_scale = math.sqrt(step_size)

    for _ in range(n_steps):
        if use_score_fn:
            score = score_fn_or_log_prob_fn(z)  # [N, D]
        else:
            score = _batched_grad_logp(z, score_fn_or_log_prob_fn)  # [N, D]

        noise = torch.randn_like(z)
        z = z + 0.5 * step_size * score + noise_scale * noise

    mean_disp = (z - z_init).norm(dim=-1).mean().item()

    return MCMCTransitionOutput(
        z=z,
        accept_rate=1.0,
        mean_disp=mean_disp,
    )


def sgld_transition_differentiable(
    z_init: Tensor,
    score_fn: Callable[[Tensor], Tensor],
    step_size: float,
    n_steps: int,
) -> MCMCTransitionOutput:
    """Run K SGLD steps while preserving the graph from ``z_init``.

    This debug helper is intentionally narrower than :func:`sgld_transition`:
    it requires an analytic score function and does not enter
    ``torch.no_grad()``. As a result, gradients can backpropagate through the
    SGLD drift terms and into the initial particles.
    """
    z = z_init.clone()
    noise_scale = math.sqrt(step_size)

    for _ in range(n_steps):
        score = score_fn(z)
        noise = torch.randn_like(z)
        z = z + 0.5 * step_size * score + noise_scale * noise

    mean_disp = (z.detach() - z_init.detach()).norm(dim=-1).mean().item()

    return MCMCTransitionOutput(
        z=z,
        accept_rate=1.0,
        mean_disp=mean_disp,
    )


@torch.no_grad()
def hmc_transition(
    z_init: Tensor,
    log_prob_fn: Callable[[Tensor], Tensor],
    step_size: float,
    n_leapfrog: int,
    n_steps: int,
) -> MCMCTransitionOutput:
    """Run K steps of batched HMC with leapfrog integration and M-H correction.

    Each of the K HMC steps:
      1. Samples fresh momentum ``p ~ N(0, I)`` for each particle.
      2. Runs ``n_leapfrog`` leapfrog sub-steps (half-step momentum, full-step
         position, half-step momentum).
      3. Applies Metropolis-Hastings accept/reject per particle based on the
         Hamiltonian energy difference.

    This is a batched version of the algorithm in ``HMCSampler._leapfrog``
    from ``utils/mcmc.py``, operating on N independent particles
    simultaneously.

    Args:
        z_init: Starting particle positions, shape ``[N, D]``. Must be
            detached from any computation graph.
        log_prob_fn: Function mapping ``z: [N, D] -> [N]`` log-densities.
            Used both for gradient computation (leapfrog) and for the M-H
            acceptance ratio.
        step_size: Leapfrog integration step size (epsilon).
        n_leapfrog: Number of leapfrog sub-steps per HMC transition (L).
        n_steps: Number of full HMC transitions (K). Each transition resamples
            momentum and does a full leapfrog + accept/reject cycle.

    Returns:
        MCMCTransitionOutput with:
            - z: final positions ``[N, D]``
            - accept_rate: mean acceptance rate across batch and K steps
            - mean_disp: mean ``||z_final - z_init||_2`` over the batch
    """
    z = z_init.clone()
    N, D = z.shape
    total_accepts = 0

    for _ in range(n_steps):
        # Sample momentum
        p = torch.randn_like(z)  # [N, D]

        # Current Hamiltonian
        current_log_p = log_prob_fn(z).squeeze()  # [N]
        current_kinetic = 0.5 * (p ** 2).sum(dim=-1)  # [N]
        current_H = -current_log_p + current_kinetic  # [N]

        # Leapfrog integration
        z_prop = z.clone()
        p_prop = p.clone()

        # Half-step momentum
        grad = _batched_grad_logp(z_prop, log_prob_fn)  # [N, D]
        p_prop = p_prop + 0.5 * step_size * grad

        # Full steps
        for lf_step in range(n_leapfrog):
            z_prop = z_prop + step_size * p_prop
            grad = _batched_grad_logp(z_prop, log_prob_fn)  # [N, D]
            if lf_step < n_leapfrog - 1:
                p_prop = p_prop + step_size * grad

        # Final half-step momentum
        p_prop = p_prop + 0.5 * step_size * grad

        # Proposed Hamiltonian
        proposed_log_p = log_prob_fn(z_prop).squeeze()  # [N]
        proposed_kinetic = 0.5 * (p_prop ** 2).sum(dim=-1)  # [N]
        proposed_H = -proposed_log_p + proposed_kinetic  # [N]

        # Metropolis-Hastings acceptance
        log_accept_ratio = current_H - proposed_H  # [N]
        accept_mask = (
            torch.log(torch.rand(N, device=z.device)) < log_accept_ratio
        )  # [N] boolean

        # Apply accept/reject per particle
        z = torch.where(accept_mask.unsqueeze(-1), z_prop, z)
        total_accepts += accept_mask.sum().item()

    accept_rate = total_accepts / (N * n_steps)
    mean_disp = (z - z_init).norm(dim=-1).mean().item()

    return MCMCTransitionOutput(
        z=z,
        accept_rate=accept_rate,
        mean_disp=mean_disp,
    )


@torch.no_grad()
def mala_transition(
    z_init: Tensor,
    log_prob_fn: Callable[[Tensor], Tensor],
    step_size: float,
    n_steps: int,
    score_fn: Callable[[Tensor], Tensor] | None = None,
) -> MCMCTransitionOutput:
    """Run K steps of MALA (Metropolis-Adjusted Langevin Algorithm) on a batch.

    Each MALA step:
      1. Proposes via Langevin dynamics:
         ``z* = z + (τ/2) * ∇log p(z) + √τ * ξ``, where ``ξ ~ N(0, I)``.
      2. Applies Metropolis-Hastings accept/reject using the asymmetric
         proposal density ratio to correct for finite step-size bias.

    The M-H acceptance ratio accounts for the asymmetry of the Langevin
    proposal::

        log α = log p(z*) - log p(z) + log q(z|z*) - log q(z*|z)

    where ``q(z*|z) = N(z*; z + (τ/2)∇log p(z), τI)``.

    For the unadjusted variant (ULA, no M-H correction), use
    ``sgld_transition`` instead — it is mathematically equivalent.

    Args:
        z_init: Starting particle positions, shape ``[N, D]``. Must be
            detached from any computation graph.
        log_prob_fn: Function mapping ``z: [N, D] -> [N]`` log-densities.
            Used for the M-H acceptance ratio. When ``score_fn`` is None it is
            also differentiated (via ``torch.autograd.grad``) to obtain the
            Langevin drift.
        step_size: Langevin step size (τ). Controls both drift and noise
            magnitude. Typical values: 0.001–0.05.
        n_steps: Number of MALA transition steps (K).
        score_fn: Optional function ``z: [N, D] -> [N, D]`` returning the
            analytic score ``∇log p(z)`` directly. When provided, the Langevin
            drift (and the forward/backward proposal densities) use this score
            with **no autograd**, mirroring
            ``IVI-via-mcmc-distillation/run_ivi.py::ImVIDrift.mala`` which uses
            ``self.target.score`` for the drift and ``self.target.logp`` only
            for the accept ratio. ``log_prob_fn`` is still used for the M-H
            ratio. Default None reverts to the autograd-of-log_prob_fn drift.

    Returns:
        MCMCTransitionOutput with:
            - z: final positions ``[N, D]``
            - accept_rate: mean acceptance rate across batch and K steps
            - mean_disp: mean ``||z_final - z_init||_2`` over the batch
    """
    z = z_init.clone()
    N, D = z.shape
    noise_scale = math.sqrt(step_size)
    total_accepts = 0

    def _grad_logp(z_in: Tensor) -> Tensor:
        # Analytic score (no autograd) when score_fn is supplied, else fall
        # back to differentiating log_prob_fn.
        if score_fn is not None:
            return score_fn(z_in)
        return _batched_grad_logp(z_in, log_prob_fn)

    for _ in range(n_steps):
        # Gradient at current position
        grad_z = _grad_logp(z)  # [N, D]

        # Langevin proposal
        noise = torch.randn_like(z)
        z_prop = z + 0.5 * step_size * grad_z + noise_scale * noise  # [N, D]

        # Gradient at proposed position (for backward proposal density)
        grad_z_prop = _grad_logp(z_prop)  # [N, D]

        # Log-densities at both positions
        log_p_current = log_prob_fn(z).squeeze()      # [N]
        log_p_proposed = log_prob_fn(z_prop).squeeze()  # [N]

        # Forward proposal log-density: log q(z*|z)
        # q(z*|z) = N(z*; z + (τ/2)∇log p(z), τI)
        # log q = -1/(2τ) * ||z* - z - (τ/2)∇log p(z)||²  (up to constant)
        diff_forward = z_prop - z - 0.5 * step_size * grad_z  # [N, D]
        log_q_forward = -0.5 / step_size * (diff_forward ** 2).sum(dim=-1)  # [N]

        # Backward proposal log-density: log q(z|z*)
        # q(z|z*) = N(z; z* + (τ/2)∇log p(z*), τI)
        diff_backward = z - z_prop - 0.5 * step_size * grad_z_prop  # [N, D]
        log_q_backward = -0.5 / step_size * (diff_backward ** 2).sum(dim=-1)  # [N]

        # M-H acceptance ratio
        log_accept_ratio = (
            log_p_proposed - log_p_current
            + log_q_backward - log_q_forward
        )  # [N]

        # Per-particle accept/reject
        accept_mask = (
            torch.log(torch.rand(N, device=z.device)) < log_accept_ratio
        )  # [N] boolean
        z = torch.where(accept_mask.unsqueeze(-1), z_prop, z)
        total_accepts += accept_mask.sum().item()

    accept_rate = total_accepts / (N * n_steps)
    mean_disp = (z - z_init).norm(dim=-1).mean().item()

    return MCMCTransitionOutput(
        z=z,
        accept_rate=accept_rate,
        mean_disp=mean_disp,
    )
