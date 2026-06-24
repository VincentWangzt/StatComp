"""
KDVI Runner: Kernel Distillation Variational Inference.

Trains an implicit variational model q_phi by distilling MCMC transition
kernels. At each iteration:

1. Draw samples z ~ q_phi via reparameterization (gradient-carrying).
2. Run K steps of MCMC (SGLD, HMC, or MALA) starting from z.detach(), targeting
   the (possibly annealed) posterior p.
3. Minimize MMD²(z, z') where z' are the MCMC-refined samples (detached).

For invariant MCMC kernels, q_phi = p is a distributional fixed point and
therefore MMD² = 0. Finite-step SGLD is used as a biased training transition.

Key properties:
- No reverse model needed (unlike RSIVI/AISIVI/DSIVI).
- No log q_phi(z) estimation needed (unlike SIVI/UIVI).
- No target score at gradient time (unlike KSIVI) — score is only used
  inside the detached MCMC step.
- Gradient flows through z via reparameterization trick only.

Reference:
    Design document: MCMC_distillation.md

Config keys under ``train.kdvi``:
    mcmc_type (str): MCMC kernel type. One of 'sgld', 'hmc', 'mala'.
        Default: 'sgld'.
    mcmc_steps (int): Number of MCMC transition steps K.
        Default: 5.
    mcmc_step_size (float): Step size for the MCMC kernel.
        Default: 0.05.
    hmc_leapfrog_steps (int): Leapfrog sub-steps per HMC transition (L).
        Only used when mcmc_type='hmc'. Default: 10.
    kernel (str): Kernel type for MMD computation. One of 'gaussian',
        'gaussian_mmd', 'imq', 'laplace', 'laplace_l2', or 'riesz'.
        Default: 'gaussian'.
    fit_bandwidth_on (str): Adaptive bandwidth fitting strategy. One of 'x'
        or 'xy'. Default: 'x'.
    kernel_bandwidth (float, optional): Positive fixed kernel bandwidth. When
        provided, disables adaptive bandwidth fitting.
    mcmc_steps_schedule (dict): Optional K-step scheduling.
        enabled (bool): Whether to ramp K over training. Default: False.
        min_steps (int): Starting K. Default: 1.
        max_steps (int): Final K. Default: 10.
        warmup_epochs (int): Epochs to ramp from min to max. Default: 10000.
"""

import time
from typing import Callable, Tuple

import torch
from omegaconf import DictConfig

from runner.base_runner import BaseSIVIRunner
from utils.mcmc_kernels import (
    sgld_transition,
    hmc_transition,
    mala_transition,
)
from utils.mmd import configure_kernel_bandwidth, mmd2_v_statistic
from utils.kernels import Kernels
from utils.annealing import annealing, mcmc_step_schedule
from utils.logging import get_logger

logger = get_logger()


class KDVIRunner(BaseSIVIRunner):
    """Kernel Distillation Variational Inference runner.

    Minimizes MMD²(q_phi samples, MCMC-refined samples) to train q_phi.
    Inherits all infrastructure from BaseSIVIRunner (config, logging,
    metrics, plotting, checkpointing) and overrides only the loss
    computation.

    Architecture::

        KDVIRunner
        ├── VI Model (models/vi_model.py)          — generates z ~ q_phi
        ├── MCMC Kernel (utils/mcmc_kernels.py)    — refines z -> z'
        ├── MMD Kernel (utils/kernels.py)          — computes MMD²(z, z')
        └── Target (models/target_models.py)       — provides log p / score
    """

    def __init__(self, config: DictConfig, name: str = "KDVI"):
        super().__init__(config=config, name=name)

        # --- Parse KDVI-specific config ---
        kdvi_cfg = self.training_cfg.get('kdvi', {})

        # MCMC kernel settings
        self.mcmc_type: str = kdvi_cfg.get('mcmc_type', 'sgld')
        assert self.mcmc_type in ('sgld', 'hmc', 'mala'), \
            f"mcmc_type must be 'sgld', 'hmc', or 'mala', got '{self.mcmc_type}'"
        self.mcmc_steps: int = int(kdvi_cfg.get('mcmc_steps', 5))
        self.mcmc_step_size: float = float(kdvi_cfg.get('mcmc_step_size', 0.05))
        self.hmc_leapfrog_steps: int = int(
            kdvi_cfg.get('hmc_leapfrog_steps', 10))

        # K-step scheduling
        schedule_cfg = kdvi_cfg.get('mcmc_steps_schedule', {})
        self.k_schedule_enabled: bool = bool(schedule_cfg.get('enabled', False))
        self.k_schedule_min: int = int(schedule_cfg.get('min_steps', 1))
        self.k_schedule_max: int = int(schedule_cfg.get('max_steps', 10))
        self.k_schedule_warmup: int = int(
            schedule_cfg.get('warmup_epochs', 10000))

        # MMD kernel settings
        kernel_type: str = kdvi_cfg.get('kernel', 'gaussian')
        assert kernel_type in Kernels, \
            f"kernel must be one of {list(Kernels.keys())}, got '{kernel_type}'"
        self.mmd_kernel = Kernels[kernel_type]()
        self.mmd_kernel_type: str = kernel_type
        # Optional fixed bandwidth — if set, overrides adaptive fitting and
        # pins kernel.h to this value for the entire training run.
        # Set kernel_bandwidth to null/None or omit it to keep current
        # adaptive median-heuristic behavior.
        kb = kdvi_cfg.get('kernel_bandwidth', None)
        self.fixed_kernel_bandwidth: float | None = (
            float(kb) if kb is not None else None
        )
        self.fit_bandwidth_on = configure_kernel_bandwidth(
            kernel=self.mmd_kernel,
            fit_bandwidth_on=kdvi_cfg.get('fit_bandwidth_on', 'x'),
            kernel_bandwidth=self.fixed_kernel_bandwidth,
        )

        # Optional step-size schedule:
        #   step_size_schedule:
        #     type: none | cosine | coupled
        #     start: <float>   # for cosine: starting step size
        #     end:   <float>   # for cosine: ending step size
        #     steps: <int>     # for cosine: number of epochs to decay over
        # 'coupled' divides mcmc_step_size by the current beta (notebook style).
        # 'none' (default) keeps mcmc_step_size constant.
        sched = kdvi_cfg.get('step_size_schedule', {}) or {}
        self.step_size_schedule_type: str = str(
            sched.get('type', 'none')).lower()
        assert self.step_size_schedule_type in ('none', 'cosine', 'coupled'), \
            f"step_size_schedule.type must be 'none', 'cosine', or " \
            f"'coupled', got '{self.step_size_schedule_type}'"
        self.step_size_schedule_start: float = float(
            sched.get('start', self.mcmc_step_size))
        self.step_size_schedule_end: float = float(
            sched.get('end', self.mcmc_step_size))
        self.step_size_schedule_steps: int = int(
            sched.get('steps', 50000))

        # KDVI has no reverse model
        self.reverse_train = False

        logger.info(
            f"KDVIRunner initialized: mcmc_type={self.mcmc_type}, "
            f"mcmc_steps={self.mcmc_steps}, "
            f"mcmc_step_size={self.mcmc_step_size}, "
            f"hmc_leapfrog_steps={self.hmc_leapfrog_steps}, "
            f"mmd_kernel={kernel_type}, "
            f"fit_bandwidth_on={self.fit_bandwidth_on}"
        )
        if self.fixed_kernel_bandwidth is not None:
            logger.info(
                f"Fixed kernel bandwidth enabled: h="
                f"{self.fixed_kernel_bandwidth} (adaptive fitting disabled)"
            )
        if self.step_size_schedule_type != 'none':
            logger.info(
                f"Step-size schedule enabled: type="
                f"{self.step_size_schedule_type}, start="
                f"{self.step_size_schedule_start}, end="
                f"{self.step_size_schedule_end}, steps="
                f"{self.step_size_schedule_steps}"
            )
        if self.k_schedule_enabled:
            logger.info(
                f"K-step scheduling enabled: K ramps from "
                f"{self.k_schedule_min} to {self.k_schedule_max} over "
                f"{self.k_schedule_warmup} epochs"
            )

    def _get_log_prob_fn(self, epoch: int) -> Tuple[Callable, float]:
        """Construct a (possibly annealed) log-probability function.

        Uses the annealing infrastructure from BaseSIVIRunner's config
        (``train.annealing``). When annealing is enabled, the target
        log-density is scaled by beta(t) which ramps from 0.1 to 1.0.

        Args:
            epoch: Current training epoch.

        Returns:
            A tuple (log_prob_fn, beta) where:
                - log_prob_fn: callable ``z: [N, D] -> [N]`` returning
                  annealed log-densities.
                - beta: current annealing factor (1.0 if annealing disabled).
        """
        beta = annealing(
            t=epoch,
            warm_up_interval=self.anneal_steps,
            anneal=self.use_annealing,
            scheme=self.anneal_scheme,
        )

        def log_prob_fn(z: torch.Tensor) -> torch.Tensor:
            return beta * self.target_model.logp(z)

        return log_prob_fn, beta

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """Not used by KDVI. Raises if accidentally called."""
        raise NotImplementedError(
            "KDVI does not estimate log q(z). "
            "This method should not be called in the KDVI training loop."
        )

    def train_reverse_model(self, epoch_outer: int):
        """No-op: KDVI has no reverse model."""
        pass

    def _compute_loss_and_step(self, epoch: int) -> dict:
        """Compute MMD² loss between q_phi samples and MCMC-refined samples.

        Training step:
          1. Sample z ~ q_phi(z|epsilon) via reparameterization.
          2. Run K MCMC steps on z.detach() → z_refined (no gradient).
          3. Compute MMD²(z, z_refined) using chosen kernel.
          4. Backpropagate through MMD² w.r.t. phi only.
          5. Optimizer step with optional gradient clipping and EMA.

        Args:
            epoch: Current training epoch (used for annealing schedule).

        Returns:
            dict with diagnostic keys matching BaseSIVIRunner interface:
                - 'loss' (Tensor): Scalar MMD² loss.
                - 'grad_norm' (Tensor or None): Gradient norm.
                - 'z' (Tensor): Sampled z for diagnostic logging.
                - 'epsilon' (Tensor): Sampled epsilon.
            Phase timings are emitted directly through ``ExperimentLogger``.
        """
        # ============================================================
        # Phase 1: Sample from q_phi (with reparameterization)
        # ============================================================
        t_vi0 = time.perf_counter()

        epsilon = self.vi_model.sample_epsilon(num=self.training_batch_size)
        z, neg_score = self.vi_model.forward(epsilon)  # z: [N, D], has grad

        t_vi1 = time.perf_counter()
        self.experiment_logger.record_timing(
            "vi_sample", t_vi1 - t_vi0, step=epoch)

        # ============================================================
        # Phase 2: MCMC refinement (no gradient through this phase)
        # ============================================================
        t_mcmc0 = time.perf_counter()

        log_prob_fn, beta = self._get_log_prob_fn(epoch)

        # Determine current number of MCMC steps (fixed or scheduled)
        if self.k_schedule_enabled:
            current_mcmc_steps = mcmc_step_schedule(
                t=epoch,
                min_steps=self.k_schedule_min,
                max_steps=self.k_schedule_max,
                warmup_epochs=self.k_schedule_warmup,
            )
        else:
            current_mcmc_steps = self.mcmc_steps

        # Determine current MCMC step size (fixed, cosine-decayed, or
        # beta-coupled per the notebook style).
        if self.step_size_schedule_type == 'cosine':
            import math as _math
            progress = min(1.0, float(epoch) /
                           max(1, self.step_size_schedule_steps))
            cos_factor = 0.5 * (1.0 + _math.cos(_math.pi * progress))
            current_step_size = (
                self.step_size_schedule_end +
                (self.step_size_schedule_start - self.step_size_schedule_end)
                * cos_factor
            )
        elif self.step_size_schedule_type == 'coupled':
            # Notebook style: divide by beta so step size is large when the
            # target is flattened, and tightens as beta -> 1.
            current_step_size = self.mcmc_step_size / max(beta, 1e-6)
        else:
            current_step_size = self.mcmc_step_size

        if self.mcmc_type == 'sgld':
            # Use direct score function for efficiency (avoids autograd)
            score_fn = lambda z_in: beta * self.target_model.score(z_in)
            mcmc_out = sgld_transition(
                z_init=z.detach(),
                score_fn_or_log_prob_fn=score_fn,
                step_size=current_step_size,
                n_steps=current_mcmc_steps,
                use_score_fn=True,
            )
        elif self.mcmc_type == 'hmc':
            mcmc_out = hmc_transition(
                z_init=z.detach(),
                log_prob_fn=log_prob_fn,
                step_size=current_step_size,
                n_leapfrog=self.hmc_leapfrog_steps,
                n_steps=current_mcmc_steps,
            )
        elif self.mcmc_type == 'mala':
            # Use the analytic annealed target score for the Langevin drift;
            # log_prob_fn is only evaluated for the M-H acceptance ratio.
            score_fn = lambda z_in: beta * self.target_model.score(z_in)
            mcmc_out = mala_transition(
                z_init=z.detach(),
                log_prob_fn=log_prob_fn,
                score_fn=score_fn,
                step_size=current_step_size,
                n_steps=current_mcmc_steps,
            )
        else:
            raise ValueError(f"Unknown mcmc_type: {self.mcmc_type}")

        z_refined = mcmc_out.z  # [N, D], detached

        t_mcmc1 = time.perf_counter()
        self.experiment_logger.record_timing(
            "neg_score", t_mcmc1 - t_mcmc0, step=epoch)

        # ============================================================
        # Phase 3: MMD² loss + backward + optimizer step
        # ============================================================
        t_bw0 = time.perf_counter()

        loss, mmd_info = mmd2_v_statistic(
            x=z,
            y=z_refined,
            kernel=self.mmd_kernel,
            fit_bandwidth_on=self.fit_bandwidth_on,
        )

        # Optimizer step
        grad_norm = None

        if torch.isfinite(loss):
            self.optimizer_vi.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(), max_norm=self.grad_clip)
            else:
                grad_norm = torch.nn.utils.get_total_norm(
                    self.vi_model.parameters())
            self.optimizer_vi.step()
            self.scheduler_vi.step()
            if self.ema_enabled:
                self.ema.update_params(self.vi_model.parameters())
        else:
            logger.warning(
                f"NaN or Inf detected in KDVI loss at epoch {epoch}. "
                f"Skipping update."
            )

        t_bw1 = time.perf_counter()
        self.experiment_logger.record_timing(
            "backward", t_bw1 - t_bw0, step=epoch)

        # ============================================================
        # KDVI-specific diagnostics
        # ============================================================
        self.experiment_logger.log_scalars(
            {
                "kdvi/accept_rate": mcmc_out.accept_rate,
                "kdvi/mean_displacement": mcmc_out.mean_disp,
                "kdvi/mcmc_step_size": current_step_size,
                "kdvi/beta_anneal": beta,
                "kdvi/mcmc_steps_K": current_mcmc_steps,
                "kdvi/kernel_bandwidth": self.mmd_kernel.h,
                "kdvi/k_xx_mean": mmd_info['k_xx_mean'],
                "kdvi/k_yy_mean": mmd_info['k_yy_mean'],
                "kdvi/k_xy_mean": mmd_info['k_xy_mean'],
            },
            step=epoch,
        )

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'z': z,
            'epsilon': epsilon,
        }
