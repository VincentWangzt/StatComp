"""
KDVI Runner: Kernel Distillation Variational Inference.

Trains an implicit variational model q_phi by distilling MCMC transition
kernels. At each iteration:

1. Draw samples z ~ q_phi via reparameterization (gradient-carrying).
2. Run K steps of MCMC (SGLD or HMC) starting from z.detach(), targeting
   the (possibly annealed) posterior p.
3. Minimize MMD²(z, z') where z' are the MCMC-refined samples (detached).

At convergence, q_phi = p implies the MCMC kernel is a fixed point
(samples don't move), so MMD² = 0.

Key properties:
- No reverse model needed (unlike RSIVI/AISIVI/DSIVI).
- No log q_phi(z) estimation needed (unlike SIVI/UIVI).
- No target score at gradient time (unlike KSIVI) — score is only used
  inside the detached MCMC step.
- Gradient flows through z via reparameterization trick only.

Reference:
    Design document: MCMC_distillation.md

Config keys under ``train.kdvi``:
    mcmc_type (str): MCMC kernel type. One of 'sgld', 'hmc'.
        Default: 'sgld'.
    mcmc_steps (int): Number of MCMC transition steps K.
        Default: 5.
    mcmc_step_size (float): Step size for the MCMC kernel.
        Default: 0.05.
    hmc_leapfrog_steps (int): Leapfrog sub-steps per HMC transition (L).
        Only used when mcmc_type='hmc'. Default: 10.
    kernel (str): Kernel type for MMD computation. One of 'gaussian',
        'imq', 'laplace', 'riesz'. Default: 'gaussian'.
    fit_bandwidth_on (str): Bandwidth fitting strategy. One of 'x', 'y',
        'xy', 'none'. Default: 'x'.
"""

import time
from typing import Callable, Tuple

import torch
from omegaconf import DictConfig

from runner.base_runner import BaseSIVIRunner
from utils.mcmc_kernels import sgld_transition, hmc_transition
from utils.mmd import mmd2_v_statistic
from utils.kernels import Kernels
from utils.annealing import annealing
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
        assert self.mcmc_type in ('sgld', 'hmc'), \
            f"mcmc_type must be 'sgld' or 'hmc', got '{self.mcmc_type}'"
        self.mcmc_steps: int = int(kdvi_cfg.get('mcmc_steps', 5))
        self.mcmc_step_size: float = float(kdvi_cfg.get('mcmc_step_size', 0.05))
        self.hmc_leapfrog_steps: int = int(
            kdvi_cfg.get('hmc_leapfrog_steps', 10))

        # MMD kernel settings
        kernel_type: str = kdvi_cfg.get('kernel', 'gaussian')
        assert kernel_type in Kernels, \
            f"kernel must be one of {list(Kernels.keys())}, got '{kernel_type}'"
        self.mmd_kernel = Kernels[kernel_type]()
        self.mmd_kernel_type: str = kernel_type
        self.fit_bandwidth_on: str = kdvi_cfg.get('fit_bandwidth_on', 'x')
        assert self.fit_bandwidth_on in ('x', 'y', 'xy', 'none'), \
            f"fit_bandwidth_on must be 'x', 'y', 'xy', or 'none', " \
            f"got '{self.fit_bandwidth_on}'"

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
                - 'time_vi_sample' (float): Time for sampling + forward.
                - 'time_neg_score' (float): Time for MCMC transitions.
                - 'time_backward' (float): Time for backward + optimizer.
        """
        # ============================================================
        # Phase 1: Sample from q_phi (with reparameterization)
        # ============================================================
        t_vi0 = time.perf_counter()

        epsilon = self.vi_model.sample_epsilon(num=self.training_batch_size)
        z, neg_score = self.vi_model.forward(epsilon)  # z: [N, D], has grad

        t_vi1 = time.perf_counter()

        # ============================================================
        # Phase 2: MCMC refinement (no gradient through this phase)
        # ============================================================
        t_mcmc0 = time.perf_counter()

        log_prob_fn, beta = self._get_log_prob_fn(epoch)

        if self.mcmc_type == 'sgld':
            # Use direct score function for efficiency (avoids autograd)
            score_fn = lambda z_in: beta * self.target_model.score(z_in)
            mcmc_out = sgld_transition(
                z_init=z.detach(),
                score_fn_or_log_prob_fn=score_fn,
                step_size=self.mcmc_step_size,
                n_steps=self.mcmc_steps,
                use_score_fn=True,
            )
        elif self.mcmc_type == 'hmc':
            mcmc_out = hmc_transition(
                z_init=z.detach(),
                log_prob_fn=log_prob_fn,
                step_size=self.mcmc_step_size,
                n_leapfrog=self.hmc_leapfrog_steps,
                n_steps=self.mcmc_steps,
            )
        else:
            raise ValueError(f"Unknown mcmc_type: {self.mcmc_type}")

        z_refined = mcmc_out.z  # [N, D], detached

        t_mcmc1 = time.perf_counter()

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

        # ============================================================
        # KDVI-specific TensorBoard diagnostics
        # ============================================================
        self.writer.add_scalar(
            "kdvi/accept_rate", mcmc_out.accept_rate, epoch)
        self.writer.add_scalar(
            "kdvi/mean_displacement", mcmc_out.mean_disp, epoch)
        self.writer.add_scalar(
            "kdvi/beta_anneal", beta, epoch)
        self.writer.add_scalar(
            "kdvi/kernel_bandwidth", self.mmd_kernel.h, epoch)
        self.writer.add_scalar(
            "kdvi/k_xx_mean", mmd_info['k_xx_mean'], epoch)
        self.writer.add_scalar(
            "kdvi/k_yy_mean", mmd_info['k_yy_mean'], epoch)
        self.writer.add_scalar(
            "kdvi/k_xy_mean", mmd_info['k_xy_mean'], epoch)

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'z': z,
            'epsilon': epsilon,
            'time_vi_sample': t_vi1 - t_vi0,
            'time_neg_score': t_mcmc1 - t_mcmc0,
            'time_backward': t_bw1 - t_bw0,
        }
