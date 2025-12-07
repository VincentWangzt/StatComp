import math
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import torch


def _grad_logp(logp_fn: Callable[[torch.Tensor], torch.Tensor],
               z: torch.Tensor) -> torch.Tensor:
    z = z.clone().detach().requires_grad_(True)
    z_in = z.view(1, -1)
    lp = logp_fn(z_in)
    if lp.dim() == 0:
        lp = lp.unsqueeze(0)
    grad = torch.autograd.grad(lp.sum(), z)[0]
    return grad


@dataclass
class HMCConfig:
    step_size: float = 0.05
    num_steps: int = 10
    num_samples: int = 1000
    burn_in: int = 200
    thinning: int = 1
    mass_diag: Optional[torch.Tensor] = None
    seed: int = 42
    device: Optional[torch.device] = None


class HMCSampler:
    """
    Simple Hamiltonian Monte Carlo (HMC) sampler.

    Args:
        logp_fn: Function returning log-density log p(z) for a batch of z.
        dim: Dimensionality of z.
        cfg: HMCConfig with hyperparameters.
    """

    def __init__(self, logp_fn: Callable[[torch.Tensor], torch.Tensor],
                 dim: int, cfg: HMCConfig):
        self.logp_fn = logp_fn
        self.dim = dim
        self.cfg = cfg
        self.device = cfg.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        torch.manual_seed(cfg.seed)
        if self.device.type == "cuda":
            torch.cuda.manual_seed_all(cfg.seed)
        self.mass_diag = cfg.mass_diag if cfg.mass_diag is not None else torch.ones(
            dim, device=self.device)
        self.inv_mass_diag = 1.0 / self.mass_diag

    def _kinetic(self, p: torch.Tensor) -> torch.Tensor:
        return 0.5 * (p.pow(2) * self.inv_mass_diag).sum(dim=-1)

    def _leapfrog(self, z: torch.Tensor, p: torch.Tensor, step_size: float,
                  num_steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        z_new = z.clone()
        p_new = p.clone()
        # Half step momentum
        grad = _grad_logp(self.logp_fn, z_new)
        p_new = p_new + 0.5 * step_size * grad
        # Full steps
        for _ in range(num_steps):
            z_new = z_new + step_size * (p_new * self.inv_mass_diag)
            grad = _grad_logp(self.logp_fn, z_new)
            if _ != num_steps - 1:
                p_new = p_new + step_size * grad
        # Final half step
        p_new = p_new + 0.5 * step_size * grad
        return z_new, p_new

    def sample(self,
               z0: Optional[torch.Tensor] = None,
               progress_bar: bool = False) -> torch.Tensor:
        cfg = self.cfg
        step_size = cfg.step_size
        L = cfg.num_steps
        N = cfg.num_samples
        burn = cfg.burn_in
        thin = max(1, cfg.thinning)

        if z0 is None:
            z = torch.zeros(self.dim, device=self.device)
        else:
            z = z0.to(self.device)
        z = z.view(-1)

        samples = []
        accepted = 0

        iterator = range(burn + N * thin)
        if progress_bar:
            try:
                from tqdm import tqdm
                iterator = tqdm(iterator, desc="HMC Sampling")
            except Exception:
                pass

        for it in iterator:
            p = torch.randn(self.dim, device=self.device) * torch.sqrt(
                self.mass_diag)
            current_logp = self.logp_fn(z.view(1, -1))
            if current_logp.dim() == 0:
                current_logp = current_logp.unsqueeze(0)
            current_logp = current_logp.sum()
            current_H = -current_logp + self._kinetic(p)

            z_prop, p_prop = self._leapfrog(z, p, step_size, L)
            proposed_logp = self.logp_fn(z_prop.view(1, -1))
            if proposed_logp.dim() == 0:
                proposed_logp = proposed_logp.unsqueeze(0)
            proposed_logp = proposed_logp.sum()
            proposed_H = -proposed_logp + self._kinetic(p_prop)

            accept_logprob = current_H - proposed_H
            if torch.log(torch.rand((), device=self.device)) < accept_logprob:
                z = z_prop.detach()
                accepted += 1
            # else keep current z

            if it >= burn and ((it - burn) % thin == 0):
                samples.append(z.detach().cpu())

        samples = torch.stack(samples, dim=0)
        acc_rate = accepted / float(burn + N * thin)
        return samples, acc_rate
