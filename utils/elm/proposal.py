from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig, OmegaConf

from models.reverse_model import (
    BaseReverseConditionalModel,
    ConditionalGaussianReverse,
    ConditionalMixtureOfGaussianReverse,
    ReverseModel,
)
from utils.logging import get_logger

from .types import ReverseProposalFit


logger = get_logger()

PROPOSAL_TYPE_ALIASES = {
    "gaussian": "ConditionalGaussianReverse",
    "mog": "ConditionalMixtureOfGaussianReverse",
    "realnvp": "ConditionalRealNVP",
}
PROPOSAL_TYPE_CANONICAL_NAMES = {
    value: key for key, value in PROPOSAL_TYPE_ALIASES.items()
}


def module_dtype(module: torch.nn.Module) -> torch.dtype:
    parameter = next(module.parameters(), None)
    if parameter is not None:
        return parameter.dtype
    buffer = next(module.buffers(), None)
    if buffer is not None:
        return buffer.dtype
    return torch.float32


def canonical_reverse_model_type(proposal_type: str) -> tuple[str, str]:
    canonical = PROPOSAL_TYPE_ALIASES.get(proposal_type, proposal_type)
    if canonical not in ReverseModel:
        supported = ", ".join(sorted(PROPOSAL_TYPE_ALIASES))
        raise ValueError(
            f"Unsupported proposal_type={proposal_type!r}. Supported aliases: {supported}."
        )
    return canonical, PROPOSAL_TYPE_CANONICAL_NAMES.get(canonical, canonical)


def default_reverse_model_config_path(canonical_type: str) -> Path:
    return Path("configs") / "reverse_models" / f"{canonical_type}.yaml"


def resolve_reverse_model_config(
    vi_model: torch.nn.Module,
    proposal_type: str,
    proposal_config_path: str | Path | None,
) -> tuple[str, str, Path, DictConfig, dict[str, Any]]:
    canonical_type, proposal_alias = canonical_reverse_model_type(proposal_type)
    config_path = (
        Path(proposal_config_path)
        if proposal_config_path is not None
        else default_reverse_model_config_path(canonical_type)
    )
    if not config_path.is_file():
        raise FileNotFoundError(f"Reverse proposal config not found: {config_path}")

    device = getattr(vi_model, "device", next(vi_model.parameters()).device)
    raw_config = OmegaConf.load(config_path)
    raw_config.z_dim = int(vi_model.z_dim)
    raw_config.epsilon_dim = int(vi_model.epsilon_dim)
    raw_config.device = str(device)
    if "logit" in raw_config:
        raw_config.logit = bool(getattr(vi_model, "uniform", False))
    resolved_container = OmegaConf.to_container(raw_config, resolve=True)
    if not isinstance(resolved_container, dict):
        raise RuntimeError(
            f"Resolved reverse proposal config must be a mapping: {config_path}"
        )
    resolved_cfg: DictConfig = OmegaConf.create(resolved_container)  # type: ignore[assignment]
    resolved_cfg.device = str(device)
    return canonical_type, proposal_alias, config_path, resolved_cfg, resolved_container


def gaussian_reverse_cache(
    reverse_model: torch.nn.Module,
    *,
    dtype: torch.dtype,
) -> dict[str, torch.Tensor] | None:
    if not isinstance(reverse_model, ConditionalGaussianReverse):
        return None
    A, b, cond_cov = reverse_model._conditional_params()
    A = A.to(dtype=dtype)
    b = b.to(dtype=dtype)
    cond_cov = cond_cov.to(dtype=dtype)
    chol = torch.linalg.cholesky(cond_cov)
    precision = torch.cholesky_inverse(chol)
    log_det = 2.0 * torch.log(torch.diagonal(chol)).sum()
    return {
        "A": A,
        "b": b,
        "chol": chol,
        "precision": precision,
        "log_det": log_det,
    }


def gaussian_reverse_mean(
    z: torch.Tensor,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    return torch.matmul(z, cache["A"].transpose(0, 1)) + cache["b"]


def gaussian_reverse_log_prob(
    epsilon: torch.Tensor,
    z: torch.Tensor,
    cache: dict[str, torch.Tensor],
) -> torch.Tensor:
    mean = gaussian_reverse_mean(z, cache)
    diff = epsilon - mean
    quad = torch.einsum("...i,ij,...j->...", diff, cache["precision"], diff)
    dim = epsilon.shape[-1]
    return -0.5 * (
        dim * torch.log(torch.tensor(2.0 * torch.pi, device=epsilon.device, dtype=epsilon.dtype))
        + cache["log_det"]
        + quad
    )


def reverse_model_log_prob(
    reverse_model: BaseReverseConditionalModel,
    epsilon: torch.Tensor,
    z: torch.Tensor,
    proposal_cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    if proposal_cache is not None:
        return gaussian_reverse_log_prob(epsilon, z, proposal_cache)

    model_dtype = module_dtype(reverse_model)
    epsilon_flat = epsilon.reshape(-1, epsilon.shape[-1]).to(
        device=reverse_model.device,
        dtype=model_dtype,
    )
    z_flat = z.reshape(-1, z.shape[-1]).to(
        device=reverse_model.device,
        dtype=model_dtype,
    )
    log_prob = reverse_model.log_prob(epsilon_flat, z_flat)
    return log_prob.reshape(epsilon.shape[:-1]).to(
        device=epsilon.device,
        dtype=epsilon.dtype,
    )


def sample_vi_joint(
    vi_model: torch.nn.Module,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        epsilon_samples, z_samples = vi_model.sampling(num=num_samples)
    return epsilon_samples.detach(), z_samples.detach()


def estimate_fit_nll(
    reverse_model: BaseReverseConditionalModel,
    epsilon: torch.Tensor,
    z: torch.Tensor,
    *,
    batch_size: int,
    proposal_cache: dict[str, torch.Tensor] | None = None,
) -> float:
    losses: list[torch.Tensor] = []
    epsilon = epsilon.reshape(-1, epsilon.shape[-1])
    z = z.reshape(-1, z.shape[-1])
    with torch.no_grad():
        for start in range(0, int(epsilon.shape[0]), batch_size):
            eps_batch = epsilon[start:start + batch_size]
            z_batch = z[start:start + batch_size]
            log_prob = reverse_model_log_prob(
                reverse_model,
                eps_batch,
                z_batch,
                proposal_cache=proposal_cache,
            )
            losses.append((-log_prob).detach().cpu())
    return float(torch.cat(losses).mean().item())


def build_reverse_proposal(
    vi_model: torch.nn.Module,
    *,
    proposal_type: str,
    proposal_config_path: str | Path | None,
) -> tuple[BaseReverseConditionalModel, dict[str, Any], dict[str, Any]]:
    canonical_type, proposal_alias, config_path, proposal_cfg, resolved_config = (
        resolve_reverse_model_config(
            vi_model,
            proposal_type,
            proposal_config_path,
        )
    )
    reverse_model = ReverseModel[canonical_type](config=proposal_cfg).to(proposal_cfg.device)
    diagnostics: dict[str, Any] = {
        "proposal_type": proposal_alias,
        "proposal_class": canonical_type,
        "proposal_config_path": config_path.as_posix(),
    }
    if isinstance(reverse_model, ConditionalMixtureOfGaussianReverse):
        diagnostics["num_components"] = int(reverse_model.num_components)
    return reverse_model, diagnostics, resolved_config


def collect_vi_joint_samples(
    vi_model: torch.nn.Module,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return sample_vi_joint(vi_model, num_samples)


def run_direct_fit(
    reverse_model: BaseReverseConditionalModel,
    *,
    epsilon_samples: torch.Tensor,
    z_samples: torch.Tensor,
) -> dict[str, Any]:
    fit_nll, fit_steps = reverse_model.fit(
        epsilon_samples,
        z_samples,
        initialize=True,
    )
    return {
        "fit_nll": float(fit_nll),
        "fit_steps": int(fit_steps),
    }


def run_optimizer_fit(
    reverse_model: BaseReverseConditionalModel,
    vi_model: torch.nn.Module,
    *,
    fit_batch_size: int,
    fit_epochs: int,
    fit_lr: float | None,
    proposal_cfg: DictConfig,
    log_every: int,
) -> dict[str, Any]:
    resolved_lr = float(
        fit_lr
        if fit_lr is not None
        else proposal_cfg.get("lr", proposal_cfg.get("warmup", {}).get("lr", 1.0e-2))
    )
    optimizer = torch.optim.Adam(reverse_model.parameters(), lr=resolved_lr)
    losses: list[float] = []
    best_loss = float("inf")
    optimizer_steps = 0
    reverse_model.train()

    for epoch in range(1, fit_epochs + 1):
        epsilon_samples, z_samples = collect_vi_joint_samples(vi_model, fit_batch_size)
        optimizer.zero_grad()
        log_prob = reverse_model.log_prob(epsilon_samples, z_samples)
        loss = -torch.mean(log_prob)
        loss_value = float(loss.item())
        losses.append(loss_value)
        if torch.isfinite(loss):
            loss.backward()
            optimizer.step()
            optimizer_steps += 1
            best_loss = min(best_loss, loss_value)
        else:
            logger.warning(
                "Skipping non-finite reverse proposal optimizer step for %s at epoch %s.",
                type(reverse_model).__name__,
                epoch,
            )
        if log_every > 0 and (epoch == 1 or epoch % log_every == 0 or epoch == fit_epochs):
            logger.info(
                "Reverse proposal fit [%s] epoch=%s/%s loss=%.6f best=%.6f",
                type(reverse_model).__name__,
                epoch,
                fit_epochs,
                loss_value,
                best_loss,
            )

    reverse_model.eval()
    return {
        "fit_steps": int(optimizer_steps),
        "fit_lr": float(resolved_lr),
        "fit_loss_initial": float(losses[0]) if losses else float("nan"),
        "fit_loss_final": float(losses[-1]) if losses else float("nan"),
        "fit_loss_best": float(best_loss) if losses else float("nan"),
    }


def finalize_reverse_proposal_fit(
    reverse_model: BaseReverseConditionalModel,
    vi_model: torch.nn.Module,
    *,
    diagnostics: dict[str, Any],
    fit_batch_size: int,
    num_fit_samples: int,
) -> tuple[dict[str, Any], dict[str, torch.Tensor] | None]:
    eval_epsilon, eval_z = collect_vi_joint_samples(vi_model, num_fit_samples)
    proposal_cache = gaussian_reverse_cache(reverse_model, dtype=module_dtype(vi_model))
    fit_nll = estimate_fit_nll(
        reverse_model,
        eval_epsilon,
        eval_z,
        batch_size=fit_batch_size,
        proposal_cache=proposal_cache,
    )
    diagnostics["fit_nll"] = float(fit_nll)
    return diagnostics, proposal_cache


def fit_reverse_proposal(
    vi_model: torch.nn.Module,
    *,
    proposal_type: str = "gaussian",
    proposal_config_path: str | Path | None = None,
    num_fit_samples: int = 32768,
    fit_batch_size: int = 8192,
    fit_epochs: int = 1000,
    fit_lr: float | None = None,
    log_every: int = 100,
) -> ReverseProposalFit:
    if num_fit_samples < 2:
        raise ValueError("num_fit_samples must be at least 2.")
    if fit_batch_size < 1:
        raise ValueError("fit_batch_size must be at least 1.")
    if fit_epochs < 1:
        raise ValueError("fit_epochs must be at least 1.")

    canonical_type, proposal_alias, config_path, proposal_cfg, resolved_config = (
        resolve_reverse_model_config(
            vi_model,
            proposal_type,
            proposal_config_path,
        )
    )
    reverse_model = ReverseModel[canonical_type](config=proposal_cfg).to(proposal_cfg.device)
    use_optimizer = bool(proposal_cfg.get("use_optimizer", True))
    fit_mode = "optimizer" if use_optimizer else "direct_fit"

    diagnostics: dict[str, Any] = {
        "proposal_type": proposal_alias,
        "proposal_class": canonical_type,
        "proposal_config_path": config_path.as_posix(),
        "fit_mode": fit_mode,
        "fit_samples": int(num_fit_samples),
        "fit_batch_size": int(fit_batch_size if use_optimizer else num_fit_samples),
        "fit_epochs": int(fit_epochs if use_optimizer else 1),
    }
    if isinstance(reverse_model, ConditionalMixtureOfGaussianReverse):
        diagnostics["num_components"] = int(reverse_model.num_components)

    was_vi_training = vi_model.training
    vi_model.eval()
    start = time.perf_counter()

    try:
        if use_optimizer:
            diagnostics.update(
                run_optimizer_fit(
                    reverse_model,
                    vi_model,
                    fit_batch_size=fit_batch_size,
                    fit_epochs=fit_epochs,
                    fit_lr=fit_lr,
                    proposal_cfg=proposal_cfg,
                    log_every=log_every,
                )
            )
        else:
            epsilon_samples, z_samples = collect_vi_joint_samples(vi_model, num_fit_samples)
            diagnostics.update(
                run_direct_fit(
                    reverse_model,
                    epsilon_samples=epsilon_samples,
                    z_samples=z_samples,
                )
            )
            diagnostics["fit_lr"] = None

        reverse_model.eval()
        diagnostics, proposal_cache = finalize_reverse_proposal_fit(
            reverse_model,
            vi_model,
            diagnostics=diagnostics,
            fit_batch_size=fit_batch_size,
            num_fit_samples=num_fit_samples,
        )
    finally:
        if was_vi_training:
            vi_model.train()

    diagnostics["fit_runtime_sec"] = float(time.perf_counter() - start)

    return ReverseProposalFit(
        reverse_model=reverse_model,
        proposal_type=proposal_alias,
        fit_mode=fit_mode,
        diagnostics=diagnostics,
        cache=proposal_cache,
        resolved_config=resolved_config,
    )


def save_reverse_proposal_fit(
    output_dir: str | Path,
    proposal_fit: ReverseProposalFit,
    *,
    save_state: bool = False,
) -> tuple[Path, Path | None]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_json_path = output_dir / "proposal_fit.json"
    fit_json_path.write_text(
        json.dumps(
            {
                **proposal_fit.diagnostics,
                "resolved_config": proposal_fit.resolved_config,
            },
            indent=2,
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )

    state_path: Path | None = None
    if save_state:
        state_path = output_dir / "proposal_state.pt"
        torch.save(
            {
                "proposal_type": proposal_fit.proposal_type,
                "fit_mode": proposal_fit.fit_mode,
                "diagnostics": proposal_fit.diagnostics,
                "resolved_config": proposal_fit.resolved_config,
                "state_dict": proposal_fit.reverse_model.state_dict(),
            },
            state_path,
        )
    return fit_json_path, state_path
