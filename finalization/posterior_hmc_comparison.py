"""Compare terminal MALA samples with retained HMC chain samples.

The comparison reuses the exact fixed ``z`` and checkpointed variational model
from ``posterior_mala_diagnostic.py``. MALA contributes only its terminal tail;
HMC contributes stratified retained draws from every chain.
"""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from .config import REPO_ROOT
from .posterior_mala_diagnostic import classical_effective_sample_size
from .score_approximation import (
    _build_runner,
    _load_checkpoint,
    _release_runner,
    build_cell_specs,
    gelman_rubin_rhat,
    posterior_log_prob_and_grad,
    seed_everything,
    stable_seed,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "configs"
    / "finalization"
    / "posterior_mala_hmc_x_shaped.yaml"
)


def load_posterior_hmc_config(
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


def posterior_hmc_samples(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    num_chains: int,
    burn_in_steps: int,
    samples_per_chain: int,
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
    trace_interval: int,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, float]]]:
    """Run parallel HMC chains for one fixed conditional epsilon posterior."""
    if z.ndim != 2 or z.shape[0] != 1:
        raise ValueError("z must have shape [1, z_dim].")
    if generating_epsilon.ndim != 2 or generating_epsilon.shape[0] != 1:
        raise ValueError(
            "generating_epsilon must have shape [1, epsilon_dim]."
        )
    if num_chains < 2:
        raise ValueError("HMC requires at least two chains.")
    if burn_in_steps < 0 or samples_per_chain < 4 or thinning < 1:
        raise ValueError("Invalid HMC burn-in, sample count, or thinning.")
    if step_size <= 0 or leapfrog_steps < 1:
        raise ValueError("Invalid HMC step size or leapfrog count.")
    if init_jitter_scale < 0:
        raise ValueError("init_jitter_scale must be non-negative.")
    if not 0 < target_acceptance < 1:
        raise ValueError("target_acceptance must lie in (0, 1).")
    if adaptation_rate < 0:
        raise ValueError("adaptation_rate must be non-negative.")
    if not 0 < min_step_size <= step_size <= max_step_size:
        raise ValueError(
            "Require min_step_size <= step_size <= max_step_size."
        )
    if divergence_threshold <= 0 or trace_interval < 1:
        raise ValueError("Invalid divergence threshold or trace interval.")

    device = generating_epsilon.device
    dtype = generating_epsilon.dtype
    epsilon_dim = generating_epsilon.shape[-1]
    z_chains = z.detach().expand(num_chains, -1)
    epsilon_current = generating_epsilon.detach().expand(
        num_chains,
        epsilon_dim,
    ).clone()
    jitter = torch.randn_like(epsilon_current)
    jitter[0].zero_()
    epsilon_current += init_jitter_scale * jitter
    initial_epsilon = epsilon_current.detach().cpu()
    current_log_prob, current_gradient = posterior_log_prob_and_grad(
        vi_model,
        epsilon_current,
        z_chains,
    )
    if (
        not torch.isfinite(current_log_prob).all()
        or not torch.isfinite(current_gradient).all()
    ):
        raise FloatingPointError("Initial HMC state is non-finite.")

    log_step = torch.full(
        (num_chains, 1),
        math.log(step_size),
        device=device,
        dtype=dtype,
    )
    min_log_step = math.log(min_step_size)
    max_log_step = math.log(max_step_size)
    accepted_total = torch.zeros(
        num_chains,
        device=device,
        dtype=torch.float64,
    )
    accepted_burn = torch.zeros_like(accepted_total)
    accepted_post = torch.zeros_like(accepted_total)
    divergence_count = torch.zeros_like(accepted_total)
    squared_jump_sum = torch.zeros_like(accepted_total)
    retained: list[torch.Tensor] = []
    trace: list[dict[str, float]] = []
    window_accepts = torch.zeros_like(accepted_total)
    window_count = 0
    retained_transition_count = samples_per_chain * thinning
    total_transitions = burn_in_steps + retained_transition_count

    def append_trace(transition: int) -> None:
        denominator = max(1, transition)
        trace.append({
            "transition": float(transition),
            "acceptance_rate": float(
                (accepted_total / denominator).mean().item()
            ),
            "window_acceptance_rate": float(
                (
                    window_accepts / max(1, window_count)
                ).mean().item()
            ),
            "mean_log_posterior": float(current_log_prob.mean().item()),
            "step_size_median": float(log_step.exp().median().item()),
            "epsilon_mean_norm": float(
                epsilon_current.mean(dim=0).norm().item()
            ),
            "epsilon_sd_mean": float(
                epsilon_current.std(dim=0, unbiased=True).mean().item()
            ),
        })

    append_trace(0)
    for transition in range(1, total_transitions + 1):
        transition_step = log_step.exp()
        epsilon_before = epsilon_current
        momentum_initial = torch.randn_like(epsilon_current)
        kinetic_initial = 0.5 * momentum_initial.square().sum(dim=-1)

        momentum = (
            momentum_initial
            + 0.5 * transition_step * current_gradient
        )
        epsilon_proposed = epsilon_current
        proposed_log_prob = current_log_prob
        proposed_gradient = current_gradient
        proposal_finite = torch.ones(
            num_chains,
            device=device,
            dtype=torch.bool,
        )
        for leapfrog_index in range(leapfrog_steps):
            epsilon_proposed = (
                epsilon_proposed + transition_step * momentum
            )
            proposed_log_prob, proposed_gradient = (
                posterior_log_prob_and_grad(
                    vi_model,
                    epsilon_proposed,
                    z_chains,
                )
            )
            proposal_finite &= (
                torch.isfinite(proposed_log_prob)
                & torch.isfinite(proposed_gradient).all(dim=-1)
            )
            if leapfrog_index != leapfrog_steps - 1:
                momentum = (
                    momentum + transition_step * proposed_gradient
                )
        momentum = momentum + (
            0.5 * transition_step * proposed_gradient
        )
        kinetic_proposed = 0.5 * momentum.square().sum(dim=-1)
        delta_h = (
            kinetic_proposed
            - proposed_log_prob
            - kinetic_initial
            + current_log_prob
        )
        finite = proposal_finite & torch.isfinite(delta_h)
        log_acceptance = torch.where(
            finite,
            (-delta_h).clamp(max=0.0),
            torch.full_like(delta_h, -torch.inf),
        )
        acceptance_probability = torch.exp(log_acceptance)
        accept = torch.log(torch.rand_like(log_acceptance)) < log_acceptance

        epsilon_current = torch.where(
            accept.unsqueeze(-1),
            epsilon_proposed,
            epsilon_current,
        ).detach()
        current_log_prob = torch.where(
            accept,
            proposed_log_prob,
            current_log_prob,
        ).detach()
        current_gradient = torch.where(
            accept.unsqueeze(-1),
            proposed_gradient,
            current_gradient,
        ).detach()

        accepted = accept.to(torch.float64)
        accepted_total += accepted
        window_accepts += accepted
        window_count += 1
        divergence_count += (
            (~finite) | (delta_h.abs() > divergence_threshold)
        ).to(torch.float64)
        squared_jump_sum += (
            epsilon_current - epsilon_before
        ).square().sum(dim=-1).to(torch.float64)

        if transition <= burn_in_steps:
            accepted_burn += accepted
            if adapt_step_size:
                gain = adaptation_rate / math.sqrt(transition)
                log_step = (
                    log_step
                    + gain
                    * (
                        acceptance_probability.detach().unsqueeze(-1)
                        - target_acceptance
                    )
                ).clamp(min=min_log_step, max=max_log_step)
        else:
            accepted_post += accepted
            retained_index = transition - burn_in_steps
            if retained_index % thinning == 0:
                retained.append(epsilon_current.detach().cpu())

        if transition % trace_interval == 0:
            append_trace(transition)
            window_accepts.zero_()
            window_count = 0

    if trace[-1]["transition"] != float(total_transitions):
        append_trace(total_transitions)
    samples = torch.stack(retained, dim=1).to(torch.float64)
    if samples.shape != (num_chains, samples_per_chain, epsilon_dim):
        raise RuntimeError(
            f"Unexpected HMC sample shape {tuple(samples.shape)}."
        )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Retained HMC samples are non-finite.")

    half = samples_per_chain // 2
    split = torch.cat(
        [samples[:, :half], samples[:, samples_per_chain - half:]],
        dim=0,
    )
    split_rhat = gelman_rubin_rhat(split.unsqueeze(0)).squeeze(0)
    ess = classical_effective_sample_size(samples)
    early = samples[:, :half].reshape(-1, epsilon_dim)
    late = samples[:, samples_per_chain - half:].reshape(
        -1,
        epsilon_dim,
    )
    pooled_sd = samples.reshape(-1, epsilon_dim).std(
        dim=0,
        unbiased=True,
    ).clamp_min(torch.finfo(torch.float64).eps)
    standardized_drift = (
        (late.mean(dim=0) - early.mean(dim=0)).abs() / pooled_sd
    )
    final_step = log_step.exp().squeeze(-1)
    post_steps = max(1, retained_transition_count)
    diagnostics: dict[str, Any] = {
        "num_chains": num_chains,
        "burn_in_steps": burn_in_steps,
        "samples_per_chain": samples_per_chain,
        "retained_samples_total": int(num_chains * samples_per_chain),
        "thinning": thinning,
        "initial_step_size": step_size,
        "leapfrog_steps": leapfrog_steps,
        "init_jitter_scale": init_jitter_scale,
        "adapt_step_size": adapt_step_size,
        "target_acceptance": target_acceptance,
        "acceptance_rate": float(
            (accepted_total / total_transitions).mean().item()
        ),
        "burn_in_acceptance_rate": float(
            (accepted_burn / max(1, burn_in_steps)).mean().item()
        ),
        "post_burn_acceptance_rate": float(
            (accepted_post / post_steps).mean().item()
        ),
        "post_burn_acceptance_min_chain": float(
            (accepted_post / post_steps).min().item()
        ),
        "divergence_fraction": float(
            divergence_count.sum().item()
            / (total_transitions * num_chains)
        ),
        "mean_squared_jump_distance": float(
            (squared_jump_sum / total_transitions).mean().item()
        ),
        "final_step_size_median": float(final_step.median().item()),
        "final_step_size_p05": float(
            torch.quantile(final_step, 0.05).item()
        ),
        "final_step_size_p95": float(
            torch.quantile(final_step, 0.95).item()
        ),
        "split_rhat": split_rhat.tolist(),
        "split_rhat_max": float(split_rhat.max().item()),
        "ess": ess.tolist(),
        "ess_min": float(ess.min().item()),
        "early_late_standardized_drift": standardized_drift.tolist(),
        "early_late_standardized_drift_max": float(
            standardized_drift.max().item()
        ),
        "initial_chain_epsilon": initial_epsilon.tolist(),
    }
    diagnostics["convergence_checks"] = {
        "split_rhat_max_le_1_01": (
            diagnostics["split_rhat_max"] <= 1.01
        ),
        "ess_min_ge_400": diagnostics["ess_min"] >= 400.0,
        "standardized_drift_max_le_0_10": (
            diagnostics["early_late_standardized_drift_max"] <= 0.10
        ),
        "divergence_fraction_le_0_01": (
            diagnostics["divergence_fraction"] <= 0.01
        ),
        "post_burn_acceptance_ge_0_60": (
            diagnostics["post_burn_acceptance_rate"] >= 0.60
        ),
    }
    diagnostics["convergence_pass"] = all(
        diagnostics["convergence_checks"].values()
    )
    return samples, diagnostics, trace


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _stratified_indices(
    chains: int,
    draws: int,
    max_samples: int,
) -> list[tuple[int, int]]:
    draws_per_chain = max(1, max_samples // chains)
    retained_per_chain = min(draws, draws_per_chain)
    draw_indices = np.linspace(
        0,
        draws - 1,
        retained_per_chain,
        dtype=np.int64,
    )
    return [
        (chain, int(draw))
        for chain in range(chains)
        for draw in draw_indices
    ]


def _write_comparison_figure(
    report_dir: Path,
    mala_tail: torch.Tensor,
    hmc_samples: torch.Tensor,
    hmc_diagnostics: dict[str, Any],
    hmc_trace: list[dict[str, float]],
    *,
    generating_epsilon: torch.Tensor,
    z: torch.Tensor,
    seed: int,
    epoch: int,
    max_plot_samples: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if mala_tail.shape[-1] != 2 or hmc_samples.shape[-1] != 2:
        raise ValueError("The comparison figure currently requires epsilon_dim=2.")
    mala_indices = _stratified_indices(
        mala_tail.shape[0],
        mala_tail.shape[1],
        max_plot_samples // 2,
    )
    hmc_indices = _stratified_indices(
        hmc_samples.shape[0],
        hmc_samples.shape[1],
        max_plot_samples // 2,
    )
    mala_plot = np.asarray([
        mala_tail[chain, draw].numpy()
        for chain, draw in mala_indices
    ])
    hmc_plot = np.asarray([
        hmc_samples[chain, draw].numpy()
        for chain, draw in hmc_indices
    ])
    mala_flat = mala_tail.reshape(-1, 2).numpy()
    hmc_flat = hmc_samples.reshape(-1, 2).numpy()
    hmc_initial = np.asarray(
        hmc_diagnostics["initial_chain_epsilon"],
        dtype=np.float64,
    )
    generating = generating_epsilon.numpy().reshape(-1)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13, 10),
        constrained_layout=True,
    )
    ax = axes[0, 0]
    ax.scatter(
        mala_plot[:, 0],
        mala_plot[:, 1],
        s=7,
        alpha=0.2,
        label="MALA terminal tail",
    )
    ax.scatter(
        hmc_plot[:, 0],
        hmc_plot[:, 1],
        s=7,
        alpha=0.2,
        label="HMC chain samples",
    )
    ax.scatter(
        hmc_initial[:, 0],
        hmc_initial[:, 1],
        s=28,
        marker="x",
        linewidths=0.8,
        label="HMC chain starts",
    )
    ax.scatter(
        generating[0],
        generating[1],
        color="black",
        marker="*",
        s=100,
        label="generating epsilon",
        zorder=5,
    )
    ax.set_xlabel("epsilon[0]")
    ax.set_ylabel("epsilon[1]")
    ax.set_title("Conditional posterior samples")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    transitions = np.asarray([
        row["transition"] for row in hmc_trace
    ])
    acceptance = np.asarray([
        row["acceptance_rate"] for row in hmc_trace
    ])
    step_sizes = np.asarray([
        row["step_size_median"] for row in hmc_trace
    ])
    ax.plot(transitions, acceptance, label="HMC acceptance")
    ax.set_xlabel("HMC transition")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0.0, 1.02)
    twin = ax.twinx()
    twin.plot(
        transitions,
        step_sizes,
        color="tab:orange",
        label="median step size",
    )
    twin.set_ylabel("HMC step size")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = twin.get_legend_handles_labels()
    ax.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
    )
    ax.set_title("HMC adaptation and acceptance")

    for dimension, ax in enumerate(axes[1]):
        ax.hist(
            mala_flat[:, dimension],
            bins=60,
            density=True,
            alpha=0.45,
            label="MALA terminal tail",
        )
        ax.hist(
            hmc_flat[:, dimension],
            bins=60,
            density=True,
            alpha=0.45,
            label="HMC chain samples",
        )
        ax.axvline(
            generating[dimension],
            color="black",
            linestyle="--",
            linewidth=1.0,
        )
        ax.set_xlabel(f"epsilon[{dimension}]")
        ax.set_ylabel("density")
        ax.set_title(
            f"HMC epsilon[{dimension}]: "
            f"R-hat={hmc_diagnostics['split_rhat'][dimension]:.3f}, "
            f"ESS={hmc_diagnostics['ess'][dimension]:.0f}"
        )
        if dimension == 0:
            ax.legend(frameon=False)

    fig.suptitle(
        "MALA tail versus HMC: DSIVI x_shaped "
        f"seed {seed}, epoch {epoch}, z="
        f"({float(z[0, 0]):.3f}, {float(z[0, 1]):.3f})"
    )
    fig.savefig(
        report_dir / "posterior_mala_hmc_comparison.png",
        dpi=180,
    )
    plt.close(fig)


def _write_comparison_report(
    report_dir: Path,
    *,
    mala_samples: torch.Tensor,
    mala_diagnostics: dict[str, Any],
    hmc_samples: torch.Tensor,
    hmc_diagnostics: dict[str, Any],
    hmc_trace: list[dict[str, float]],
    generating_epsilon: torch.Tensor,
    z: torch.Tensor,
    metadata: dict[str, Any],
    mala_tail_draws_per_chain: int,
    max_plot_samples: int,
    max_csv_samples: int,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    tail_draws = min(mala_samples.shape[1], mala_tail_draws_per_chain)
    mala_tail = mala_samples[:, -tail_draws:]
    _write_comparison_figure(
        report_dir,
        mala_tail,
        hmc_samples,
        hmc_diagnostics,
        hmc_trace,
        generating_epsilon=generating_epsilon,
        z=z,
        seed=int(metadata["seed"]),
        epoch=int(metadata["epoch"]),
        max_plot_samples=max_plot_samples,
    )

    payload = {
        **metadata,
        "z": z[0].tolist(),
        "generating_epsilon": generating_epsilon[0].tolist(),
        "mala_tail_draws_per_chain": tail_draws,
        "mala": mala_diagnostics,
        "hmc": hmc_diagnostics,
    }
    with (report_dir / "posterior_mala_hmc_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_csv(report_dir / "posterior_hmc_trace.csv", hmc_trace)
    mala_indices = _stratified_indices(
        mala_tail.shape[0],
        mala_tail.shape[1],
        max_csv_samples // 2,
    )
    hmc_indices = _stratified_indices(
        hmc_samples.shape[0],
        hmc_samples.shape[1],
        max_csv_samples // 2,
    )
    rows: list[dict[str, Any]] = []
    for sampler, values, indices in [
        ("MALA_tail", mala_tail, mala_indices),
        ("HMC", hmc_samples, hmc_indices),
    ]:
        for chain, draw in indices:
            rows.append({
                "sampler": sampler,
                "chain": chain,
                "draw": draw,
                "epsilon_0": f"{float(values[chain, draw, 0]):.8f}",
                "epsilon_1": f"{float(values[chain, draw, 1]):.8f}",
            })
    _write_csv(
        report_dir / "posterior_mala_hmc_samples.csv",
        rows,
    )

    lines = [
        "# MALA-tail versus HMC posterior-epsilon comparison",
        "",
        (
            f"DSIVI `x_shaped`, seed {metadata['seed']}, epoch "
            f"{metadata['epoch']}, fixed z. MALA shows only its last "
            f"{tail_draws} retained draws per chain; HMC diagnostics use "
            f"all {hmc_diagnostics['samples_per_chain']} retained draws "
            "from every chain."
        ),
        "",
        "| Sampler | Acceptance | Split R-hat max | ESS min | Converged |",
        "|---|---:|---:|---:|:---:|",
        (
            f"| MALA | {mala_diagnostics['acceptance_rate']:.8f} | "
            f"{mala_diagnostics['split_rhat_max']:.8f} | "
            f"{mala_diagnostics['ess_min']:.2f} | "
            f"{'yes' if mala_diagnostics['convergence_pass'] else 'no'} |"
        ),
        (
            f"| HMC | {hmc_diagnostics['acceptance_rate']:.8f} | "
            f"{hmc_diagnostics['split_rhat_max']:.8f} | "
            f"{hmc_diagnostics['ess_min']:.2f} | "
            f"{'yes' if hmc_diagnostics['convergence_pass'] else 'no'} |"
        ),
        "",
        (
            "![MALA-tail versus HMC samples]"
            "(posterior_mala_hmc_comparison.png)"
        ),
        "",
    ]
    (report_dir / "posterior_mala_hmc_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_posterior_hmc_comparison(cfg: DictConfig) -> dict[str, Any]:
    specs = build_cell_specs(cfg)
    if len(specs) != 1:
        raise RuntimeError(
            f"Posterior HMC comparison requires exactly one cell; got "
            f"{len(specs)}."
        )
    spec = specs[0]
    mala_path = REPO_ROOT / str(cfg.input.mala_runtime_path)
    mala_payload = torch.load(
        mala_path,
        map_location="cpu",
        weights_only=False,
    )
    mala_samples = mala_payload["samples"].to(torch.float64)
    mala_z = mala_payload["z"].to(torch.float64)
    mala_generating = mala_payload["generating_epsilon"].to(torch.float64)
    mala_diagnostics = mala_payload["diagnostics"]

    runner: Any | None = None
    try:
        runner = _build_runner(spec.record, cfg)
        _load_checkpoint(runner, spec)
        if str(runner.device) != "cuda":
            raise RuntimeError("Production posterior HMC requires CUDA.")
        z_index = int(cfg.evaluation.z_index)
        forward_seed = stable_seed(spec.key, "posterior_mala_forward")
        seed_everything(forward_seed, use_cuda=True)
        epsilon_bank, z_bank = runner.vi_model.sampling(num=z_index + 1)
        generating_epsilon = epsilon_bank[z_index:z_index + 1]
        z = z_bank[z_index:z_index + 1]
        if (
            not torch.allclose(z.cpu().to(torch.float64), mala_z)
            or not torch.allclose(
                generating_epsilon.cpu().to(torch.float64),
                mala_generating,
            )
        ):
            raise RuntimeError(
                "Saved MALA trajectory does not match the selected fixed z."
            )

        hmc_cfg = cfg.evaluation.hmc
        sampler_seed = stable_seed(spec.key, "posterior_hmc_sampler_v1")
        seed_everything(sampler_seed, use_cuda=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        hmc_samples, hmc_diagnostics, hmc_trace = posterior_hmc_samples(
            runner.vi_model,
            z,
            generating_epsilon,
            num_chains=int(hmc_cfg.num_chains),
            burn_in_steps=int(hmc_cfg.burn_in_steps),
            samples_per_chain=int(hmc_cfg.samples_per_chain),
            thinning=int(hmc_cfg.thinning),
            step_size=float(hmc_cfg.step_size),
            leapfrog_steps=int(hmc_cfg.leapfrog_steps),
            init_jitter_scale=float(hmc_cfg.init_jitter_scale),
            adapt_step_size=bool(hmc_cfg.adapt_step_size),
            target_acceptance=float(hmc_cfg.target_acceptance),
            adaptation_rate=float(hmc_cfg.adaptation_rate),
            min_step_size=float(hmc_cfg.min_step_size),
            max_step_size=float(hmc_cfg.max_step_size),
            divergence_threshold=float(hmc_cfg.divergence_threshold),
            trace_interval=int(hmc_cfg.trace_interval),
        )
        torch.cuda.synchronize()
        hmc_diagnostics["runtime_sec"] = time.perf_counter() - started

        runtime_dir = REPO_ROOT / str(cfg.output.runtime_dir)
        report_dir = REPO_ROOT / str(cfg.output.report_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "samples": hmc_samples,
                "z": z.detach().cpu(),
                "generating_epsilon": generating_epsilon.detach().cpu(),
                "diagnostics": hmc_diagnostics,
                "trace": hmc_trace,
            },
            runtime_dir / "posterior_hmc_samples.pt",
        )
        metadata = {
            "method": spec.record.method,
            "target": spec.record.target,
            "seed": spec.record.seed,
            "epoch": spec.epoch,
            "checkpoint_dir": spec.checkpoint_dir.as_posix(),
            "epsilon_dim": int(hmc_samples.shape[-1]),
            "z_dim": int(z.shape[-1]),
            "forward_seed": forward_seed,
            "hmc_sampler_seed": sampler_seed,
            "gpu_name": torch.cuda.get_device_name(),
        }
        _write_comparison_report(
            report_dir,
            mala_samples=mala_samples,
            mala_diagnostics=mala_diagnostics,
            hmc_samples=hmc_samples,
            hmc_diagnostics=hmc_diagnostics,
            hmc_trace=hmc_trace,
            generating_epsilon=generating_epsilon.detach().cpu().to(
                torch.float64
            ),
            z=z.detach().cpu().to(torch.float64),
            metadata=metadata,
            mala_tail_draws_per_chain=int(
                cfg.evaluation.mala_tail_draws_per_chain
            ),
            max_plot_samples=int(cfg.output.max_plot_samples),
            max_csv_samples=int(cfg.output.max_csv_samples),
        )
        return {**metadata, **hmc_diagnostics}
    finally:
        _release_runner(runner)
