"""Posterior-MALA diagnostics for a checkpointed semi-implicit model.

The target density is the conditional auxiliary posterior

    q_phi(epsilon | z) propto q(epsilon) q_phi(z | epsilon).

This module intentionally keeps the diagnostic small: it selects one
reproducible ``(epsilon, z)`` pair, evolves multiple MALA chains for that fixed
``z``, and writes convergence diagnostics plus a dimension-aware epsilon
visualization.
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
    REPO_ROOT / "configs" / "finalization" / "posterior_mala_x_shaped.yaml"
)


def load_posterior_mala_config(
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


def _validate_mala_inputs(
    *,
    num_chains: int,
    num_steps: int,
    burn_in_steps: int,
    thinning: int,
    step_size: float,
    init_jitter_scale: float,
    trace_interval: int,
) -> None:
    if num_chains < 2:
        raise ValueError("MALA requires at least two chains.")
    if num_steps < 1:
        raise ValueError("num_steps must be positive.")
    if not 0 <= burn_in_steps < num_steps:
        raise ValueError("burn_in_steps must lie in [0, num_steps).")
    if thinning < 1:
        raise ValueError("thinning must be positive.")
    if (num_steps - burn_in_steps) // thinning < 4:
        raise ValueError("MALA must retain at least four draws per chain.")
    if step_size <= 0:
        raise ValueError("step_size must be positive.")
    if init_jitter_scale < 0:
        raise ValueError("init_jitter_scale must be non-negative.")
    if trace_interval < 1:
        raise ValueError("trace_interval must be positive.")


def classical_effective_sample_size(samples: torch.Tensor) -> torch.Tensor:
    """Estimate per-coordinate ESS from samples shaped ``[C, S, D]``.

    This is a classical multi-chain autocorrelation ESS with Geyer's initial
    positive sequence. It is deliberately reported as a mixing diagnostic, not
    as proof that all posterior modes were discovered.
    """
    if samples.ndim != 3:
        raise ValueError("samples must have shape [chains, draws, dimensions].")
    chains, draws, dimensions = samples.shape
    if chains < 2 or draws < 4:
        raise ValueError("ESS requires at least two chains and four draws.")

    values = samples.detach().to(dtype=torch.float64, device="cpu")
    centered = values - values.mean(dim=1, keepdim=True)
    fft_size = 1 << (2 * draws - 1).bit_length()
    spectrum = torch.fft.rfft(centered, n=fft_size, dim=1)
    autocov = torch.fft.irfft(
        spectrum * spectrum.conj(),
        n=fft_size,
        dim=1,
    )[:, :draws, :].real
    denominators = torch.arange(
        draws,
        0,
        -1,
        dtype=torch.float64,
    ).view(1, draws, 1)
    autocov = autocov / denominators

    within = values.var(dim=1, unbiased=True).mean(dim=0)
    between = draws * values.mean(dim=1).var(dim=0, unbiased=True)
    variance_plus = ((draws - 1.0) / draws) * within + between / draws
    mean_autocov = autocov.mean(dim=0)

    rho = torch.zeros(draws, dimensions, dtype=torch.float64)
    rho[0] = 1.0
    valid = variance_plus > 0
    if draws > 1:
        rho[1:, valid] = (
            1.0
            - (
                within[valid].unsqueeze(0)
                - mean_autocov[1:, valid]
            )
            / variance_plus[valid].unsqueeze(0)
        )

    tau = torch.ones(dimensions, dtype=torch.float64)
    for dimension in range(dimensions):
        if not valid[dimension]:
            tau[dimension] = float("inf")
            continue
        positive_sum = 0.0
        lag = 1
        previous_pair = float("inf")
        while lag + 1 < draws:
            pair = float(
                (rho[lag, dimension] + rho[lag + 1, dimension]).item()
            )
            if pair <= 0:
                break
            pair = min(pair, previous_pair)
            positive_sum += pair
            previous_pair = pair
            lag += 2
        tau[dimension] = max(1.0, 1.0 + 2.0 * positive_sum)

    total = float(chains * draws)
    return (total / tau).clamp(max=total)


def posterior_mala_samples(
    vi_model: torch.nn.Module,
    z: torch.Tensor,
    generating_epsilon: torch.Tensor,
    *,
    num_chains: int,
    num_steps: int,
    burn_in_steps: int,
    thinning: int,
    step_size: float,
    init_jitter_scale: float,
    trace_interval: int,
    snapshot_steps: list[int] | None = None,
) -> tuple[torch.Tensor, dict[str, Any], list[dict[str, float]]]:
    """Run parallel MALA chains for one fixed ``z``.

    The proposal convention matches the earlier Langevin analysis:

    ``epsilon' = epsilon + 0.5 * h * grad log pi(epsilon)
                 + sqrt(h) * Normal(0, I)``.

    The asymmetric forward/reverse Gaussian proposal densities are included in
    the Metropolis-Hastings correction.
    """
    _validate_mala_inputs(
        num_chains=num_chains,
        num_steps=num_steps,
        burn_in_steps=burn_in_steps,
        thinning=thinning,
        step_size=step_size,
        init_jitter_scale=init_jitter_scale,
        trace_interval=trace_interval,
    )
    if z.ndim != 2 or z.shape[0] != 1:
        raise ValueError("z must have shape [1, z_dim].")
    if generating_epsilon.ndim != 2 or generating_epsilon.shape[0] != 1:
        raise ValueError(
            "generating_epsilon must have shape [1, epsilon_dim]."
        )

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
    epsilon_current = epsilon_current + init_jitter_scale * jitter
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
        raise FloatingPointError("Initial MALA state is non-finite.")

    accepted_total = torch.zeros(
        num_chains,
        dtype=torch.float64,
        device=device,
    )
    accepted_burn = torch.zeros_like(accepted_total)
    accepted_post = torch.zeros_like(accepted_total)
    invalid_total = torch.zeros_like(accepted_total)
    squared_jump_sum = torch.zeros_like(accepted_total)
    retained: list[torch.Tensor] = []
    trace: list[dict[str, float]] = []
    requested_snapshots = set(snapshot_steps or [])
    requested_snapshots.update({0, num_steps})
    snapshots: dict[int, torch.Tensor] = {0: initial_epsilon}

    sqrt_step = math.sqrt(step_size)
    inverse_two_step = 0.5 / step_size
    window_accepts = torch.zeros_like(accepted_total)
    window_count = 0

    def append_trace(step: int) -> None:
        denominator = max(1, step)
        trace.append({
            "step": float(step),
            "acceptance_rate": float(
                (accepted_total / denominator).mean().item()
            ),
            "window_acceptance_rate": float(
                (
                    window_accepts / max(1, window_count)
                ).mean().item()
            ),
            "mean_log_posterior": float(current_log_prob.mean().item()),
            "epsilon_mean_norm": float(
                epsilon_current.mean(dim=0).norm().item()
            ),
            "epsilon_sd_mean": float(
                epsilon_current.std(dim=0, unbiased=True).mean().item()
            ),
        })

    append_trace(0)
    for step in range(1, num_steps + 1):
        epsilon_before = epsilon_current
        forward_mean = (
            epsilon_current + 0.5 * step_size * current_gradient
        )
        epsilon_proposed = (
            forward_mean + sqrt_step * torch.randn_like(epsilon_current)
        )
        proposed_log_prob, proposed_gradient = posterior_log_prob_and_grad(
            vi_model,
            epsilon_proposed,
            z_chains,
        )

        forward_residual = epsilon_proposed - forward_mean
        reverse_mean = (
            epsilon_proposed + 0.5 * step_size * proposed_gradient
        )
        reverse_residual = epsilon_current - reverse_mean
        log_proposal_ratio = -inverse_two_step * (
            reverse_residual.square().sum(dim=-1)
            - forward_residual.square().sum(dim=-1)
        )
        log_ratio = (
            proposed_log_prob - current_log_prob + log_proposal_ratio
        )
        finite = (
            torch.isfinite(log_ratio)
            & torch.isfinite(proposed_log_prob)
            & torch.isfinite(proposed_gradient).all(dim=-1)
        )
        log_acceptance = torch.where(
            finite,
            log_ratio.clamp(max=0.0),
            torch.full_like(log_ratio, -torch.inf),
        )
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
        invalid_total += (~finite).to(torch.float64)
        squared_jump_sum += (
            epsilon_current - epsilon_before
        ).square().sum(dim=-1).to(torch.float64)
        if step <= burn_in_steps:
            accepted_burn += accepted
        else:
            accepted_post += accepted
            if (step - burn_in_steps) % thinning == 0:
                retained.append(epsilon_current.detach().cpu())

        if step in requested_snapshots:
            snapshots[step] = epsilon_current.detach().cpu()
        if step % trace_interval == 0 or step == num_steps:
            append_trace(step)
            window_accepts.zero_()
            window_count = 0

    samples = torch.stack(retained, dim=1).to(torch.float64)
    draws = samples.shape[1]
    expected_draws = (num_steps - burn_in_steps) // thinning
    if draws != expected_draws:
        raise RuntimeError(
            f"Retained {draws} MALA draws; expected {expected_draws}."
        )
    if not torch.isfinite(samples).all():
        raise FloatingPointError("Retained MALA samples are non-finite.")

    half = draws // 2
    split = torch.cat(
        [samples[:, :half], samples[:, draws - half:]],
        dim=0,
    )
    split_rhat = gelman_rubin_rhat(split.unsqueeze(0)).squeeze(0)
    ess = classical_effective_sample_size(samples)
    early = samples[:, :half].reshape(-1, epsilon_dim)
    late = samples[:, draws - half:].reshape(-1, epsilon_dim)
    early_mean = early.mean(dim=0)
    late_mean = late.mean(dim=0)
    pooled_sd = samples.reshape(-1, epsilon_dim).std(
        dim=0,
        unbiased=True,
    ).clamp_min(torch.finfo(torch.float64).eps)
    standardized_drift = (late_mean - early_mean).abs() / pooled_sd

    post_steps = num_steps - burn_in_steps
    diagnostics: dict[str, Any] = {
        "num_chains": num_chains,
        "num_steps": num_steps,
        "burn_in_steps": burn_in_steps,
        "thinning": thinning,
        "retained_draws_per_chain": draws,
        "retained_draws_total": int(num_chains * draws),
        "step_size": step_size,
        "langevin_time": num_steps * step_size,
        "post_burn_langevin_time": post_steps * step_size,
        "init_jitter_scale": init_jitter_scale,
        "acceptance_rate": float(
            (accepted_total / num_steps).mean().item()
        ),
        "acceptance_rate_min_chain": float(
            (accepted_total / num_steps).min().item()
        ),
        "acceptance_rate_max_chain": float(
            (accepted_total / num_steps).max().item()
        ),
        "burn_in_acceptance_rate": float(
            (accepted_burn / max(1, burn_in_steps)).mean().item()
        ),
        "post_burn_acceptance_rate": float(
            (accepted_post / max(1, post_steps)).mean().item()
        ),
        "invalid_proposal_fraction": float(
            invalid_total.sum().item() / (num_steps * num_chains)
        ),
        "mean_squared_jump_distance": float(
            (squared_jump_sum / num_steps).mean().item()
        ),
        "split_rhat": split_rhat.tolist(),
        "split_rhat_max": float(split_rhat.max().item()),
        "ess": ess.tolist(),
        "ess_min": float(ess.min().item()),
        "early_late_mean_l2": float(
            (late_mean - early_mean).square().sum().item()
        ),
        "early_late_standardized_drift": standardized_drift.tolist(),
        "early_late_standardized_drift_max": float(
            standardized_drift.max().item()
        ),
        "generating_epsilon": generating_epsilon.detach().cpu()[0].tolist(),
        "initial_chain_epsilon": initial_epsilon.tolist(),
        "snapshot_chain_means": {
            str(step): value.mean(dim=0).tolist()
            for step, value in sorted(snapshots.items())
        },
        "snapshot_chain_sds": {
            str(step): value.std(dim=0, unbiased=True).tolist()
            for step, value in sorted(snapshots.items())
        },
    }
    diagnostics["convergence_checks"] = {
        "split_rhat_max_le_1_01": (
            diagnostics["split_rhat_max"] <= 1.01
        ),
        "ess_min_ge_400": diagnostics["ess_min"] >= 400.0,
        "standardized_drift_max_le_0_10": (
            diagnostics["early_late_standardized_drift_max"] <= 0.10
        ),
        "all_proposals_finite": (
            diagnostics["invalid_proposal_fraction"] == 0.0
        ),
    }
    diagnostics["convergence_pass"] = all(
        diagnostics["convergence_checks"].values()
    )
    return samples, diagnostics, trace


def _pca_projection(
    samples: torch.Tensor,
    generating_epsilon: torch.Tensor,
    initial_epsilon: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    flat = samples.reshape(-1, samples.shape[-1]).numpy()
    center = flat.mean(axis=0)
    _, singular_values, right = np.linalg.svd(
        flat - center,
        full_matrices=False,
    )
    components = right[:2]
    projected = (flat - center) @ components.T
    generating_projected = (
        generating_epsilon.numpy().reshape(1, -1) - center
    ) @ components.T
    initial_projected = (
        initial_epsilon.numpy() - center
    ) @ components.T
    explained = singular_values**2
    explained = explained / explained.sum()
    return projected, generating_projected, initial_projected, explained[:2]


def _write_csv(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_visualization(
    report_dir: Path,
    samples: torch.Tensor,
    diagnostics: dict[str, Any],
    trace: list[dict[str, float]],
    *,
    seed: int,
    epoch: int,
    z: torch.Tensor,
    max_plot_samples: int,
) -> tuple[np.ndarray, np.ndarray, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    initial = torch.tensor(
        diagnostics["initial_chain_epsilon"],
        dtype=torch.float64,
    )
    generating = torch.tensor(
        diagnostics["generating_epsilon"],
        dtype=torch.float64,
    )
    flat = samples.reshape(-1, samples.shape[-1]).numpy()
    dimensions = flat.shape[-1]
    if dimensions == 2:
        projected = flat.copy()
        projected_generating = generating.numpy().reshape(1, -1)
        projected_initial = initial.numpy()
        coordinate_variance = flat.var(axis=0)
        explained = coordinate_variance / coordinate_variance.sum()
        projection_kind = "raw_epsilon"
        projection_x_label = "epsilon[0]"
        projection_y_label = "epsilon[1]"
        projection_title = "Posterior epsilon samples"
    else:
        projected, projected_generating, projected_initial, explained = (
            _pca_projection(samples, generating, initial)
        )
        projection_kind = "pca"
        projection_x_label = (
            f"PC1 ({100 * explained[0]:.1f}% variance)"
        )
        projection_y_label = (
            f"PC2 ({100 * explained[1]:.1f}% variance)"
        )
        projection_title = (
            f"{dimensions}-dimensional posterior projected to PCA"
        )
    draws = samples.shape[1]
    flat_draw = np.tile(np.arange(draws), samples.shape[0])
    early_mask = flat_draw < draws // 2
    if len(flat) > max_plot_samples:
        selected = np.linspace(
            0,
            len(flat) - 1,
            max_plot_samples,
            dtype=np.int64,
        )
    else:
        selected = np.arange(len(flat))

    histogram_rows = math.ceil(dimensions / 2)
    figure_rows = 1 + histogram_rows
    fig, axes = plt.subplots(
        figure_rows,
        2,
        figsize=(13, 4.0 + 3.6 * histogram_rows),
        constrained_layout=True,
        squeeze=False,
    )
    ax = axes[0, 0]
    early_selected = selected[early_mask[selected]]
    late_selected = selected[~early_mask[selected]]
    ax.scatter(
        projected[early_selected, 0],
        projected[early_selected, 1],
        s=7,
        alpha=0.18,
        label="retained first half",
    )
    ax.scatter(
        projected[late_selected, 0],
        projected[late_selected, 1],
        s=7,
        alpha=0.18,
        label="retained second half",
    )
    ax.scatter(
        projected_initial[:, 0],
        projected_initial[:, 1],
        s=24,
        marker="x",
        linewidths=0.7,
        alpha=0.55,
        label="chain starts",
    )
    ax.scatter(
        projected_generating[:, 0],
        projected_generating[:, 1],
        s=90,
        marker="*",
        color="black",
        label="generating epsilon",
        zorder=5,
    )
    ax.set_xlabel(projection_x_label)
    ax.set_ylabel(projection_y_label)
    ax.set_title(projection_title)
    ax.legend(frameon=False, fontsize=8)

    ax = axes[0, 1]
    trace_steps = np.asarray([row["step"] for row in trace])
    trace_accept = np.asarray([row["acceptance_rate"] for row in trace])
    trace_logp = np.asarray([row["mean_log_posterior"] for row in trace])
    ax.plot(trace_steps, trace_accept, label="cumulative acceptance")
    ax.set_xlabel("MALA step")
    ax.set_ylabel("acceptance rate")
    ax.set_ylim(0.0, 1.02)
    twin = ax.twinx()
    twin.plot(
        trace_steps,
        trace_logp,
        color="tab:orange",
        alpha=0.8,
        label="mean log posterior",
    )
    twin.set_ylabel("mean unnormalized log posterior")
    ax.set_title("Acceptance and posterior trace")
    handles_a, labels_a = ax.get_legend_handles_labels()
    handles_b, labels_b = twin.get_legend_handles_labels()
    ax.legend(
        handles_a + handles_b,
        labels_a + labels_b,
        frameon=False,
        fontsize=8,
    )

    histogram_axes = axes.reshape(-1)[2:]
    for dimension in range(dimensions):
        ax = histogram_axes[dimension]
        ax.hist(
            flat[early_mask, dimension],
            bins=55,
            density=True,
            alpha=0.45,
            label="first half",
        )
        ax.hist(
            flat[~early_mask, dimension],
            bins=55,
            density=True,
            alpha=0.45,
            label="second half",
        )
        ax.axvline(
            diagnostics["generating_epsilon"][dimension],
            color="black",
            linestyle="--",
            linewidth=1.0,
            label="generating epsilon" if dimension == 0 else None,
        )
        ax.set_xlabel(f"epsilon[{dimension}]")
        ax.set_ylabel("density")
        ax.set_title(
            f"epsilon[{dimension}] marginal; "
            f"R-hat={diagnostics['split_rhat'][dimension]:.3f}"
        )
        if dimension == 0:
            ax.legend(frameon=False, fontsize=8)
    for ax in histogram_axes[dimensions:]:
        ax.axis("off")

    fig.suptitle(
        "Posterior MALA: DSIVI x_shaped "
        f"seed {seed}, epoch {epoch}, z="
        f"({float(z[0, 0]):.3f}, {float(z[0, 1]):.3f})"
    )
    figure_path = report_dir / "posterior_epsilon_diagnostic.png"
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    return projected, explained, projection_kind


def _write_report(
    report_dir: Path,
    samples: torch.Tensor,
    diagnostics: dict[str, Any],
    trace: list[dict[str, float]],
    *,
    metadata: dict[str, Any],
    max_plot_samples: int,
    max_csv_samples: int,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    projected, explained, projection_kind = _write_visualization(
        report_dir,
        samples,
        diagnostics,
        trace,
        seed=int(metadata["seed"]),
        epoch=int(metadata["epoch"]),
        z=torch.tensor([metadata["z"]], dtype=torch.float64),
        max_plot_samples=max_plot_samples,
    )
    diagnostics["projection_kind"] = projection_kind
    if projection_kind == "pca":
        diagnostics["pca_explained_variance_ratio"] = explained.tolist()
    else:
        diagnostics["epsilon_coordinate_variance_ratio"] = (
            explained.tolist()
        )
    payload = {**metadata, **diagnostics}
    with (report_dir / "posterior_mala_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")

    _write_csv(report_dir / "posterior_mala_trace.csv", trace)
    flat = samples.reshape(-1, samples.shape[-1]).numpy()
    chains, draws, dimensions = samples.shape
    if len(flat) > max_csv_samples:
        selected = np.linspace(
            0,
            len(flat) - 1,
            max_csv_samples,
            dtype=np.int64,
        )
    else:
        selected = np.arange(len(flat))
    rows: list[dict[str, Any]] = []
    for flat_index in selected:
        chain = int(flat_index // draws)
        draw = int(flat_index % draws)
        row: dict[str, Any] = {
            "chain": chain,
            "draw": draw,
            "phase": "early" if draw < draws // 2 else "late",
            "pc1": f"{projected[flat_index, 0]:.8f}",
            "pc2": f"{projected[flat_index, 1]:.8f}",
        }
        for dimension in range(dimensions):
            row[f"epsilon_{dimension}"] = (
                f"{flat[flat_index, dimension]:.8f}"
            )
        rows.append(row)
    _write_csv(report_dir / "posterior_epsilon_samples.csv", rows)

    convergence = (
        "passes the configured diagnostics"
        if diagnostics["convergence_pass"]
        else "does not pass all configured diagnostics"
    )
    lines = [
        "# Posterior-epsilon MALA diagnostic",
        "",
        (
            f"DSIVI `x_shaped`, seed {metadata['seed']}, epoch "
            f"{metadata['epoch']}; epsilon dimension "
            f"{metadata['epsilon_dim']}."
        ),
        "",
        (
            f"At 10,000 steps with step size 0.0001, this run "
            f"**{convergence}**. Acceptance alone is not used as evidence "
            "of convergence."
        ),
        "",
        "| Metric | Value |",
        "|---|---:|",
        (
            "| Overall acceptance rate | "
            f"{diagnostics['acceptance_rate']:.8f} |"
        ),
        (
            "| Post-burn acceptance rate | "
            f"{diagnostics['post_burn_acceptance_rate']:.8f} |"
        ),
        (
            "| Maximum split R-hat | "
            f"{diagnostics['split_rhat_max']:.8f} |"
        ),
        f"| Minimum classical ESS | {diagnostics['ess_min']:.2f} |",
        (
            "| Maximum standardized early/late drift | "
            f"{diagnostics['early_late_standardized_drift_max']:.8f} |"
        ),
        (
            "| Invalid proposal fraction | "
            f"{diagnostics['invalid_proposal_fraction']:.8f} |"
        ),
        "",
        "![Posterior epsilon diagnostic](posterior_epsilon_diagnostic.png)",
        "",
    ]
    (report_dir / "posterior_mala_report.md").write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def run_posterior_mala_diagnostic(cfg: DictConfig) -> dict[str, Any]:
    specs = build_cell_specs(cfg)
    if len(specs) != 1:
        raise RuntimeError(
            f"Posterior MALA diagnostic requires exactly one cell; got "
            f"{len(specs)}."
        )
    spec = specs[0]
    runner: Any | None = None
    try:
        runner = _build_runner(spec.record, cfg)
        _load_checkpoint(runner, spec)
        if str(runner.device) != "cuda":
            raise RuntimeError("Production posterior MALA requires CUDA.")
        epsilon_dim = int(runner.vi_model.epsilon_dim)
        z_index = int(cfg.evaluation.z_index)
        if z_index < 0:
            raise ValueError("z_index must be non-negative.")

        forward_seed = stable_seed(spec.key, "posterior_mala_forward")
        seed_everything(forward_seed, use_cuda=True)
        generating_bank, z_bank = runner.vi_model.sampling(num=z_index + 1)
        generating_epsilon = generating_bank[z_index:z_index + 1]
        z = z_bank[z_index:z_index + 1]

        sampler = cfg.evaluation.sampler
        sampler_seed = stable_seed(spec.key, "posterior_mala_sampler")
        seed_everything(sampler_seed, use_cuda=True)
        torch.cuda.synchronize()
        started = time.perf_counter()
        samples, diagnostics, trace = posterior_mala_samples(
            runner.vi_model,
            z,
            generating_epsilon,
            num_chains=int(sampler.num_chains),
            num_steps=int(sampler.num_steps),
            burn_in_steps=int(sampler.burn_in_steps),
            thinning=int(sampler.thinning),
            step_size=float(sampler.step_size),
            init_jitter_scale=float(sampler.init_jitter_scale),
            trace_interval=int(sampler.trace_interval),
            snapshot_steps=[
                int(value) for value in sampler.snapshot_steps
            ],
        )
        torch.cuda.synchronize()
        runtime_sec = time.perf_counter() - started
        diagnostics["runtime_sec"] = runtime_sec

        runtime_dir = REPO_ROOT / str(cfg.output.runtime_dir)
        report_dir = REPO_ROOT / str(cfg.output.report_dir)
        runtime_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "samples": samples,
                "z": z.detach().cpu(),
                "generating_epsilon": generating_epsilon.detach().cpu(),
                "diagnostics": diagnostics,
                "trace": trace,
            },
            runtime_dir / "posterior_mala_samples.pt",
        )
        metadata = {
            "method": spec.record.method,
            "target": spec.record.target,
            "seed": spec.record.seed,
            "epoch": spec.epoch,
            "checkpoint_dir": spec.checkpoint_dir.as_posix(),
            "epsilon_dim": epsilon_dim,
            "z_dim": int(z.shape[-1]),
            "z": z.detach().cpu()[0].tolist(),
            "forward_seed": forward_seed,
            "sampler_seed": sampler_seed,
            "gpu_name": torch.cuda.get_device_name(),
        }
        _write_report(
            report_dir,
            samples,
            diagnostics,
            trace,
            metadata=metadata,
            max_plot_samples=int(cfg.output.max_plot_samples),
            max_csv_samples=int(cfg.output.max_csv_samples),
        )
        return {**metadata, **diagnostics}
    finally:
        _release_runner(runner)


def regenerate_posterior_mala_report(cfg: DictConfig) -> dict[str, Any]:
    """Regenerate report artifacts from a successfully saved MALA trajectory."""
    specs = build_cell_specs(cfg)
    if len(specs) != 1:
        raise RuntimeError(
            f"Posterior MALA diagnostic requires exactly one cell; got "
            f"{len(specs)}."
        )
    spec = specs[0]
    runtime_dir = REPO_ROOT / str(cfg.output.runtime_dir)
    report_dir = REPO_ROOT / str(cfg.output.report_dir)
    payload = torch.load(
        runtime_dir / "posterior_mala_samples.pt",
        map_location="cpu",
        weights_only=False,
    )
    samples = payload["samples"]
    z = payload["z"]
    diagnostics = payload["diagnostics"]
    trace = payload["trace"]
    if samples.ndim != 3 or not torch.isfinite(samples).all():
        raise FloatingPointError("Saved MALA samples are invalid.")

    metadata = {
        "method": spec.record.method,
        "target": spec.record.target,
        "seed": spec.record.seed,
        "epoch": spec.epoch,
        "checkpoint_dir": spec.checkpoint_dir.as_posix(),
        "epsilon_dim": int(samples.shape[-1]),
        "z_dim": int(z.shape[-1]),
        "z": z[0].tolist(),
        "forward_seed": stable_seed(spec.key, "posterior_mala_forward"),
        "sampler_seed": stable_seed(spec.key, "posterior_mala_sampler"),
        "gpu_name": (
            torch.cuda.get_device_name()
            if torch.cuda.is_available()
            else "not queried during report regeneration"
        ),
    }
    _write_report(
        report_dir,
        samples,
        diagnostics,
        trace,
        metadata=metadata,
        max_plot_samples=int(cfg.output.max_plot_samples),
        max_csv_samples=int(cfg.output.max_csv_samples),
    )
    return {**metadata, **diagnostics}
