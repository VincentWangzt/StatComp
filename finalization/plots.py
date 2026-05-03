from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
import scipy.stats as st
import torch
from omegaconf import OmegaConf

from models.target_models import target_distribution

from .artifacts import (
    RunRecord,
    find_final_samples,
    load_baseline_samples,
    load_grad_norm_series,
    load_kl_ite_series,
    load_sample_z,
    normalize_target,
    run_index,
)
from .config import REPO_ROOT, repo_path


def _target_bbox(target: str) -> list[float] | None:
    path = REPO_ROOT / "configs" / "targets" / f"{target}.yaml"
    if not path.exists():
        return None
    cfg = OmegaConf.load(path)
    bbox = cfg.get("bbox")
    return list(bbox) if bbox is not None else None


def _take_points(samples: torch.Tensor, count: int) -> np.ndarray:
    if samples.shape[0] > count:
        samples = samples[torch.randperm(samples.shape[0])[:count]]
    return samples.detach().cpu().numpy()


def _toy_logp_grid(target: str, bbox: list[float], grid_size: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_model = target_distribution[target](device="cpu")
    xx, yy = np.mgrid[bbox[0]:bbox[1]:complex(grid_size), bbox[2]:bbox[3]:complex(grid_size)]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    with torch.no_grad():
        logp = target_model.logp(torch.as_tensor(positions.T, dtype=torch.float32))
    logp_grid = logp.detach().cpu().numpy().reshape(xx.shape)
    return xx, yy, logp_grid


def _draw_toy_contours(ax: plt.Axes, target: str, bbox: list[float]) -> None:
    xx, yy, logp_grid = _toy_logp_grid(target, bbox)
    with np.errstate(divide="ignore", invalid="ignore"):
        density_surface = -np.log(-logp_grid)
    if not np.isfinite(density_surface).any():
        density_surface = logp_grid
    ax.contourf(xx, yy, density_surface, cmap="Blues", alpha=0.8, levels=11)
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))


def _draw_target_line_contours(
    ax: plt.Axes,
    target: str,
    bbox: list[float],
    *,
    grid_size: int,
    num_levels: int,
    linewidth: float,
) -> None:
    xx, yy, logp_grid = _toy_logp_grid(target, bbox, grid_size)
    finite = logp_grid[np.isfinite(logp_grid)]
    if finite.size == 0:
        return
    levels = np.unique(np.quantile(finite, np.linspace(0.28, 0.94, num_levels)))
    if levels.size < 2:
        return
    ax.contour(
        xx,
        yy,
        logp_grid,
        levels=levels,
        colors="#2f2f2f",
        linewidths=linewidth,
        linestyles="solid",
    )


def _draw_sample_hist2d(
    ax: plt.Axes,
    points: np.ndarray,
    bbox: list[float],
    *,
    bins: int,
    alpha: float,
) -> None:
    hist, x_edges, y_edges = np.histogram2d(
        points[:, 0],
        points[:, 1],
        bins=bins,
        range=[[bbox[0], bbox[1]], [bbox[2], bbox[3]]],
        density=True,
    )
    hist = np.ma.masked_where(hist <= 0, hist)
    vmax = float(hist.max()) if hist.count() else 1.0
    ax.pcolormesh(
        x_edges,
        y_edges,
        hist.T,
        cmap="Blues",
        norm=Normalize(vmin=0.0, vmax=vmax),
        alpha=alpha,
        shading="auto",
        rasterized=True,
    )


def _load_plot_samples(column: str, target: str, seed: int, idx: dict[tuple[int, str, str], RunRecord], count: int, cfg: Any, plot_cfg: Any) -> torch.Tensor:
    if _is_truth_column(column):
        return load_baseline_samples(target)

    rec = idx[(seed, column.upper(), target)]
    sample_path, _ = find_final_samples(rec.result_path)
    samples = load_sample_z(sample_path)
    if samples.shape[0] >= count:
        return samples

    try:
        from .runner_eval import _sample_vi, build_runner, prepare_config, remove_file_handlers

        runner_cfg = prepare_config(
            rec,
            device=str(cfg.evaluation.device),
            scratch_results=str(cfg.campaign.scratch_results_dir),
            scratch_tb=str(cfg.campaign.scratch_tb_dir),
        )
        runner, _ckpt_dir, _epoch = build_runner(rec, runner_cfg)
        try:
            return _sample_vi(runner, count, int(plot_cfg.get("sample_batch_size", 10000)))
        finally:
            if hasattr(runner, "writer"):
                runner.writer.close()
            remove_file_handlers()
            del runner
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    except Exception:
        return samples


def render_scatter_grid(records: list[RunRecord], cfg: Any) -> Path:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [normalize_target(str(target)) for target in cfg.selection.scatter_targets]
    configured_columns = [str(method) for method in cfg.selection.scatter_methods]
    columns = configured_columns if any(_is_truth_column(column) for column in configured_columns) else configured_columns + ["GroundTruth"]
    seed = int(cfg.selection.seed_for_figures)
    idx = run_index(records)
    num_points = int(cfg.plots.scatter.num_points)
    panel_w, panel_h = [float(x) for x in cfg.plots.scatter.figsize_per_panel]
    title_fontsize = int(cfg.plots.scatter.get("title_fontsize", 12))
    label_fontsize = int(cfg.plots.scatter.get("label_fontsize", 12))
    w_pad = float(cfg.plots.scatter.get("w_pad", 0.8))
    h_pad = float(cfg.plots.scatter.get("h_pad", 0.35))

    fig, axes = plt.subplots(
        len(targets),
        len(columns),
        figsize=(panel_w * len(columns), panel_h * len(targets)),
        squeeze=False,
    )
    for row_idx, target in enumerate(targets):
        bbox = _target_bbox(target)
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(_scatter_column_label(column), fontsize=title_fontsize)
            if col_idx == 0:
                ax.set_ylabel(target, fontsize=label_fontsize)
            try:
                if bbox is not None:
                    _draw_toy_contours(ax, target, bbox)
                if _is_truth_column(column):
                    samples = load_baseline_samples(target)
                else:
                    rec = idx[(seed, column.upper(), target)]
                    sample_path, _ = find_final_samples(rec.result_path)
                    samples = load_sample_z(sample_path)
                points = _take_points(samples[:, :2], num_points)
                ax.plot(
                    points[:, 0],
                    points[:, 1],
                    ".",
                    markersize=float(cfg.plots.scatter.point_size),
                    color="#ff7f0e",
                    alpha=float(cfg.plots.scatter.alpha),
                )
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"missing\n{type(exc).__name__}", ha="center", va="center", fontsize=7)
            if bbox is not None:
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
            ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
            if col_idx != 0:
                ax.tick_params(labelleft=False)
    fig.tight_layout(pad=0.35, w_pad=w_pad, h_pad=h_pad)
    png_path = out_dir / "toy_scatter_grid.png"
    pdf_path = out_dir / "toy_scatter_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_scatter_hist_grid(records: list[RunRecord], cfg: Any) -> Path:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [normalize_target(str(target)) for target in cfg.selection.scatter_targets]
    configured_columns = [str(method) for method in cfg.selection.scatter_methods]
    columns = configured_columns if any(_is_truth_column(column) for column in configured_columns) else configured_columns + ["GroundTruth"]
    seed = int(cfg.selection.seed_for_figures)
    idx = run_index(records)
    plot_cfg = cfg.plots.get("scatter_hist", cfg.plots.scatter)
    num_points = int(plot_cfg.get("num_points", cfg.plots.scatter.num_points))
    bins = int(plot_cfg.get("bins", 70))
    contour_grid_size = int(plot_cfg.get("contour_grid_size", 160))
    contour_levels = int(plot_cfg.get("contour_levels", 5))
    contour_linewidth = float(plot_cfg.get("contour_linewidth", 0.75))
    hist_alpha = float(plot_cfg.get("hist_alpha", 0.78))
    panel_w, panel_h = [float(x) for x in plot_cfg.get("figsize_per_panel", cfg.plots.scatter.figsize_per_panel)]
    title_fontsize = int(plot_cfg.get("title_fontsize", cfg.plots.scatter.get("title_fontsize", 12)))
    label_fontsize = int(plot_cfg.get("label_fontsize", cfg.plots.scatter.get("label_fontsize", 12)))
    w_pad = float(plot_cfg.get("w_pad", cfg.plots.scatter.get("w_pad", 0.8)))
    h_pad = float(plot_cfg.get("h_pad", cfg.plots.scatter.get("h_pad", 0.35)))

    fig, axes = plt.subplots(
        len(targets),
        len(columns),
        figsize=(panel_w * len(columns), panel_h * len(targets)),
        squeeze=False,
    )
    for row_idx, target in enumerate(targets):
        bbox = _target_bbox(target)
        for col_idx, column in enumerate(columns):
            ax = axes[row_idx][col_idx]
            if row_idx == 0:
                ax.set_title(_scatter_column_label(column), fontsize=title_fontsize)
            if col_idx == 0:
                ax.set_ylabel(target, fontsize=label_fontsize)
            try:
                if bbox is None:
                    raise ValueError(f"No bbox configured for {target}")
                samples = _load_plot_samples(column, target, seed, idx, num_points, cfg, plot_cfg)
                points = _take_points(samples[:, :2], num_points)
                _draw_sample_hist2d(ax, points, bbox, bins=bins, alpha=hist_alpha)
                _draw_target_line_contours(
                    ax,
                    target,
                    bbox,
                    grid_size=contour_grid_size,
                    num_levels=contour_levels,
                    linewidth=contour_linewidth,
                )
            except Exception as exc:  # noqa: BLE001
                ax.text(0.5, 0.5, f"missing\n{type(exc).__name__}", ha="center", va="center", fontsize=7)
            if bbox is not None:
                ax.set_xlim(bbox[0], bbox[1])
                ax.set_ylim(bbox[2], bbox[3])
                ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
            ax.tick_params(axis="both", labelsize=7, length=2, width=0.5)
            if col_idx != 0:
                ax.tick_params(labelleft=False)
    fig.tight_layout(pad=0.35, w_pad=w_pad, h_pad=h_pad)
    png_path = out_dir / "toy_scatter_hist_grid.png"
    pdf_path = out_dir / "toy_scatter_hist_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def _is_truth_column(column: str) -> bool:
    normalized = column.replace("_", "").replace(" ", "").lower()
    return normalized in {"truth", "groundtruth", "groundtruthdistribution"}


def _scatter_column_label(column: str) -> str:
    return "GroundTruth" if _is_truth_column(column) else column.upper()


def _langevin_target(device: str = "cpu"):
    return target_distribution["Langevin_post"](device=device)


def _trace_stats(samples: torch.Tensor, target_model: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    u_np = samples.detach().cpu().numpy()
    u_mean = u_np.mean(0)
    low, high = st.t.interval(0.95, len(u_mean), loc=u_mean, scale=np.std(u_np, 0))
    true_path = target_model.u.detach().cpu().numpy().flatten()
    obs = target_model.data.detach().cpu().numpy().reshape(-1)
    t = np.arange(target_model.dt, target_model.T + target_model.dt, target_model.dt)
    obs_t = np.arange(target_model.T / target_model.num_obs, target_model.T + target_model.T / target_model.num_obs, target_model.T / target_model.num_obs)
    return t, true_path, u_mean, low, high, np.vstack([obs_t[: obs.shape[0]], obs]).T


def langevin_panel_labels(methods: list[str], available_methods: set[str] | None = None) -> list[str]:
    available = None if available_methods is None else {method.upper() for method in available_methods}
    method_labels = [method.upper() for method in methods if available is None or method.upper() in available]
    labels = ["SGLD"] + [method for method in method_labels if method != "DSIVI"]
    if "DSIVI" not in method_labels:
        return labels
    cols = math.ceil((len(labels) + 1) / 2)
    labels.insert(cols, "DSIVI")
    return labels


def render_langevin_trace_grid(records: list[RunRecord], cfg: Any) -> Path:
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(cfg.selection.seed_for_figures)
    methods = [str(method).upper() for method in cfg.selection.methods]
    idx = run_index(records)
    target_model = _langevin_target()
    num_samples = int(cfg.plots.langevin_trace.num_samples)

    available = {rec.method.upper() for rec in records if rec.seed == seed and rec.target == "Langevin_post"}
    panel_labels = langevin_panel_labels(methods, available)

    panels: list[tuple[str, torch.Tensor]] = []
    for method in panel_labels:
        if method == "SGLD":
            panels.append(("SGLD", load_baseline_samples("Langevin_post")[:num_samples]))
            continue
        rec = idx.get((seed, method, "Langevin_post"))
        if rec is None:
            continue
        sample_path, _ = find_final_samples(rec.result_path)
        panels.append((method, load_sample_z(sample_path)[:num_samples]))

    if not panels:
        raise RuntimeError("No Langevin_post panels available.")

    stats = [(label, _trace_stats(samples, target_model)) for label, samples in panels]
    y_values = []
    for _, (_, true_path, mean, low, high, obs_points) in stats:
        y_values.extend([true_path, mean, low, high, obs_points[:, 1]])
    y_min = min(float(np.nanmin(values)) for values in y_values)
    y_max = max(float(np.nanmax(values)) for values in y_values)
    margin = 0.05 * max(y_max - y_min, 1.0)

    cols = math.ceil(len(panels) / 2)
    panel_w, panel_h = [float(x) for x in cfg.plots.langevin_trace.figsize_per_panel]
    fig, axes = plt.subplots(2, cols, figsize=(panel_w * cols, panel_h * 2), squeeze=False, sharey=True)
    for ax in axes.ravel()[len(panels):]:
        ax.axis("off")
    for i, (ax, (label, (t, true_path, mean, low, high, obs_points))) in enumerate(zip(axes.ravel(), stats)):
        first = i == 0
        ax.plot(t, true_path, color="magenta", linewidth=1.0, label="true path" if first else None)
        ax.plot(t, mean, color="blue", linewidth=1.0, label="sample path" if first else None)
        ax.plot(t, low, color="black", linewidth=0.6)
        ax.plot(t, high, color="black", linewidth=0.6)
        ax.fill_between(t, low, high, facecolor="aqua", alpha=0.3, label="confidence interval" if first else None)
        ax.scatter(obs_points[:, 0], obs_points[:, 1], color="red", marker=".", linewidth=0.5, s=10, label="observation" if first else None)
        ax.set_title(label, fontsize=10)
        ax.grid(True, linewidth=0.3)
        ax.set_ylim(y_min - margin, y_max + margin)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = out_dir / "langevin_trace_grid.png"
    pdf_path = out_dir / "langevin_trace_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


# ---------------------------------------------------------------------------
# KL convergence curve plots
# ---------------------------------------------------------------------------

_KL_METHOD_COLORS: dict[str, str] = {
    "UIVI": "#1f77b4",   # blue
    "AISIVI": "#ff7f0e", # orange
    "DSIVI": "#2ca02c",  # green
    "KSIVI": "#d62728",  # red
    "SIVI": "#9467bd",   # purple
}


def _method_color(method: str, idx: int = 0) -> str:
    """Return a stable color for *method*, falling back to the tab10 cycle."""
    key = method.upper()
    if key in _KL_METHOD_COLORS:
        return _KL_METHOD_COLORS[key]
    tab10 = plt.cm.tab10.colors  # type: ignore[attr-defined]
    return tab10[idx % len(tab10)]


def _aggregate_curves(
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves to a common grid, return (grid, mean, se).

    The common x-range is the intersection of all seeds' [min, max] ranges
    so that ``np.interp`` never extrapolates.
    """
    if not seed_curves:
        return None
    global_min = max(float(xs.min()) for xs, _ in seed_curves)
    global_max = min(float(xs.max()) for xs, _ in seed_curves)
    if global_min >= global_max:
        return None
    grid = np.linspace(global_min, global_max, n_grid)
    interpolated = np.empty((len(seed_curves), n_grid), dtype=np.float64)
    for i, (xs, ys) in enumerate(seed_curves):
        interpolated[i] = np.interp(grid, xs, ys)
    mean = interpolated.mean(axis=0)
    se = (
        interpolated.std(axis=0, ddof=1) / np.sqrt(len(seed_curves))
        if len(seed_curves) > 1
        else np.zeros_like(mean)
    )
    return grid, mean, se


def _aggregate_curves_minmax(
    seed_curves: list[tuple[np.ndarray, np.ndarray]],
    n_grid: int = 200,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Interpolate seed curves to a common grid, return (grid, mean, min, max).

    The common x-range is the intersection of all seeds' [min, max] ranges
    so that ``np.interp`` never extrapolates.
    """
    if not seed_curves:
        return None
    global_min = max(float(xs.min()) for xs, _ in seed_curves)
    global_max = min(float(xs.max()) for xs, _ in seed_curves)
    if global_min >= global_max:
        return None
    grid = np.linspace(global_min, global_max, n_grid)
    interpolated = np.empty((len(seed_curves), n_grid), dtype=np.float64)
    for i, (xs, ys) in enumerate(seed_curves):
        interpolated[i] = np.interp(grid, xs, ys)
    mean = interpolated.mean(axis=0)
    min_vals = interpolated.min(axis=0)
    max_vals = interpolated.max(axis=0)
    return grid, mean, min_vals, max_vals


def _collect_kl_curves(
    records: list[RunRecord],
    targets: list[str],
    methods: list[str],
    x_mode: str,
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load KL_ITE series for all (method, target) pairs grouped by seed.

    Args:
        x_mode: ``"step"`` uses iteration number; ``"time"`` uses seconds elapsed.

    Returns:
        Mapping of ``(method_upper, target)`` to list of ``(x, y)`` seed curves.
    """
    method_set = {m.upper() for m in methods}
    target_set = set(targets)

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for rec in records:
        mu = rec.method.upper()
        if mu not in method_set or rec.target not in target_set:
            continue
        loaded = load_kl_ite_series(rec)
        if loaded is None:
            continue
        steps, wall_times, values = loaded
        if x_mode == "step":
            x = steps
        else:
            x = wall_times - wall_times[0]
        key = (mu, rec.target)
        curves.setdefault(key, []).append((x, values))
    return curves


def _collect_grad_norm_curves(
    records: list[RunRecord],
    targets: list[str],
    methods: list[str],
) -> dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]]:
    """Load gradient-norm series for all (method, target) pairs grouped by seed.

    Returns:
        Mapping of ``(method_upper, target)`` to list of ``(steps, values)`` seed curves.
    """
    method_set = {m.upper() for m in methods}
    target_set = set(targets)

    curves: dict[tuple[str, str], list[tuple[np.ndarray, np.ndarray]]] = {}
    for rec in records:
        mu = rec.method.upper()
        if mu not in method_set or rec.target not in target_set:
            continue
        loaded = load_grad_norm_series(rec)
        if loaded is None:
            continue
        steps, _wall_times, values = loaded
        key = (mu, rec.target)
        curves.setdefault(key, []).append((steps, values))
    return curves


def render_kl_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render KL(p, q_theta) vs iteration — one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.kl_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.kl_curve_targets]
    methods = [str(m) for m in cfg.selection.kl_curve_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    all_curves = _collect_kl_curves(records, targets, methods, x_mode="step")

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(target, fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$\mathrm{KL}(p,\, q_\theta)$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, se = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label=mu)
            ax.fill_between(grid, mean - se, mean + se, color=color, alpha=band_alpha)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "kl_iteration_grid.png"
    pdf_path = out_dir / "kl_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_kl_time_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render KL(p, q_theta) vs wall-clock time — one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.kl_time
    targets = [normalize_target(str(t)) for t in cfg.selection.kl_curve_targets]
    methods = [str(m) for m in cfg.selection.kl_curve_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))

    all_curves = _collect_kl_curves(records, targets, methods, x_mode="time")

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(target, fontsize=title_fontsize)
        ax.set_xlabel("Time (s)", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel(r"$\mathrm{KL}(p,\, q_\theta)$", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, se = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label=mu)
            ax.fill_between(grid, mean - se, mean + se, color=color, alpha=band_alpha)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "kl_time_grid.png"
    pdf_path = out_dir / "kl_time_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path


def render_grad_norm_iteration_grid(records: list[RunRecord], cfg: Any) -> Path:
    """Render gradient norm vs iteration -- one subplot per target, methods overlaid."""
    root = repo_path(str(cfg.campaign.output_dir))
    assert root is not None
    out_dir = root / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_cfg = cfg.plots.grad_norm_iteration
    targets = [normalize_target(str(t)) for t in cfg.selection.grad_norm_targets]
    methods = [str(m) for m in cfg.selection.grad_norm_methods]
    n_grid = int(plot_cfg.get("n_grid", 200))
    panel_w, panel_h = [float(x) for x in plot_cfg.figsize_per_panel]
    band_alpha = float(plot_cfg.get("band_alpha", 0.25))
    linewidth = float(plot_cfg.get("linewidth", 1.5))
    title_fontsize = int(plot_cfg.get("title_fontsize", 13))
    label_fontsize = int(plot_cfg.get("label_fontsize", 12))
    log_scale = bool(plot_cfg.get("log_scale", True))

    all_curves = _collect_grad_norm_curves(records, targets, methods)

    n_cols = len(targets)
    fig, axes = plt.subplots(1, n_cols, figsize=(panel_w * n_cols, panel_h), squeeze=False)

    for col_idx, target in enumerate(targets):
        ax = axes[0][col_idx]
        ax.set_title(target, fontsize=title_fontsize)
        ax.set_xlabel("Iteration", fontsize=label_fontsize)
        if col_idx == 0:
            ax.set_ylabel("Gradient Norm", fontsize=label_fontsize)
        for m_idx, method in enumerate(methods):
            mu = method.upper()
            seed_curves = all_curves.get((mu, target), [])
            agg = _aggregate_curves_minmax(seed_curves, n_grid=n_grid)
            if agg is None:
                continue
            grid, mean, min_vals, max_vals = agg
            color = _method_color(mu, m_idx)
            ax.plot(grid, mean, color=color, linewidth=linewidth, label="Mean")
            ax.fill_between(grid, min_vals, max_vals, color=color, alpha=band_alpha, label="Min/Max")
        if log_scale:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(plt.ScalarFormatter())
            ax.yaxis.get_major_formatter().set_scientific(False)
            ax.yaxis.set_minor_formatter(plt.ScalarFormatter())
            ax.yaxis.get_minor_formatter().set_scientific(False)
        ax.grid(True, linewidth=0.3)
        ax.tick_params(axis="both", labelsize=9, length=2, width=0.5)
        if col_idx == 0:
            ax.legend(fontsize=9)

    fig.tight_layout(pad=0.4, w_pad=0.6)
    png_path = out_dir / "grad_norm_iteration_grid.png"
    pdf_path = out_dir / "grad_norm_iteration_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path
