from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import torch
from omegaconf import OmegaConf

from models.target_models import target_distribution

from .artifacts import (
    RunRecord,
    find_final_samples,
    load_baseline_samples,
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


def _draw_toy_contours(ax: plt.Axes, target: str, bbox: list[float]) -> None:
    target_model = target_distribution[target](device="cpu")
    xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    with torch.no_grad():
        logp = target_model.logp(torch.as_tensor(positions.T, dtype=torch.float32))
    logp_grid = logp.detach().cpu().numpy().reshape(xx.shape)
    with np.errstate(divide="ignore", invalid="ignore"):
        density_surface = -np.log(-logp_grid)
    if not np.isfinite(density_surface).any():
        density_surface = logp_grid
    ax.contourf(xx, yy, density_surface, cmap="Blues", alpha=0.8, levels=11)
    ax.axis(bbox)
    ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))


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
    for ax, (label, (t, true_path, mean, low, high, obs_points)) in zip(axes.ravel(), stats):
        ax.plot(t, true_path, color="magenta", linewidth=1.0, label="true path")
        ax.plot(t, mean, color="blue", linewidth=1.0, label="sample path")
        ax.plot(t, low, color="black", linewidth=0.6)
        ax.plot(t, high, color="black", linewidth=0.6)
        ax.fill_between(t, low, high, facecolor="aqua", alpha=0.3)
        ax.scatter(obs_points[:, 0], obs_points[:, 1], color="red", marker=".", linewidth=0.5, s=10)
        ax.set_title(label, fontsize=10)
        ax.grid(True, linewidth=0.3)
        ax.set_ylim(y_min - margin, y_max + margin)
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    png_path = out_dir / "langevin_trace_grid.png"
    pdf_path = out_dir / "langevin_trace_grid.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)
    return png_path
