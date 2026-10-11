"""Plot four selected 50K configurations across all five held-out seeds.

Run on the experiment server against the audited, saved sample arrays. The
panel scores are the original training diagnostics used in the summary table.
"""
from pathlib import Path
import argparse
import hashlib
import json
import math
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.colors import LogNorm
import numpy as np


HERE = Path(__file__).resolve().parent
CONFIGURATIONS = [
    ("shared_r1", "Shared conditional", "Equal learning rates", r"$\eta_v=10^{-3}$", "#b64038"),
    ("shared_r0p01", "Shared conditional", "Slow variance head", r"$\eta_v=10^{-5}$", "#2b8c75"),
    ("linear_r1", "Independent linear", "Variance map", r"$\eta_v=10^{-3}$", "#6056a5"),
    ("global", "Global variance", "One variance vector", r"$\eta_v=10^{-3}$", "#2f6c9d"),
]
SEEDS = [45, 46, 47, 48, 49]
NAME = "selected_4x5_samples"


def make(root, report, output):
    records = json.loads((report / "training_results.json").read_text())["rows"]
    lookup = {}
    for record in records:
        spec = record["spec"]
        if spec["phase"] == "confirmation":
            key = (spec["name"], spec["seed"])
            if key in lookup:
                raise ValueError(f"Duplicate held-out result: {key}")
            lookup[key] = record

    panels = []
    for name, *_ in CONFIGURATIONS:
        for seed in SEEDS:
            record = lookup[(name, seed)]
            if record["spec"]["steps"] != 50000:
                raise ValueError("The grid requires exactly 50K optimizer updates")
            path = root / "confirmation" / f"{name}_s{seed}" / "samples_50000.npy"
            samples = np.load(path, allow_pickle=False)
            if samples.shape != (10000, 2) or not np.isfinite(samples).all():
                raise ValueError(f"Expected 10K finite two-dimensional samples: {path}")
            panels.append({
                "name": name, "seed": seed, "samples": samples,
                "sw2": float(record["final"]["sw2"]),
                "sample_path": str(path),
                "sample_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "outside_plot_fraction": float((np.abs(samples) > 5).any(axis=1).mean()),
            })

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "mathtext.fontset": "dejavusans", "pdf.fonttype": 42,
        "ps.fonttype": 42, "axes.linewidth": .65,
    })
    fig, axes = plt.subplots(4, 5, figsize=(14.4, 10.6), sharex=True, sharey=True)
    fig.subplots_adjust(left=.245, right=.933, bottom=.135, top=.88, wspace=.12, hspace=.15)
    fig.suptitle("Conditional variance: samples behind the comparison", y=.98,
                 fontsize=18, weight="semibold")
    fig.text(.5, .943, "50,000 optimizer updates | all five held-out seeds | original KSIVI loss",
             ha="center", fontsize=11, color="#475569")

    grid = np.linspace(-5, 5, 250)
    xx, yy = np.meshgrid(grid, grid)
    # Equal mixture with covariances [[2, +/-1.8], [+/-1.8, 2]].
    density = (np.exp(-(2*xx*xx-3.6*xx*yy+2*yy*yy)/(2*.76)) +
               np.exp(-(2*xx*xx+3.6*xx*yy+2*yy*yy)/(2*.76))) / (4*math.pi*math.sqrt(.76))
    meshes = []
    summaries = []
    for row, (name, title, detail, rate, color) in enumerate(CONFIGURATIONS):
        group = [p for p in panels if p["name"] == name]
        scores = np.array([p["sw2"] for p in group])
        mean, sd = float(scores.mean()), float(scores.std(ddof=1))
        summaries.append({"name": name, "mean_sw2": mean, "sd_sw2": sd})
        for col, panel in enumerate(group):
            ax = axes[row, col]
            samples = panel["samples"]
            mesh = ax.hexbin(samples[:, 0], samples[:, 1], gridsize=40,
                             extent=(-5, 5, -5, 5), mincnt=1, cmap="viridis",
                             linewidths=0, rasterized=True)
            meshes.append(mesh)
            contour = ax.contour(xx, yy, density, levels=[.005, .02, .05, .1],
                                 colors="#20252b", linewidths=.7, linestyles="dashed")
            contour.set_path_effects([pe.Stroke(linewidth=1.6, foreground="white", alpha=.8),
                                      pe.Normal()])
            ax.set(xlim=(-5, 5), ylim=(-5, 5), xticks=[-5, 0, 5], yticks=[-5, 0, 5],
                   facecolor="#f1f5f9")
            ax.set_aspect("equal")
            ax.tick_params(labelsize=8, length=2.5, color="#64748b")
            for spine in ax.spines.values():
                spine.set_color("#94a3b8")
            ax.text(.97, .96, f"SW2 = {panel['sw2']:.3f}", transform=ax.transAxes,
                    ha="right", va="top", fontsize=8.5,
                    bbox={"boxstyle": "round,pad=.25", "facecolor": "white",
                          "edgecolor": "none", "alpha": .94})
            if panel["outside_plot_fraction"] > .01:
                ax.text(.03, .04, f"Outside: {panel['outside_plot_fraction']:.1%}",
                        transform=ax.transAxes, fontsize=7.5, color="#991b1b",
                        bbox={"facecolor": "white", "edgecolor": "none", "alpha": .94})
            if row == 0:
                ax.set_title(f"Seed {panel['seed']}", fontsize=12, pad=9)
            if row == 3:
                ax.set_xlabel(r"$x_1$", fontsize=10, labelpad=2)
            if col == 0:
                ax.set_ylabel(r"$x_2$", fontsize=10, labelpad=2)

        position = axes[row, 0].get_position()
        y = (position.y0 + position.y1) / 2
        fig.text(.022, y+.047, title, color=color, fontsize=13, weight="semibold", va="center")
        fig.text(.022, y+.018, detail, color="#334155", fontsize=11, va="center")
        fig.text(.022, y-.011, rate, color="#334155", fontsize=11, va="center")
        fig.text(.022, y-.047, f"Mean SW2: {mean:.3f} +/- {sd:.3f}",
                 color="#0f172a", fontsize=10.5, va="center")

    max_count = max(float(mesh.get_array().max()) for mesh in meshes)
    norm = LogNorm(vmin=1, vmax=max_count)
    for mesh in meshes:
        mesh.set_norm(norm)
    color_axis = fig.add_axes([.951, .173, .011, .666])
    colorbar = fig.colorbar(meshes[0], cax=color_axis)
    colorbar.set_label("Samples per hexagon (shared log scale)", fontsize=10, labelpad=9)
    colorbar.ax.tick_params(labelsize=8)
    fig.text(.5, .071, "10,000 saved samples per panel. Dashed contours show the exact X-shaped target.",
             ha="center", fontsize=10, color="#334155")
    fig.text(.5, .044, "Mean/backbone LR = 0.001; common schedule and axes. SW2 uses all samples; lower is better.",
             ha="center", fontsize=9.5, color="#475569")
    fig.text(.5, .019, "Row summaries: mean +/- sample SD across seeds. Off-panel sample fractions are noted if above 1%.",
             ha="center", fontsize=9, color="#64748b")

    output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(output / f"{NAME}.{extension}", dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    metadata = {
        "source_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True).strip(),
        "phase": "confirmation", "steps": 50000, "layout": {"rows": 4, "columns": 5},
        "seeds": SEEDS, "sample_count_per_panel": 10000, "plot_limits": [-5, 5],
        "metric": "Original 10K-sample, 128-direction SW2 diagnostic used in the summary table",
        "shared_count_scale": {"vmin": 1, "vmax": max_count, "normalization": "log"},
        "summaries": summaries,
        "panels": [{k: v for k, v in p.items() if k != "samples"} for p in panels],
    }
    (output / f"{NAME}.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"output": str(output / NAME), "panels": len(panels), "summaries": summaries}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/root/ruivi/results/ksivi_variance_lr_20261011"))
    parser.add_argument("--report", type=Path, default=HERE / "report")
    parser.add_argument("--output", type=Path, default=HERE / "report/figures")
    args = parser.parse_args()
    make(args.root, args.report, args.output)
