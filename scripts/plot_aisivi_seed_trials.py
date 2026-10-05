"""Compare the saved AISIVI eight-Gaussian seed trials on fixed target axes."""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from finalization.artifacts import completed_runs, find_final_samples, load_manifest, load_sample_z
from finalization.plots import _draw_toy_contours, _scatter_generator, _take_points, _target_bbox


def main() -> None:
    target = "8_gaussians"
    seeds = (0, 1, 2, 44)
    bbox = _target_bbox(target)
    assert bbox is not None
    records = {}
    for campaign in ("toy_scatter_grid", "toy_scatter_seed_aisivi"):
        for record in completed_runs(load_manifest(f"campaigns/{campaign}/manifest.json")):
            if (record.method, record.target) == ("AISIVI", target):
                records[record.seed] = record

    points_by_seed = {}
    for seed in seeds:
        path, epoch = find_final_samples(records[seed].result_path)
        samples = load_sample_z(path)
        assert epoch == 10000 and len(samples) == 10000 and torch.isfinite(samples).all()
        points = _take_points(samples[:, :2], 2000, generator=_scatter_generator(42, target, "AISIVI"))
        points_by_seed[seed] = points
        inside = ((samples[:, 0] >= bbox[0]) & (samples[:, 0] <= bbox[1])
                  & (samples[:, 1] >= bbox[2]) & (samples[:, 1] <= bbox[3]))
        print(f"Seed {seed}: {float(inside.float().mean()):.2%} of saved samples in target bounds")

    out_dir = REPO_ROOT / "campaigns/toy_scatter_grid/generated_reports/finalization/figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(seeds), figsize=(11.2, 3.4), squeeze=False)
    for ax, seed in zip(axes[0], seeds):
        _draw_toy_contours(ax, target, bbox)
        points = points_by_seed[seed]
        ax.plot(points[:, 0], points[:, 1], ".", markersize=3, color="#ff7f0e", alpha=0.45)
        ax.set_title(f"Seed {seed}" if seed != 44 else "Seed 44 (current rerun)", fontsize=12)
        ax.set_xlabel("x")
        ax.set_xticks([-5, 0, 5])
        ax.set_yticks([-5, 0, 5])
        ax.tick_params(labelsize=10)
    axes[0, 0].set_ylabel("y")
    fig.suptitle("AISIVI on eight Gaussians: 10,000 training iterations, 2,000 plotted samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.92), w_pad=1.1)
    comparison = out_dir / "aisivi_8_gaussians_seed_comparison.png"
    fig.savefig(comparison, dpi=240, facecolor="white")
    plt.close(fig)

    points = points_by_seed[44]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    _draw_toy_contours(ax, target, bbox)
    ax.plot(points[:, 0], points[:, 1], ".", markersize=3, color="#ff7f0e", alpha=0.45)
    ax.plot(0, 0, "+", color="#164d82", markersize=9, markeredgewidth=1.5, label="Target centre")
    lower = np.minimum(points.min(axis=0), [bbox[0], bbox[2]])
    upper = np.maximum(points.max(axis=0), [bbox[1], bbox[3]])
    padding = 0.06 * (upper - lower)
    ax.set_xlim(lower[0] - padding[0], upper[0] + padding[0])
    ax.set_ylim(lower[1] - padding[1], upper[1] + padding[1])
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("AISIVI seed 44: full sample extent (2,000 points)", fontsize=13)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    full_extent = out_dir / "aisivi_8_gaussians_seed44_full_extent.png"
    fig.savefig(full_extent, dpi=240, facecolor="white")
    plt.close(fig)
    print(f"Wrote {comparison.relative_to(REPO_ROOT)}")
    print(f"Wrote {full_extent.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
