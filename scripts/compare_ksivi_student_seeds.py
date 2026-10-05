"""Compare additional KSIVI Student-t seeds with the current paper seed and target."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
import torch

from finalization.artifacts import (
    completed_runs,
    find_final_checkpoint,
    find_final_samples,
    load_baseline_samples,
    load_manifest,
    load_sample_z,
)
from finalization.config import repo_path
from finalization.plots import _draw_toy_contours, _scatter_generator, _take_points, _target_bbox
from utils.metrics import compute_sliced_wasserstein


def sample_summary(samples: torch.Tensor, bbox: list[float]) -> dict:
    inside = ((samples[:, 0] >= bbox[0]) & (samples[:, 0] <= bbox[1])
              & (samples[:, 1] >= bbox[2]) & (samples[:, 1] <= bbox[3]))
    medians = torch.quantile(samples, 0.5, dim=0)
    return {
        "sample_count": len(samples),
        "fraction_in_plot_bounds": float(inside.float().mean()),
        "mean_x": float(samples[:, 0].mean()), "mean_y": float(samples[:, 1].mean()),
        "median_x": float(medians[0]), "median_y": float(medians[1]),
        "std_x": float(samples[:, 0].std()), "std_y": float(samples[:, 1].std()),
        "radius_q90": float(torch.quantile(samples.norm(dim=1), 0.9)),
    }


def draw_comparison(panels: list[tuple[str, np.ndarray]], target: str,
                    bbox: list[float], path: Path, *, full_range: bool,
                    title: str = "KSIVI on Student-t: Riesz kernel, 50,000 training iterations, 2,000 plotted samples") -> None:
    fig, axes = plt.subplots(1, len(panels), figsize=(3.0 * len(panels), 3.6), squeeze=False)
    for ax, (label, points) in zip(axes[0], panels):
        _draw_toy_contours(ax, target, bbox)
        ax.plot(points[:, 0], points[:, 1], ".", markersize=3, color="#ff7f0e", alpha=0.45)
        if full_range:
            bounds = [min(bbox[0], float(points[:, 0].min())), max(bbox[1], float(points[:, 0].max())),
                      min(bbox[2], float(points[:, 1].min())), max(bbox[3], float(points[:, 1].max()))]
            x_pad, y_pad = 0.03 * (bounds[1] - bounds[0]), 0.03 * (bounds[3] - bounds[2])
            ax.set_xlim(bounds[0] - x_pad, bounds[1] + x_pad)
            ax.set_ylim(bounds[2] - y_pad, bounds[3] + y_pad)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])
        ax.set_title(label, fontsize=13)
        ax.set_xlabel("x")
        ax.tick_params(labelsize=10)
    axes[0, 0].set_ylabel("y")
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.1)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", default=[
        "campaigns/toy_scatter_seed_ksivi/manifest.json",
        "campaigns/toy_scatter_ksivi_additional/manifest.json",
    ])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 43, 45, 46])
    parser.add_argument("--output-dir", default="campaigns/toy_scatter_ksivi_additional/generated_reports/finalization")
    args = parser.parse_args()
    target = "student_uc"
    bbox = _target_bbox(target)
    assert bbox is not None
    records = {}
    for manifest in args.manifests:
        for record in completed_runs(load_manifest(manifest)):
            if (record.method, record.target) == ("KSIVI", target) and record.seed in args.seeds:
                if record.seed in records:
                    raise ValueError(f"Multiple completed runs for seed {record.seed}")
                records[record.seed] = record
    missing = set(args.seeds) - records.keys()
    if missing:
        raise ValueError(f"Missing completed KSIVI runs: {sorted(missing)}")

    baseline = load_baseline_samples(target)[:, :2]
    assert torch.isfinite(baseline).all()
    reference_points = _take_points(baseline, 2000, generator=_scatter_generator(42, target, "Ground truth"))
    reference_metric = torch.from_numpy(_take_points(baseline, 10000, generator=_scatter_generator(42, target, "W1 reference")))
    panels = [("Ground truth", reference_points)]
    rows = [{"seed": "ground_truth", "final_epoch": "", "kernel": "", **sample_summary(baseline, bbox),
             "sliced_w1_256_projections": "", "vi_checkpoint_finite": "", "result_path": "baselines/exact/student_uc_exact_100k.pt"}]

    for seed in args.seeds:
        record = records[seed]
        config = OmegaConf.load(record.result_path / "full_config.yaml")
        assert config.train.ksivi.kernel == "riesz", seed
        assert config.train.epochs == 50000 and config.train.batch_size == 128, seed
        assert config.train.vi.lr == 0.001 and not config.train.annealing.enabled, seed
        path, epoch = find_final_samples(record.result_path)
        samples = load_sample_z(path)[:, :2]
        assert epoch == 50000 and len(samples) == 10000 and torch.isfinite(samples).all(), seed
        checkpoint_dir, checkpoint_epoch = find_final_checkpoint(record.result_path)
        assert checkpoint_epoch == epoch, seed
        state = torch.load(checkpoint_dir / "vi_model.pt", map_location="cpu", weights_only=True)
        finite = all(bool(torch.isfinite(value).all()) for value in state.values())
        assert finite, seed
        points = _take_points(samples, 2000, generator=_scatter_generator(42, target, "KSIVI"))
        label = f"Seed {seed}" + (" (current)" if seed == 1 else "")
        panels.append((label, points))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261006)
            sliced_w1 = compute_sliced_wasserstein(samples, reference_metric, num_projections=256, p=1)
        row = {"seed": seed, "final_epoch": epoch, "kernel": config.train.ksivi.kernel,
               **sample_summary(samples, bbox), "sliced_w1_256_projections": sliced_w1,
               "vi_checkpoint_finite": finite, "result_path": record.entry["result_path"]}
        rows.append(row)
        print(f"Seed {seed}: {row['fraction_in_plot_bounds']:.2%} in bounds; "
              f"median=({row['median_x']:.3f}, {row['median_y']:.3f}); sliced-W1={sliced_w1:.3f}")

    out_dir = repo_path(args.output_dir)
    assert out_dir is not None
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for full_range, name in [(False, "ksivi_student_seed_comparison.png"), (True, "ksivi_student_full_range.png")]:
        path = figure_dir / name
        draw_comparison(panels, target, bbox, path, full_range=full_range)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    with (out_dir / "seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
