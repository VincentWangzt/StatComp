"""Plot the requested AISIVI seeds and record their complete sample summaries."""
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
import torch

from finalization.artifacts import completed_runs, find_final_checkpoint, find_final_samples, load_manifest, load_sample_z
from finalization.config import repo_path
from finalization.plots import _draw_toy_contours, _scatter_generator, _take_points, _target_bbox


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="campaigns/toy_scatter_aisivi_additional/manifest.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[45, 43, 42, 46])
    parser.add_argument("--output-dir", default="campaigns/toy_scatter_aisivi_additional/generated_reports/finalization")
    parser.add_argument("--full-range", action="store_true", help="Show each complete sample cloud on its own axes.")
    args = parser.parse_args()
    target = "8_gaussians"
    bbox = _target_bbox(target)
    assert bbox is not None
    records = {
        record.seed: record for record in completed_runs(load_manifest(args.manifest))
        if (record.method, record.target) == ("AISIVI", target)
    }
    missing = set(args.seeds) - records.keys()
    if missing:
        raise ValueError(f"Missing completed AISIVI runs: {sorted(missing)}")
    points_by_seed = {}
    rows = []
    for seed in args.seeds:
        record = records[seed]
        path, epoch = find_final_samples(record.result_path)
        samples = load_sample_z(path)
        assert epoch == 10000 and len(samples) == 10000 and torch.isfinite(samples).all()
        checkpoint_dir, checkpoint_epoch = find_final_checkpoint(record.result_path)
        assert checkpoint_epoch == epoch
        checkpoint_finite = {}
        for name in ("vi_model", "reverse_model"):
            state = torch.load(checkpoint_dir / f"{name}.pt", map_location="cpu", weights_only=True)
            checkpoint_finite[name] = all(bool(torch.isfinite(value).all()) for value in state.values())
        assert checkpoint_finite["vi_model"], seed
        points = _take_points(samples[:, :2], 2000, generator=_scatter_generator(42, target, "AISIVI"))
        points_by_seed[seed] = points
        inside = ((samples[:, 0] >= bbox[0]) & (samples[:, 0] <= bbox[1])
                  & (samples[:, 1] >= bbox[2]) & (samples[:, 1] <= bbox[3]))
        rows.append({
            "seed": seed, "final_epoch": epoch, "sample_count": len(samples),
            "fraction_in_plot_bounds": float(inside.float().mean()),
            "mean_x": float(samples[:, 0].mean()), "mean_y": float(samples[:, 1].mean()),
            "std_x": float(samples[:, 0].std()), "std_y": float(samples[:, 1].std()),
            "vi_checkpoint_finite": checkpoint_finite["vi_model"],
            "reverse_checkpoint_finite": checkpoint_finite["reverse_model"],
            "result_path": record.entry["result_path"],
        })
        print(f"Seed {seed}: {rows[-1]['fraction_in_plot_bounds']:.2%} of saved samples in target bounds; "
              f"finite reverse checkpoint: {checkpoint_finite['reverse_model']}")

    out_dir = repo_path(args.output_dir)
    assert out_dir is not None
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(args.seeds), figsize=(3.0 * len(args.seeds), 3.6), squeeze=False)
    for ax, seed in zip(axes[0], args.seeds):
        _draw_toy_contours(ax, target, bbox)
        points = points_by_seed[seed]
        ax.plot(points[:, 0], points[:, 1], ".", markersize=3, color="#ff7f0e", alpha=0.45)
        if args.full_range:
            bounds = [min(bbox[0], points[:, 0].min()), max(bbox[1], points[:, 0].max()),
                      min(bbox[2], points[:, 1].min()), max(bbox[3], points[:, 1].max())]
            x_pad, y_pad = 0.03 * (bounds[1] - bounds[0]), 0.03 * (bounds[3] - bounds[2])
            ax.set_xlim(bounds[0] - x_pad, bounds[1] + x_pad)
            ax.set_ylim(bounds[2] - y_pad, bounds[3] + y_pad)
            ax.set_aspect("equal", adjustable="box")
        else:
            ax.set_xticks([-5, 0, 5])
            ax.set_yticks([-5, 0, 5])
        ax.set_title(f"Seed {seed}", fontsize=13)
        ax.set_xlabel("x")
        ax.tick_params(labelsize=10)
    axes[0, 0].set_ylabel("y")
    fig.suptitle("AISIVI on eight Gaussians: 10,000 training iterations, 2,000 plotted samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=1.1)
    name = "aisivi_8_gaussians_full_range.png" if args.full_range else "aisivi_8_gaussians_seed_comparison.png"
    comparison = figure_dir / name
    fig.savefig(comparison, dpi=240, facecolor="white")
    plt.close(fig)
    with (out_dir / "seed_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {comparison.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
