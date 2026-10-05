"""Plot and summarize five matched seeds for both KSIVI Student-t controls."""
from __future__ import annotations

import argparse
import csv
import math
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
    completed_runs, find_final_checkpoint, find_final_samples,
    load_baseline_samples, load_manifest, load_sample_z,
)
from finalization.config import repo_path
from finalization.plots import _draw_toy_contours, _scatter_generator, _take_points, _target_bbox
from scripts.compare_ksivi_student_seeds import draw_comparison, sample_summary
from utils.metrics import compute_sliced_wasserstein


VARIANTS = [
    ("Annealing + warmup reg.", "toy_scatter_ksivi_annealing", True, "warmup_only"),
    ("Always-on reg.", "toy_scatter_ksivi_logp", False, "always"),
]


def draw_grid(points_by_case: dict, reference: np.ndarray, seeds: list[int],
              bbox: list[float], path: Path, *, full_range: bool) -> None:
    fig, axes = plt.subplots(2, len(seeds) + 1, figsize=(3.0 * (len(seeds) + 1), 6.8), squeeze=False)
    for row_index, (label, campaign, _, _) in enumerate(VARIANTS):
        points_by_column = [reference] + [points_by_case[(campaign, seed)] for seed in seeds]
        for column, (ax, points) in enumerate(zip(axes[row_index], points_by_column)):
            _draw_toy_contours(ax, "student_uc", bbox)
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
            if row_index == 0:
                ax.set_title("Ground truth" if column == 0 else f"Seed {seeds[column - 1]}", fontsize=13)
            ax.set_xlabel("x", fontsize=11)
            ax.tick_params(labelsize=10)
        row_label = "Annealing +\nwarmup reg.\ny" if row_index == 0 else "Always-on reg.\ny"
        axes[row_index, 0].set_ylabel(row_label, fontsize=12, labelpad=10)
    fig.suptitle("KSIVI on Student-t: 50,000 training iterations, 2,000 plotted samples per panel", fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=1.1, h_pad=1.4)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def mean_and_se(values: list[float]) -> tuple[float, float]:
    tensor = torch.as_tensor(values, dtype=torch.float64)
    return float(tensor.mean()), float(tensor.std(unbiased=True) / math.sqrt(len(values)))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--output-dir", default="campaigns/toy_scatter_ksivi_controls/generated_reports/finalization")
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("At least two distinct seeds are required for an SE summary.")
    bbox = _target_bbox("student_uc")
    assert bbox is not None
    baseline = load_baseline_samples("student_uc")[:, :2]
    assert torch.isfinite(baseline).all()
    reference = _take_points(baseline, 2000, generator=_scatter_generator(42, "student_uc", "Ground truth"))
    reference_metric = torch.from_numpy(_take_points(baseline, 10000, generator=_scatter_generator(42, "student_uc", "W1 reference")))
    points_by_case, rows, aggregates = {}, [], []
    reference_config = None
    for label, campaign, enabled, mode in VARIANTS:
        records = {}
        for record in completed_runs(load_manifest(f"campaigns/{campaign}/manifest.json")):
            if (record.method, record.target) == ("KSIVI", "student_uc") and record.seed in args.seeds:
                if record.seed in records:
                    raise ValueError(f"Duplicate completed seed {record.seed} in {campaign}")
                records[record.seed] = record
        if set(records) != set(args.seeds):
            raise ValueError(f"Missing completed seeds in {campaign}: {sorted(set(args.seeds) - set(records))}")
        variant_rows = []
        for seed in args.seeds:
            record = records[seed]
            config = OmegaConf.load(record.result_path / "full_config.yaml")
            assert config.seed == seed and config.train.epochs == 50000 and config.train.batch_size == 128
            assert config.train.vi.lr == 0.001 and config.train.ksivi.kernel == "riesz"
            assert config.train.annealing.enabled == enabled
            assert config.train.annealing.scheme == "linear" and config.train.annealing.steps == 25000
            assert config.train.ksivi.log_p_reg == 0.05 and config.train.ksivi.log_p_reg_mode == mode
            assert config.train.log.metric_log_freq == 0 and config.train.plot.freq == 1000000000
            if reference_config is None:
                reference_config = config
            else:
                for section in ("vi_model", "target"):
                    assert OmegaConf.to_container(config[section], resolve=True) == OmegaConf.to_container(reference_config[section], resolve=True)
                for section in ("vi", "checkpoint", "sample", "log", "plot"):
                    assert OmegaConf.to_container(config.train[section], resolve=True) == OmegaConf.to_container(reference_config.train[section], resolve=True)
                for key in ("statistic", "detach_kernel", "affine_invariant"):
                    assert config.train.ksivi[key] == reference_config.train.ksivi[key]
            sample_path, epoch = find_final_samples(record.result_path)
            samples = load_sample_z(sample_path)[:, :2]
            assert epoch == 50000 and len(samples) == 10000 and torch.isfinite(samples).all()
            checkpoint_dir, checkpoint_epoch = find_final_checkpoint(record.result_path)
            assert checkpoint_epoch == epoch
            state = torch.load(checkpoint_dir / "vi_model.pt", map_location="cpu", weights_only=True)
            finite = all(bool(torch.isfinite(value).all()) for value in state.values())
            assert finite
            points_by_case[(campaign, seed)] = _take_points(samples, 2000, generator=_scatter_generator(42, "student_uc", "KSIVI"))
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(20261006)
                distance = compute_sliced_wasserstein(samples, reference_metric, num_projections=256, p=1)
            row = {"condition": label, "campaign": campaign, "seed": seed, "final_epoch": epoch,
                   "annealing_enabled": enabled, "log_p_reg": 0.05, "log_p_reg_mode": mode,
                   **sample_summary(samples, bbox), "sliced_w1_256_projections": distance,
                   "vi_checkpoint_finite": finite, "result_path": record.entry["result_path"]}
            variant_rows.append(row)
            rows.append(row)
            print(f"{label}, seed {seed}: {row['fraction_in_plot_bounds']:.2%} in bounds; sliced-W1={distance:.3f}")
        coverage_mean, coverage_se = mean_and_se([row["fraction_in_plot_bounds"] for row in variant_rows])
        distance_mean, distance_se = mean_and_se([row["sliced_w1_256_projections"] for row in variant_rows])
        best = min(variant_rows, key=lambda row: row["sliced_w1_256_projections"])
        aggregate = {"condition": label, "seed_count": len(args.seeds),
                     "fraction_in_plot_bounds_mean": coverage_mean, "fraction_in_plot_bounds_se": coverage_se,
                     "sliced_w1_mean": distance_mean, "sliced_w1_se": distance_se,
                     "best_seed_by_sliced_w1": best["seed"], "best_sliced_w1": best["sliced_w1_256_projections"],
                     "reference_fraction_in_plot_bounds": sample_summary(baseline, bbox)["fraction_in_plot_bounds"]}
        aggregates.append(aggregate)
        print(f"{label}: sliced-W1 mean +/- SE = {distance_mean:.3f} +/- {distance_se:.3f}; best seed {best['seed']}")
    out_dir = repo_path(args.output_dir)
    assert out_dir is not None
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for full_range, name in [(False, "ksivi_student_control_seeds.png"), (True, "ksivi_student_control_seeds_full_range.png")]:
        path = figure_dir / name
        draw_grid(points_by_case, reference, args.seeds, bbox, path, full_range=full_range)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    best_panels = [("Ground truth", reference)]
    for aggregate, (_, campaign, _, _) in zip(aggregates, VARIANTS):
        seed = aggregate["best_seed_by_sliced_w1"]
        label = "Annealing + warmup reg." if campaign.endswith("annealing") else "Always-on reg."
        best_panels.append((f"{label}\nSeed {seed}", points_by_case[(campaign, seed)]))
    draw_comparison(best_panels, "student_uc", bbox, figure_dir / "ksivi_student_control_best_seeds.png",
                    full_range=False, title="KSIVI on Student-t: minimum empirical sliced-W1 within each five-seed setup")
    write_csv(out_dir / "seed_control_summary.csv", rows)
    write_csv(out_dir / "control_seed_aggregate.csv", aggregates)


if __name__ == "__main__":
    main()
