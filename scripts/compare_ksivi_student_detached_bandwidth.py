"""Compare three KSIVI Student-t setups with only the median bandwidth detached."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from finalization.artifacts import (
    completed_runs, find_final_checkpoint, find_final_samples,
    load_baseline_samples, load_manifest, load_sample_z,
)
from finalization.config import repo_path
from finalization.plots import _draw_toy_contours, _scatter_generator, _take_points, _target_bbox
from scripts.compare_ksivi_student_control_seeds import mean_and_se, write_csv
from scripts.compare_ksivi_student_seeds import draw_comparison, sample_summary
from utils.metrics import compute_sliced_wasserstein


VARIANTS = [
    ("Annealing + warmup reg.", "annealing", True, 0.05, "warmup_only", ["toy_scatter_ksivi_annealing"]),
    ("Always-on reg.", "logp", False, 0.05, "always", ["toy_scatter_ksivi_logp"]),
    ("No reg., no warmup", "plain", False, 0.0, "warmup_only", ["toy_scatter_grid", "toy_scatter_ksivi_additional"]),
]


def matched_settings(config: DictConfig) -> dict:
    """Settings shared by all controls, independent of bandwidth and regularization."""
    return {
        "vi_model": OmegaConf.to_container(config.vi_model, resolve=True),
        "target": OmegaConf.to_container(config.target, resolve=True),
        "train": {key: OmegaConf.to_container(config.train[key], resolve=True)
                  for key in ("vi", "checkpoint", "sample", "log", "plot")},
        "epochs": config.train.epochs, "batch_size": config.train.batch_size,
        "annealing_steps": config.train.annealing.steps, "annealing_scheme": config.train.annealing.scheme,
        "ksivi": {key: config.train.ksivi[key] for key in ("statistic", "kernel", "detach_kernel", "affine_invariant")},
    }


def records_for(manifests: list[str], seeds: list[int]) -> dict:
    records = {}
    for campaign in manifests:
        for record in completed_runs(load_manifest(f"campaigns/{campaign}/manifest.json")):
            if (record.method, record.target) == ("KSIVI", "student_uc") and record.seed in seeds:
                if record.seed in records:
                    raise ValueError(f"Duplicate completed seed {record.seed} in {manifests}")
                records[record.seed] = record
    return records


def evaluate_record(record, reference_metric: torch.Tensor, bbox: list[float]) -> tuple[dict, np.ndarray]:
    sample_path, epoch = find_final_samples(record.result_path)
    samples = load_sample_z(sample_path)[:, :2]
    assert epoch == 50000 and len(samples) == 10000 and torch.isfinite(samples).all()
    checkpoint_dir, checkpoint_epoch = find_final_checkpoint(record.result_path)
    assert checkpoint_epoch == epoch
    state = torch.load(checkpoint_dir / "vi_model.pt", map_location="cpu", weights_only=True)
    assert all(bool(torch.isfinite(value).all()) for value in state.values())
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261006)
        distance = compute_sliced_wasserstein(samples, reference_metric, num_projections=256, p=1)
    row = {"seed": record.seed, "final_epoch": epoch, **sample_summary(samples, bbox),
           "sliced_w1_256_projections": distance, "vi_checkpoint_finite": True,
           "result_path": record.entry["result_path"]}
    points = _take_points(samples, 2000, generator=_scatter_generator(42, "student_uc", "KSIVI"))
    return row, points


def draw_grid(points_by_case: dict, reference: np.ndarray, seeds: list[int],
              bbox: list[float], path: Path, *, full_range: bool) -> None:
    fig, axes = plt.subplots(3, len(seeds) + 1, figsize=(3.0 * (len(seeds) + 1), 9.8), squeeze=False)
    row_labels = ["Annealing +\nwarmup reg.\ny", "Always-on reg.\ny", "No reg.,\nno warmup\ny"]
    for row_index, (_, suffix, *_) in enumerate(VARIANTS):
        columns = [reference] + [points_by_case[(suffix, seed)] for seed in seeds]
        for column, (ax, points) in enumerate(zip(axes[row_index], columns)):
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
        axes[row_index, 0].set_ylabel(row_labels[row_index], fontsize=12, labelpad=10)
    fig.suptitle("KSIVI on Student-t: detached median bandwidth, 50,000 iterations, 2,000 plotted samples", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96), w_pad=1.1, h_pad=1.3)
    fig.savefig(path, dpi=240, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--output-dir", default="campaigns/toy_scatter_ksivi_detached/generated_reports/finalization")
    args = parser.parse_args()
    if len(args.seeds) < 2 or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("At least two distinct seeds are required for an SE summary.")
    bbox = _target_bbox("student_uc")
    assert bbox is not None
    baseline = load_baseline_samples("student_uc")[:, :2]
    assert torch.isfinite(baseline).all()
    reference = _take_points(baseline, 2000, generator=_scatter_generator(42, "student_uc", "Ground truth"))
    reference_metric = torch.from_numpy(_take_points(baseline, 10000, generator=_scatter_generator(42, "student_uc", "W1 reference")))
    points_by_case, rows, aggregates, paired = {}, [], [], []
    reference_settings = None
    for label, suffix, enabled, reg, mode, prior_manifests in VARIANTS:
        campaign = f"toy_scatter_ksivi_detached_{suffix}"
        records = records_for([campaign], args.seeds)
        assert set(records) == set(args.seeds), f"Missing completed seeds in {campaign}"
        prior_records = records_for(prior_manifests, args.seeds)
        variant_rows = []
        for seed in args.seeds:
            record = records[seed]
            config = OmegaConf.load(record.result_path / "full_config.yaml")
            assert config.seed == seed and config.train.epochs == 50000 and config.train.batch_size == 128
            assert config.train.vi.lr == 0.001 and config.train.ksivi.kernel == "riesz"
            assert not config.train.ksivi.detach_kernel and config.train.ksivi.detach_bandwidth
            assert config.train.annealing.enabled == enabled
            assert config.train.annealing.scheme == "linear" and config.train.annealing.steps == 25000
            assert config.train.ksivi.log_p_reg == reg and config.train.ksivi.log_p_reg_mode == mode
            assert config.train.log.metric_log_freq == 0 and config.train.plot.freq == 1000000000
            settings = matched_settings(config)
            if reference_settings is None:
                reference_settings = settings
            assert settings == reference_settings
            metrics, points = evaluate_record(record, reference_metric, bbox)
            row = {"condition": label, "campaign": campaign, "detach_bandwidth": True, "detach_kernel": False,
                   "annealing_enabled": enabled, "log_p_reg": reg, "log_p_reg_mode": mode, **metrics}
            points_by_case[(suffix, seed)] = points
            rows.append(row)
            variant_rows.append(row)
            comparison = {"condition": label, "seed": seed, "baseline_available": seed in prior_records,
                          "attached_bandwidth_sliced_w1": "", "detached_bandwidth_sliced_w1": metrics["sliced_w1_256_projections"],
                          "sliced_w1_change_detached_minus_attached": "", "attached_bandwidth_coverage": "",
                          "detached_bandwidth_coverage": metrics["fraction_in_plot_bounds"], "attached_result_path": "",
                          "detached_result_path": metrics["result_path"]}
            if seed in prior_records:
                prior = prior_records[seed]
                prior_config = OmegaConf.load(prior.result_path / "full_config.yaml")
                assert prior_config.seed == seed and matched_settings(prior_config) == settings
                assert not prior_config.train.ksivi.get("detach_bandwidth", False)
                assert prior_config.train.annealing.enabled == enabled and prior_config.train.ksivi.log_p_reg_mode == mode
                if reg > 0:
                    assert prior_config.train.ksivi.log_p_reg == reg
                else:
                    assert not prior_config.train.annealing.enabled and prior_config.train.ksivi.log_p_reg_mode == "warmup_only"
                old_metrics, _ = evaluate_record(prior, reference_metric, bbox)
                comparison.update(attached_bandwidth_sliced_w1=old_metrics["sliced_w1_256_projections"],
                                  sliced_w1_change_detached_minus_attached=metrics["sliced_w1_256_projections"] - old_metrics["sliced_w1_256_projections"],
                                  attached_bandwidth_coverage=old_metrics["fraction_in_plot_bounds"],
                                  attached_result_path=old_metrics["result_path"])
            paired.append(comparison)
            print(f"{label}, seed {seed}: {row['fraction_in_plot_bounds']:.2%} in bounds; sliced-W1={row['sliced_w1_256_projections']:.3f}")
        coverage_mean, coverage_se = mean_and_se([r["fraction_in_plot_bounds"] for r in variant_rows])
        distance_mean, distance_se = mean_and_se([r["sliced_w1_256_projections"] for r in variant_rows])
        best = min(variant_rows, key=lambda r: r["sliced_w1_256_projections"])
        aggregates.append({"condition": label, "seed_count": len(args.seeds),
                           "fraction_in_plot_bounds_mean": coverage_mean, "fraction_in_plot_bounds_se": coverage_se,
                           "sliced_w1_mean": distance_mean, "sliced_w1_se": distance_se,
                           "best_seed_by_sliced_w1": best["seed"], "best_sliced_w1": best["sliced_w1_256_projections"],
                           "reference_fraction_in_plot_bounds": sample_summary(baseline, bbox)["fraction_in_plot_bounds"]})
        print(f"{label}: sliced-W1 mean +/- SE = {distance_mean:.3f} +/- {distance_se:.3f}; best seed {best['seed']}")
    out_dir = repo_path(args.output_dir)
    assert out_dir is not None
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for full_range, name in [(False, "ksivi_student_detached_bandwidth.png"), (True, "ksivi_student_detached_bandwidth_full_range.png")]:
        draw_grid(points_by_case, reference, args.seeds, bbox, figure_dir / name, full_range=full_range)
    best_panels = [("Ground truth", reference)]
    for aggregate, (_, suffix, *_) in zip(aggregates, VARIANTS):
        seed = aggregate["best_seed_by_sliced_w1"]
        best_panels.append((f"{aggregate['condition']}\nSeed {seed}", points_by_case[(suffix, seed)]))
    draw_comparison(best_panels, "student_uc", bbox, figure_dir / "ksivi_student_detached_bandwidth_best_seeds.png",
                    full_range=False, title="KSIVI: detached median bandwidth, minimum empirical sliced-W1 within each three-seed setup")
    write_csv(out_dir / "seed_summary.csv", rows)
    write_csv(out_dir / "aggregate_summary.csv", aggregates)
    write_csv(out_dir / "bandwidth_paired_comparison.csv", paired)


if __name__ == "__main__":
    main()
