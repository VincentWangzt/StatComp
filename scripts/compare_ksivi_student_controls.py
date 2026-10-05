"""Compare matched KSIVI Student-t runs with different annealing and log-p settings."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from omegaconf import OmegaConf
import torch

from finalization.artifacts import (
    completed_runs, find_final_checkpoint, find_final_samples,
    load_baseline_samples, load_manifest, load_sample_z,
)
from finalization.config import repo_path
from finalization.plots import _scatter_generator, _take_points, _target_bbox
from scripts.compare_ksivi_student_seeds import draw_comparison, sample_summary
from utils.metrics import compute_sliced_wasserstein


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-dir", default="campaigns/toy_scatter_ksivi_controls/generated_reports/finalization")
    args = parser.parse_args()
    variants = [
        ("Original setup", "toy_scatter_ksivi_additional", False, "warmup_only"),
        ("Annealing + warmup reg.", "toy_scatter_ksivi_annealing", True, "warmup_only"),
        ("Always-on reg.", "toy_scatter_ksivi_logp", False, "always"),
    ]
    target = "student_uc"
    bbox = _target_bbox(target)
    assert bbox is not None
    baseline = load_baseline_samples(target)[:, :2]
    assert torch.isfinite(baseline).all()
    reference_points = _take_points(baseline, 2000, generator=_scatter_generator(42, target, "Ground truth"))
    reference_metric = torch.from_numpy(_take_points(baseline, 10000, generator=_scatter_generator(42, target, "W1 reference")))
    panels = [("Ground truth", reference_points)]
    rows = [{"condition": "ground_truth", "seed": "", "final_epoch": "", "annealing_enabled": "",
             "log_p_reg": "", "log_p_reg_mode": "", "reg_last_iteration": "",
             **sample_summary(baseline, bbox), "sliced_w1_256_projections": "",
             "vi_checkpoint_finite": "", "result_path": "baselines/exact/student_uc_exact_100k.pt"}]
    original_config = None
    for label, campaign, annealing_enabled, mode in variants:
        matches = [record for record in completed_runs(load_manifest(f"campaigns/{campaign}/manifest.json"))
                   if (record.method, record.target, record.seed) == ("KSIVI", target, args.seed)]
        if len(matches) != 1:
            raise ValueError(f"Expected one completed run for {campaign}, seed {args.seed}; found {len(matches)}")
        record = matches[0]
        config = OmegaConf.load(record.result_path / "full_config.yaml")
        assert config.seed == args.seed
        assert config.train.epochs == 50000 and config.train.batch_size == 128
        assert config.train.vi.lr == 0.001 and config.train.ksivi.kernel == "riesz"
        assert config.train.annealing.enabled == annealing_enabled
        assert config.train.annealing.scheme == "linear" and config.train.annealing.steps == 25000
        assert config.train.ksivi.log_p_reg == 0.05 and config.train.ksivi.log_p_reg_mode == mode
        assert config.train.log.metric_log_freq == 0 and config.train.plot.freq == 1000000000
        if original_config is None:
            original_config = config
        else:
            for section in ("vi_model", "target"):
                assert OmegaConf.to_container(config[section], resolve=True) == OmegaConf.to_container(original_config[section], resolve=True)
            for section in ("vi", "checkpoint", "sample", "log", "plot"):
                assert OmegaConf.to_container(config.train[section], resolve=True) == OmegaConf.to_container(original_config.train[section], resolve=True)
            for key in ("statistic", "detach_kernel", "affine_invariant"):
                assert config.train.ksivi[key] == original_config.train.ksivi[key]
        sample_path, epoch = find_final_samples(record.result_path)
        samples = load_sample_z(sample_path)[:, :2]
        assert epoch == 50000 and len(samples) == 10000 and torch.isfinite(samples).all()
        checkpoint_dir, checkpoint_epoch = find_final_checkpoint(record.result_path)
        assert checkpoint_epoch == epoch
        state = torch.load(checkpoint_dir / "vi_model.pt", map_location="cpu", weights_only=True)
        finite = all(bool(torch.isfinite(value).all()) for value in state.values())
        assert finite
        points = _take_points(samples, 2000, generator=_scatter_generator(42, target, "KSIVI"))
        panels.append((label, points))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(20261006)
            sliced_w1 = compute_sliced_wasserstein(samples, reference_metric, num_projections=256, p=1)
        row = {"condition": label, "seed": args.seed, "final_epoch": epoch, "annealing_enabled": annealing_enabled,
               "log_p_reg": config.train.ksivi.log_p_reg, "log_p_reg_mode": mode,
               "reg_last_iteration": 50000 if mode == "always" else (24999 if annealing_enabled else 0),
               **sample_summary(samples, bbox), "sliced_w1_256_projections": sliced_w1,
               "vi_checkpoint_finite": finite, "result_path": record.entry["result_path"]}
        rows.append(row)
        print(f"{label}: {row['fraction_in_plot_bounds']:.2%} in bounds; sliced-W1={sliced_w1:.3f}")
    out_dir = repo_path(args.output_dir)
    assert out_dir is not None
    figure_dir = out_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    title = f"KSIVI on Student-t: seed {args.seed}, 50,000 training iterations, 2,000 plotted samples"
    for full_range, name in [(False, "ksivi_student_controls.png"), (True, "ksivi_student_controls_full_range.png")]:
        path = figure_dir / name
        draw_comparison(panels, target, bbox, path, full_range=full_range, title=title)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    with (out_dir / "control_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
