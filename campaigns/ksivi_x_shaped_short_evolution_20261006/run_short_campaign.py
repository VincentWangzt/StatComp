"""Repeat nine KSIVI comparisons for 5,000 steps with initial/500-step plots."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(REPO / "campaigns/ksivi_x_shaped_evolution_20261006"))
from run_campaign import CANONICAL, execute_rounds, specifications as previous_specifications

VARIANCE_INIT = 1.31326162815094
TRAIN_OVERRIDES = {"epochs": 5000,
                   "sample": {"freq": 500, "preserve_rng": True},
                   "plot": {"freq": 500, "initial": True}}


def specifications(results_root: Path, tb_root: Path) -> dict[str, list[dict]]:
    plans = previous_specifications(results_root, tb_root)
    fixed = previous_specifications(results_root, tb_root, VARIANCE_INIT)["seeds"]
    for spec in fixed:
        spec["results_root"] = str(results_root / "constant_variance_init" / spec["key"])
        spec["tb_root"] = str(tb_root / "constant_variance_init" / spec["key"])
        spec["command"] = [f"output.results_dir={spec['results_root']}" if arg.startswith("output.results_dir=")
                           else f"output.tb_dir={spec['tb_root']}" if arg.startswith("output.tb_dir=")
                           else arg for arg in spec["command"]]
    plans["constant_variance_init"] = fixed
    for specs in plans.values():
        for spec in specs:
            spec["train_overrides"] = TRAIN_OVERRIDES
            spec["command"].extend([
                "train.epochs=5000", "train.sample.freq=500", "train.plot.freq=500",
                "train.plot.initial=true", "train.sample.preserve_rng=true",
            ])
    return plans


def run_path(spec: dict) -> Path:
    paths = list((Path(spec["results_root"]) / "KSIVI/x_shaped").glob("*/full_config.yaml"))
    assert len(paths) == 1, paths
    return paths[0].parent


def validate_campaign(plans: dict, report_root: Path, previous_results_root: Path | None) -> dict:
    import torch

    def load(spec, step):
        return torch.load(run_path(spec) / f"checkpoints/epoch_{step}/vi_model.pt",
                          map_location="cpu", weights_only=True)

    def identical(left, right):
        return left.keys() == right.keys() and all(torch.equal(left[key], right[key]) for key in left)

    seeds = {spec["seed"]: spec for spec in plans["seeds"]}
    fixed = {spec["seed"]: spec for spec in plans["constant_variance_init"]}
    mean_checks = []
    for seed, spec in seeds.items():
        original = load(spec, 0)
        initialized = load(fixed[seed], 0)
        for key, value in original.items():
            if key.startswith("net.4."):
                assert torch.equal(value[:2], initialized[key][:2])
            else:
                assert torch.equal(value, initialized[key])
        assert torch.count_nonzero(initialized["net.4.weight"][2:]).item() == 0
        assert torch.allclose(initialized["net.4.bias"][2:], torch.ones(2))
        mean_checks.append({"seed": seed, "same_hidden_layers_and_mean_head": True,
                            "constant_variance_head_weights_zero": True})
    reference = load(seeds[43], 0)
    reference_samples = torch.load(run_path(seeds[43]) / "samples/samples_epoch_0.pt",
                                  map_location="cpu", weights_only=True)
    for spec in plans["learning_rates"]:
        assert identical(reference, load(spec, 0))
        samples = torch.load(run_path(spec) / "samples/samples_epoch_0.pt",
                             map_location="cpu", weights_only=True)
        assert torch.equal(reference_samples["epsilon"], samples["epsilon"])
        assert torch.equal(reference_samples["z"], samples["z"])
    checks = {"learning_rates_have_identical_initial_weights_and_samples": True,
              "constant_variance_mean_initialization_checks": mean_checks,
              "annealing_steps": 25000, "annealing_factor_at_step_5000": 0.28,
              "previous_run_checkpoint_comparisons": []}
    if previous_results_root is not None:
        for group, specs in plans.items():
            previous = ("ksivi_x_shaped_constant_variance_init_20261006" if group == "constant_variance_init"
                        else "ksivi_x_shaped_evolution_20261006")
            previous_group = "seeds" if group == "constant_variance_init" else group
            for spec in specs:
                paths = list((previous_results_root / previous / previous_group / spec["key"] /
                              "KSIVI/x_shaped").glob("*/checkpoints/epoch_5000/vi_model.pt"))
                assert len(paths) == 1, paths
                earlier = torch.load(paths[0], map_location="cpu", weights_only=True)
                current = load(spec, 5000)
                assert earlier.keys() == current.keys()
                maximum_difference = max((current[key] - earlier[key]).abs().max().item() for key in current)
                checks["previous_run_checkpoint_comparisons"].append({
                    "group": group, "key": spec["key"], "previous_checkpoint": str(paths[0]),
                    "identical_weights": identical(current, earlier),
                    "maximum_absolute_weight_difference": maximum_difference})
    (report_root / "initialization_checks.json").write_text(json.dumps(checks, indent=2) + "\n")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tb-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=CAMPAIGN_DIR / "generated_reports")
    parser.add_argument("--previous-results-root", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    canonical = OmegaConf.load(CANONICAL)
    assert canonical.vi_model_type == "ConditionalGaussian"
    assert canonical.train.epochs == 50000 and canonical.train.batch_size == 128
    assert canonical.train.vi.lr == 0.001 and canonical.train.annealing.enabled
    assert canonical.train.annealing.steps == 25000
    assert canonical.train.plot.num == canonical.train.sample.num == 10000
    plans = specifications(args.results_root, args.tb_root)
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return
    execute_rounds(plans, args.results_root, args.report_root,
                   lambda p, r: validate_campaign(p, r, args.previous_results_root))


if __name__ == "__main__":
    main()
