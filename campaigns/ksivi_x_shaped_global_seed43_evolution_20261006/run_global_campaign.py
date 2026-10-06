"""Run seed-43 global Gaussian KSIVI for 5k/50k steps with initial plots."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

from omegaconf import OmegaConf

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
sys.path.insert(0, str(REPO / "campaigns/ksivi_x_shaped_evolution_20261006"))
from run_campaign import CANONICAL, execute_rounds


def specifications(results_root: Path, tb_root: Path) -> dict[str, list[dict]]:
    specs = []
    for key, epochs, freq in (("short", 5000, 500), ("long", 50000, 5000)):
        row_results, row_tb = results_root / key, tb_root / key
        train_overrides = {"epochs": epochs,
                           "sample": {"freq": freq, "preserve_rng": True},
                           "plot": {"freq": freq, "initial": True}}
        command = [sys.executable, "-u", str(REPO / "src.py"), "--config", str(CANONICAL),
                   "seed=43", "vi_model_type=ConditionalGaussianGlobal",
                   f"train.epochs={epochs}", f"train.sample.freq={freq}", f"train.plot.freq={freq}",
                   "train.plot.initial=true", "train.sample.preserve_rng=true",
                   "train.log.metric_log_freq=0", "metric.kl_ite.enabled=false",
                   "metric.w2.enabled=false", "metric.elbo.enabled=false",
                   f"output.results_dir={row_results}", f"output.tb_dir={row_tb}"]
        specs.append({"key": key, "label": f"{epochs:,} steps", "seed": 43,
                      "epochs": epochs, "plot_freq": freq, "train_overrides": train_overrides,
                      "results_root": str(row_results), "tb_root": str(row_tb), "command": command})
    return {"global_seed43": specs}


def finalize_round(specs: list[dict], report_root: Path, source_commit: str,
                   canonical_path: Path, round_name: str) -> dict:
    import torch
    from PIL import Image
    from plot_evolution import render_grid

    canonical = OmegaConf.load(canonical_path)
    report_root.mkdir(parents=True, exist_ok=True)
    rows, runs, paths, model_configs = [], [], {}, []
    for spec in specs:
        root = Path(spec["results_root"])
        configs = list((root / "KSIVI/x_shaped").glob("*/full_config.yaml"))
        assert len(configs) == 1, configs
        run_path = configs[0].parent
        paths[spec["key"]] = run_path
        cfg = OmegaConf.load(configs[0])
        expected_train = OmegaConf.to_container(
            OmegaConf.merge(canonical.train, spec["train_overrides"]), resolve=True)
        expected_train["log"]["metric_log_freq"] = 0
        assert OmegaConf.to_container(cfg.train, resolve=True) == expected_train
        assert cfg.seed == 43 and cfg.vi_model_type == "ConditionalGaussianGlobal"
        assert cfg.vi_model.hidden_dim == 128 and cfg.vi_model.num_layers == 2
        assert cfg.vi_model.epsilon_dim == cfg.vi_model.z_dim == 2
        assert cfg.vi_model.get("variance_parameterization", "softplus_var") == "softplus_var"
        assert cfg.vi_model.get("global_variance_init", None) is None
        assert not any(value.get("enabled", False) for value in cfg.metric.values())
        model_configs.append(OmegaConf.to_container(cfg.vi_model, resolve=True))
        log = (run_path / "run.log").read_text()
        assert "Training completed." in log and "NaN or Inf detected" not in log
        assert int((root / "runtime/exit_code.txt").read_text()) == 0
        steps = tuple(range(0, spec["epochs"] + 1, spec["plot_freq"]))
        panels = []
        for step in steps:
            plot = run_path / f"plots/contour_epoch_{step}.png"
            with Image.open(plot) as image:
                image.verify()
            data = torch.load(run_path / f"samples/samples_epoch_{step}.pt",
                              map_location="cpu", weights_only=True)
            assert data["epoch"] == step
            assert data["epsilon"].shape == data["z"].shape == (10000, 2)
            assert torch.isfinite(data["epsilon"]).all().item() and torch.isfinite(data["z"]).all().item()
            panels.append({"step": step, "sample_count": 10000, "all_samples_finite": True,
                           "plot": str(plot), "plot_sha256": hashlib.sha256(plot.read_bytes()).hexdigest()})
        variance = {}
        for step in (0, spec["epochs"]):
            checkpoint = run_path / f"checkpoints/epoch_{step}"
            state = torch.load(checkpoint / "vi_model.pt", map_location="cpu", weights_only=True)
            optimizer = torch.load(checkpoint / "vi_optim.pt", map_location="cpu", weights_only=True)
            scheduler = torch.load(checkpoint / "vi_sched.pt", map_location="cpu", weights_only=True)
            assert state["net.0.weight"].shape == (128, 2)
            assert state["net.2.weight"].shape == (128, 128)
            assert state["net.4.weight"].shape == (2, 128) and state["var_raw"].shape == (2,)
            assert all(torch.isfinite(value).all().item() for value in state.values())
            assert len(optimizer["param_groups"]) == 2
            assert all(tuple(group["betas"]) == (0.9, 0.999) for group in optimizer["param_groups"])
            expected_lrs = (cfg.train.vi.lr, cfg.train.vi.var_lr)
            assert all(abs(group["lr"] - lr * 0.9 ** (step // 1000)) < 1e-12
                       for group, lr in zip(optimizer["param_groups"], expected_lrs))
            assert scheduler["last_epoch"] == step
            if step == 0:
                assert not optimizer["state"] and torch.equal(state["var_raw"], torch.ones(2))
            variance[str(step)] = torch.nn.functional.softplus(state["var_raw"]).clamp(min=1e-4).tolist()
        row = {"label": spec["label"], "plots": [p["plot"] for p in panels]}
        title = f"KSIVI x-shaped: global variance, seed 43, {spec['epochs']:,} steps"
        grid = render_grid([row], report_root / f"{spec['key']}_evolution_1x{len(steps)}.png",
                           title, " | initial variance 1.3133", steps, "ConditionalGaussianGlobal")
        duration = re.search(r"Training completed\. Total time: ([0-9.]+)s", log)
        rows.append(row)
        runs.append({"key": spec["key"], "seed": 43, "total_steps": spec["epochs"],
                     "steps": list(steps), "plot_freq": spec["plot_freq"], "grid_shape": [1, len(steps)],
                     "run_path": str(run_path), "exit_code": 0, "initial_lr": cfg.train.vi.lr,
                     "final_lr": cfg.train.vi.lr * 0.9 ** (spec["epochs"] // 1000),
                     "annealing_steps": cfg.train.annealing.steps,
                     "final_annealing_factor": 0.1 + 0.9 * min(1, spec["epochs"] / cfg.train.annealing.steps),
                     "global_variance": variance, "panels": panels,
                     "elapsed_seconds": float(duration.group(1)), **grid})
        shutil.copy2(configs[0], report_root / f"full_config_{spec['key']}.yaml")
    assert all(config == model_configs[0] for config in model_configs)
    checks = {}
    if len(specs) == 2:
        initial = [torch.load(paths[spec["key"]] / "checkpoints/epoch_0/vi_model.pt",
                              map_location="cpu", weights_only=True) for spec in specs]
        assert all(torch.equal(initial[0][key], initial[1][key]) for key in initial[0])
        samples = [torch.load(paths[spec["key"]] / "samples/samples_epoch_0.pt",
                              map_location="cpu", weights_only=True) for spec in specs]
        assert torch.equal(samples[0]["epsilon"], samples[1]["epsilon"])
        assert torch.equal(samples[0]["z"], samples[1]["z"])
        shared_step = min(spec["epochs"] for spec in specs)
        states = [torch.load(paths[spec["key"]] / f"checkpoints/epoch_{shared_step}/vi_model.pt",
                             map_location="cpu", weights_only=True) for spec in specs]
        assert all(torch.equal(states[0][key], states[1][key]) for key in states[0])
        checks = {"initial_weights_identical": True, "initial_samples_identical": True,
                  "shared_step": shared_step, "shared_step_weights_identical": True}
    grid = render_grid(rows, report_root / f"{round_name}_evolution_{len(rows)}x{len(runs[0]['steps'])}.png",
                       "KSIVI x-shaped: global variance, seed 43, LR 0.001",
                       " | initial variance 1.3133", tuple(runs[0]["steps"]), "ConditionalGaussianGlobal",
                       "Top: 0 to 5,000, every 500. Bottom: 0 to 50,000, every 5,000. "
                       "First column: initialization. Orange: variational samples; blue: target contours."
                       if len(rows) == 2 else None)
    manifest = {"round": round_name, "source_commit": source_commit,
                "vi_model_type": "ConditionalGaussianGlobal", "seed": 43,
                "hidden_width": 128, "input_noise_dim": 2, "batch_size": 128,
                "annealing_enabled": True, "annealing_steps": 25000,
                "metric_evaluation_temporarily_disabled": True,
                "includes_initial_distribution": True, "reporting_preserves_training_rng": True,
                "grid_shape": [len(rows), len(runs[0]["steps"])],
                "runs": runs, "checks": checks, **grid}
    (report_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tb-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=CAMPAIGN_DIR / "generated_reports")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    canonical = OmegaConf.load(CANONICAL)
    assert canonical.train.vi.lr == canonical.train.vi.var_lr == 0.001
    assert canonical.train.batch_size == 128 and canonical.train.annealing.enabled
    assert canonical.train.annealing.steps == 25000
    assert canonical.train.plot.num == canonical.train.sample.num == 10000
    plans = specifications(args.results_root, args.tb_root)
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return
    execute_rounds(plans, args.results_root, args.report_root, round_finalizer=finalize_round)


if __name__ == "__main__":
    main()
