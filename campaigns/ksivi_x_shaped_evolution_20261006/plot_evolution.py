"""Validate saved KSIVI runs and assemble their native plots into a 3x10 grid."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

STEPS = tuple(range(5000, 50001, 5000))


def render_grid(rows: list[dict], output: Path, title: str,
                initialization: str = "", steps: tuple[int, ...] = STEPS,
                model_type: str = "ConditionalGaussian", footer: str | None = None) -> dict:
    """Use the exact native training plots, without resampling the model."""
    if not rows or any(len(row["plots"]) != len(steps) for row in rows):
        raise ValueError("The evolution grid requires one plot per column in each row")
    output.parent.mkdir(parents=True, exist_ok=True)
    height = 3.0 * len(rows) + 1.6
    fig, axes = plt.subplots(len(rows), len(steps), figsize=(3.4 * len(steps), height), squeeze=False)
    fig.subplots_adjust(left=0.06, right=0.997, bottom=0.477 / height, top=1 - 1.06 / height,
                        hspace=0.01, wspace=0.01)
    fig.suptitle(title, fontsize=23, y=1 - 0.053 / height)
    fig.text(0.5, 1 - 0.689 / height,
             f"{model_type} | width 128 | noise dimension 2 | batch 128 | annealing enabled"
             + initialization,
             ha="center", fontsize=16)
    for row_index, row in enumerate(rows):
        for col, path in enumerate(row["plots"]):
            with Image.open(path) as source:
                source.load()
                pixels = np.asarray(source.convert("RGB"))
            axes[row_index, col].imshow(pixels)
            axes[row_index, col].set_axis_off()
        box = axes[row_index, 0].get_position()
        fig.text(0.027, (box.y0 + box.y1) / 2, row["label"],
                 ha="center", va="center", rotation=90, fontsize=19)
    interval = steps[1] - steps[0]
    initial = " (first column: initialization)" if steps[0] == 0 else ""
    if footer is None:
        footer = (f"Columns: {steps[0]:,} to {steps[-1]:,} updates, every {interval:,}{initial}. "
                  "Orange: variational samples; blue: target contours.")
    fig.text(0.5, 0.1802 / height, footer,
             ha="center", fontsize=14)
    fig.savefig(output, dpi=160, facecolor="white")
    fig.savefig(output.with_suffix(".pdf"), facecolor="white")
    plt.close(fig)
    with Image.open(output) as image:
        size = list(image.size)
        image.verify()
    return {"grid": output.name, "pdf": output.with_suffix(".pdf").name,
            "grid_size": size, "grid_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}


def finalize_round(specs: list[dict], report_root: Path, source_commit: str,
                   canonical_path: Path, round_name: str) -> dict:
    """Check configs, samples, and checkpoints before producing a report."""
    canonical = OmegaConf.load(canonical_path)
    report_root.mkdir(parents=True, exist_ok=True)
    rows, summaries = [], []
    model_configs = []
    snapshot_steps = None
    for spec in specs:
        root = Path(spec["results_root"])
        paths = list((root / "KSIVI/x_shaped").glob("*/full_config.yaml"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one run for {spec['key']}, found {paths}")
        run_path = paths[0].parent
        cfg = OmegaConf.load(paths[0])
        expected_train = OmegaConf.to_container(
            OmegaConf.merge(canonical.train, spec.get("train_overrides", {})), resolve=True)
        expected_train["vi"]["lr"] = spec["lr"]
        expected_train["log"]["metric_log_freq"] = 0
        assert OmegaConf.to_container(cfg.train, resolve=True) == expected_train
        assert cfg.seed == spec["seed"] and cfg.vi_model_type == "ConditionalGaussian"
        assert cfg.vi_model.hidden_dim == 128 and cfg.vi_model.num_layers == 2
        assert cfg.vi_model.epsilon_dim == cfg.vi_model.z_dim == 2
        assert cfg.vi_model.get("variance_init", None) == spec.get("variance_init")
        assert not any(v.get("enabled", False) for v in cfg.metric.values())
        initial_enabled = bool(cfg.train.plot.get("initial", False))
        steps = ((0,) if initial_enabled else ()) + tuple(
            range(cfg.train.plot.freq, cfg.train.epochs + 1, cfg.train.plot.freq))
        if snapshot_steps is None:
            snapshot_steps = steps
        assert steps == snapshot_steps
        model_configs.append(OmegaConf.to_container(cfg.vi_model, resolve=True))
        log = (run_path / "run.log").read_text()
        assert "Training completed." in log and "NaN or Inf detected" not in log
        exit_code = int((root / "runtime/exit_code.txt").read_text().strip())
        assert exit_code == 0
        panels = []
        for step in steps:
            plot = run_path / f"plots/contour_epoch_{step}.png"
            with Image.open(plot) as image:
                image.verify()
            samples = torch.load(run_path / f"samples/samples_epoch_{step}.pt",
                                 map_location="cpu", weights_only=True)
            assert samples["epoch"] == step
            assert samples["epsilon"].shape == samples["z"].shape == (10000, 2)
            assert torch.isfinite(samples["epsilon"]).all().item()
            assert torch.isfinite(samples["z"]).all().item()
            panels.append({"step": step, "plot": str(plot), "sample_count": 10000,
                           "all_samples_finite": True,
                           "plot_sha256": hashlib.sha256(plot.read_bytes()).hexdigest()})
        checkpoint = run_path / f"checkpoints/epoch_{cfg.train.epochs}"
        state = torch.load(checkpoint / "vi_model.pt", map_location="cpu", weights_only=True)
        optimizer = torch.load(checkpoint / "vi_optim.pt", map_location="cpu", weights_only=True)
        assert "var_raw" not in state
        assert state["net.0.weight"].shape == (128, 2)
        assert state["net.2.weight"].shape == (128, 128)
        assert state["net.4.weight"].shape == (4, 128)
        assert all(torch.isfinite(value).all().item() for value in state.values())
        assert all(tuple(group["betas"]) == (0.9, 0.999) for group in optimizer["param_groups"])
        expected_final_lr = spec["lr"] * 0.9 ** (cfg.train.epochs // 1000)
        assert all(abs(group["lr"] - expected_final_lr) < 1e-12
                   for group in optimizer["param_groups"])
        variance_summary = None
        if spec.get("variance_init") is not None or initial_enabled:
            sys.path.insert(0, str(canonical_path.resolve().parents[1]))
            from models.vi_model import ConditionalGaussian

            model_cfg = OmegaConf.create(OmegaConf.to_container(cfg.vi_model, resolve=True))
            model_cfg.device = "cpu"
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(spec["seed"])
                model = ConditionalGaussian(model_cfg)
            if initial_enabled:
                initial_state = torch.load(run_path / "checkpoints/epoch_0/vi_model.pt",
                                           map_location="cpu", weights_only=True)
                assert all(torch.isfinite(value).all().item() for value in initial_state.values())
                model.load_state_dict(initial_state)
            probe = samples["epsilon"]
            with torch.no_grad():
                initial_var = model.getstd(probe).square()
                if spec.get("variance_init") is not None:
                    assert torch.count_nonzero(model.net[-1].weight[2:]).item() == 0
                    assert torch.allclose(initial_var, torch.full_like(initial_var, spec["variance_init"]))
                model.load_state_dict(state)
                final_var = model.getstd(probe).square()
                assert torch.isfinite(final_var).all().item()
                variance_summary = {
                    "initial_per_dimension": initial_var[0].tolist(),
                    "initial_mean": initial_var.mean(0).tolist(),
                    "initial_std_across_epsilon": initial_var.std(0).tolist(),
                    "final_mean": final_var.mean(0).tolist(),
                    "final_std_across_epsilon": final_var.std(0).tolist(),
                    "final_min": final_var.amin(0).tolist(),
                    "final_max": final_var.amax(0).tolist(),
                    "final_variance_head_weight_norm": state["net.4.weight"][2:].norm().item(),
                    "probe_count": len(probe),
                }
        duration = re.search(r"Training completed\. Total time: ([0-9.]+)s", log)
        rows.append({"label": spec["label"], "plots": [panel["plot"] for panel in panels]})
        summaries.append({"key": spec["key"], "label": spec["label"],
                          "seed": spec["seed"], "initial_lr": spec["lr"],
                          "final_lr": expected_final_lr, "source_commit": source_commit,
                          "run_path": str(run_path), "exit_code": exit_code,
                          "elapsed_seconds": float(duration.group(1)) if duration else None,
                          "conditional_variance": variance_summary,
                          "panels": panels})
        shutil.copy2(paths[0], report_root / f"full_config_{spec['key']}.yaml")
    assert all(model == model_configs[0] for model in model_configs)
    seed_comparison = round_name in ("seeds", "constant_variance_init")
    title = ("KSIVI x-shaped: seed evolution, learning rate 0.001" if seed_comparison
             else "KSIVI x-shaped: learning-rate evolution, seed 43")
    variance_init = specs[0].get("variance_init")
    initialization = ""
    if variance_init is not None:
        title = ("KSIVI x-shaped: seed evolution with constant initial variance, LR 0.001"
                 if seed_comparison else
                 "KSIVI x-shaped: learning-rate evolution with constant initial variance, seed 43")
        initialization = f" | initial variance {variance_init:.4f}"
    grid = render_grid(rows, report_root / f"{round_name}_evolution_3x{len(snapshot_steps)}.png",
                       title, initialization, snapshot_steps)
    manifest = {"round": round_name, "steps": list(snapshot_steps),
                "grid_shape": [3, len(snapshot_steps)], "total_steps": cfg.train.epochs,
                "includes_initial_distribution": initial_enabled,
                "reporting_preserves_training_rng": bool(cfg.train.sample.get("preserve_rng", False)),
                "source_commit": source_commit, "vi_model_type": "ConditionalGaussian",
                "batch_size": 128, "hidden_width": 128, "input_noise_dim": 2,
                "annealing_enabled": True, "metric_evaluation_temporarily_disabled": True,
                "annealing_steps": cfg.train.annealing.steps,
                "final_annealing_factor": 0.1 + 0.9 * min(1, cfg.train.epochs / cfg.train.annealing.steps),
                "variance_init": variance_init, "variance_remains_trainable": True,
                "scheduler": {"type": "StepLR", "step_size": 1000, "gamma": 0.9},
                "runs": summaries, **grid}
    (report_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
