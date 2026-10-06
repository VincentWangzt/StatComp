"""Validate saved KSIVI runs and assemble their native plots into a 3x10 grid."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

STEPS = tuple(range(5000, 50001, 5000))


def render_grid(rows: list[dict], output: Path, title: str) -> dict:
    """Use the exact native training plots, without resampling the model."""
    if len(rows) != 3 or any(len(row["plots"]) != 10 for row in rows):
        raise ValueError("The evolution grid requires three rows of ten plots")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 10, figsize=(34, 10.6))
    fig.subplots_adjust(left=0.06, right=0.997, bottom=0.045, top=0.90,
                        hspace=0.01, wspace=0.01)
    fig.suptitle(title, fontsize=23, y=0.995)
    fig.text(0.5, 0.935,
             "ConditionalGaussian | width 128 | noise dimension 2 | batch 128 | annealing enabled",
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
    fig.text(0.5, 0.017,
             "Columns: 5,000 to 50,000 updates, every 5,000. Orange: variational samples; blue: target contours.",
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
    for spec in specs:
        root = Path(spec["results_root"])
        paths = list((root / "KSIVI/x_shaped").glob("*/full_config.yaml"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one run for {spec['key']}, found {paths}")
        run_path = paths[0].parent
        cfg = OmegaConf.load(paths[0])
        expected_train = OmegaConf.to_container(canonical.train, resolve=True)
        expected_train["vi"]["lr"] = spec["lr"]
        expected_train["log"]["metric_log_freq"] = 0
        assert OmegaConf.to_container(cfg.train, resolve=True) == expected_train
        assert cfg.seed == spec["seed"] and cfg.vi_model_type == "ConditionalGaussian"
        assert cfg.vi_model.hidden_dim == 128 and cfg.vi_model.num_layers == 2
        assert cfg.vi_model.epsilon_dim == cfg.vi_model.z_dim == 2
        assert not any(v.get("enabled", False) for v in cfg.metric.values())
        model_configs.append(OmegaConf.to_container(cfg.vi_model, resolve=True))
        log = (run_path / "run.log").read_text()
        assert "Training completed." in log and "NaN or Inf detected" not in log
        exit_code = int((root / "runtime/exit_code.txt").read_text().strip())
        assert exit_code == 0
        panels = []
        for step in STEPS:
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
        checkpoint = run_path / "checkpoints/epoch_50000"
        state = torch.load(checkpoint / "vi_model.pt", map_location="cpu", weights_only=True)
        optimizer = torch.load(checkpoint / "vi_optim.pt", map_location="cpu", weights_only=True)
        assert "var_raw" not in state
        assert state["net.0.weight"].shape == (128, 2)
        assert state["net.2.weight"].shape == (128, 128)
        assert state["net.4.weight"].shape == (4, 128)
        assert all(torch.isfinite(value).all().item() for value in state.values())
        assert all(tuple(group["betas"]) == (0.9, 0.999) for group in optimizer["param_groups"])
        expected_final_lr = spec["lr"] * 0.9 ** 50
        assert all(abs(group["lr"] - expected_final_lr) < 1e-12
                   for group in optimizer["param_groups"])
        duration = re.search(r"Training completed\. Total time: ([0-9.]+)s", log)
        rows.append({"label": spec["label"], "plots": [panel["plot"] for panel in panels]})
        summaries.append({"key": spec["key"], "label": spec["label"],
                          "seed": spec["seed"], "initial_lr": spec["lr"],
                          "final_lr": expected_final_lr, "source_commit": source_commit,
                          "run_path": str(run_path), "exit_code": exit_code,
                          "elapsed_seconds": float(duration.group(1)) if duration else None,
                          "panels": panels})
        shutil.copy2(paths[0], report_root / f"full_config_{spec['key']}.yaml")
    assert all(model == model_configs[0] for model in model_configs)
    title = ("KSIVI x-shaped: seed evolution, learning rate 0.001" if round_name == "seeds"
             else "KSIVI x-shaped: learning-rate evolution, seed 43")
    grid = render_grid(rows, report_root / f"{round_name}_evolution_3x10.png", title)
    manifest = {"round": round_name, "steps": list(STEPS), "grid_shape": [3, 10],
                "source_commit": source_commit, "vi_model_type": "ConditionalGaussian",
                "batch_size": 128, "hidden_width": 128, "input_noise_dim": 2,
                "annealing_enabled": True, "metric_evaluation_temporarily_disabled": True,
                "scheduler": {"type": "StepLR", "step_size": 1000, "gamma": 0.9},
                "runs": summaries, **grid}
    (report_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest
