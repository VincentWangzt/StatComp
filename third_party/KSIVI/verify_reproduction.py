"""Audit saved reproduction samples and gradients at the trained checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import reproduce_x_shaped as experiment
import numpy as np
import torch


def verify(root):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    upstream = json.loads((experiment.HERE / "UPSTREAM.json").read_text())
    original_hashes = {item["path"]: item["sha256"] for item in upstream["files"]}
    for path in ["configs/x_shaped.yml", "utils/kernels.py", "models/target_models.py"]:
        assert hashlib.sha256((experiment.HERE / path).read_bytes()).hexdigest() == original_hashes[path]
    current_hashes = experiment.source_info()["source_sha256"]
    records = []
    for seed in [42, 43, 44]:
        for condition in experiment.TREATMENTS:
            directory = root / f"{condition}_s{seed}"
            result = json.loads((directory / "result.json").read_text())
            assert result["status"] == "completed" and result["steps"] == 50000
            assert result["specification"]["source"]["source_sha256"] == current_hashes
            runner, config = experiment.make_runner(condition, seed)
            assert result["specification"]["config"] == config
            checkpoint = torch.load(directory / "checkpoint.pt", map_location="cpu", weights_only=True)
            assert checkpoint["step"] == 50000
            runner.SemiVInet.load_state_dict(checkpoint["model"])
            x, metrics = experiment.sample_and_diagnose(runner, seed)
            np.testing.assert_array_equal(x.numpy(), np.load(directory / "final_samples.npy"))
            for key, value in metrics.items():
                np.testing.assert_allclose(value, result["metrics"][key], rtol=1e-12, atol=1e-12)

            # Compare all parameter gradients to an independently written loss
            # expression on the next two batches at the trained checkpoint.
            torch.random.set_rng_state(checkpoint["rng_state"])
            z = torch.randn(500, runner.trainpara.z_dim)
            x, b = runner.SemiVInet(z)
            z_aux = torch.randn(500, runner.trainpara.z_dim)
            y, c = runner.SemiVInet(z_aux)
            parameters = list(runner.SemiVInet.parameters())
            sx, sy = runner.target_model.score(x), runner.target_model.score(y)
            stein = condition == "stein_cond"
            kernel, h = experiment.gaussian_kernel(x, y, get_width=True, detach=stein)
            if stein:
                delta = x[:, None, :] - y[None, :, :]
                inner = (sx @ sy.T + ((sx[:, None, :] - sy[None, :, :]) * delta).sum(-1) / h**2
                         + 2 / h**2 - delta.square().sum(-1) / h**4)
                reference = (inner * kernel).mean()
            else:
                reference = (((sx + b) @ (sy + c).T) * kernel).mean()
            reference_grad = torch.cat([g.flatten() for g in torch.autograd.grad(
                reference, parameters, retain_graph=True)])
            actual, _ = runner.loss_from_samples(x, b, y, c)
            actual_grad = torch.cat([g.flatten() for g in torch.autograd.grad(actual, parameters)])
            difference = actual_grad - reference_grad
            relative = float(difference.norm() / reference_grad.norm().clamp_min(1e-12))
            assert relative < 1e-3, (condition, seed, relative)
            torch.testing.assert_close(actual, reference, rtol=5e-5, atol=3e-6)
            if not stein:
                torch.testing.assert_close(actual_grad, reference_grad, rtol=0, atol=0)
            record = {"condition": condition, "seed": seed,
                      "samples_match_saved_checkpoint_exactly": True,
                      "all_saved_metrics_recomputed": True,
                      "gradient_relative_l2_error": relative,
                      "gradient_max_absolute_error": float(difference.abs().max()),
                      "reference_loss": float(reference.detach()),
                      "training_loss": float(actual.detach())}
            records.append(record)
            print(json.dumps(record), flush=True)

    # A scale reference for finite-sample evaluation, not a fitted-method score.
    reference = experiment.target_samples(10000)
    sampling_baselines = {
        str(seed): experiment.sliced_w2(experiment.target_samples(10000, seed=seed), reference)
        for seed in [42, 43, 44]
    }
    output = {"status": "passed", "canonical_config_target_and_kernel_unchanged_from_upstream": True,
              "source_sha256": current_hashes, "runs": records,
              "independent_exact_target_sample_sw2": sampling_baselines,
              "note": "The exact-target sample distances indicate evaluation noise; they are not an error correction."}
    experiment.write_json(root / "report/verification.json", output)
    print(json.dumps({"status": "passed", "runs": len(records),
                      "independent_exact_target_sample_sw2": sampling_baselines}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=experiment.DEFAULT_ROOT)
    args = parser.parse_args()
    verify(args.root.resolve())
