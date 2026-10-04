#!/usr/bin/env python
"""Capture a short historical-branch training run for numerical comparison.

Run this committed harness against each checkout in a separate process. It
executes that checkout's src.py, keeps the method's training configuration,
and shortens only the run/evaluation budgets. TensorBoard is replaced with an
in-memory scalar collector so verification artifacts stay under results/.
This is a parity check, not a reproduction of the full paper experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import runpy
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--numpy-seed", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--warmup-epochs", type=int, default=20)
    parser.add_argument("--metric-frequency", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=2000)
    parser.add_argument("--evaluation-samples", type=int, default=256)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--jacobian", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    args = parser.parse_args()
    repo = args.repo.resolve()
    output = args.output.resolve()
    if "results" not in output.parts:
        parser.error("--output must be inside a results directory")
    if args.epochs < 1 or args.metric_frequency < 1:
        parser.error("run and metric budgets must be positive")
    output.mkdir(parents=True, exist_ok=True)
    os.chdir(repo)
    sys.path.insert(0, str(repo))

    import numpy as np
    import torch
    import torch.utils.tensorboard as tensorboard
    from omegaconf import OmegaConf

    if args.numpy_seed is not None:
        np.random.seed(args.numpy_seed)
        random.seed(args.numpy_seed)

    rows: list[dict] = []

    class ScalarCollector:
        def __init__(self, *unused_args, **unused_kwargs):
            pass

        def add_scalar(self, tag, scalar_value, global_step=None, **kwargs):
            value = scalar_value.item() if hasattr(scalar_value, "item") else scalar_value
            rows.append({"tag": str(tag), "step": global_step, "value": float(value)})

        def add_text(self, *unused_args, **unused_kwargs):
            pass

        def flush(self):
            pass

        def close(self):
            pass

    tensorboard.SummaryWriter = ScalarCollector
    overrides = [
        f"seed={args.seed}",
        f"train.epochs={args.epochs}",
        f"reverse_model.warmup.epochs={args.warmup_epochs}",
        f"train.log.metric_log_freq={args.metric_frequency}",
        f"train.log.loss_log_freq={args.metric_frequency}",
        f"train.checkpoint.freq={args.metric_frequency}",
        f"train.sample.freq={args.metric_frequency}",
        f"train.sample.num={args.sample_count}",
        "train.plot.freq=1000000000",
        f"metric.kl_ite.num_samples={args.evaluation_samples}",
        f"metric.w2.num_samples={args.evaluation_samples}",
        "metric.w2.num_projections=32",
        f"metric.elbo.num_z_samples={args.evaluation_samples}",
        "metric.elbo.batch_size=64",
        "metric.elbo.num_batches=2",
        f"output.results_dir={output.as_posix()}/runs",
        f"output.tb_dir={output.as_posix()}/scalar_collector",
    ]
    if args.baseline:
        overrides.append(f"target.baseline_path={args.baseline.resolve().as_posix()}")
    if args.jacobian:
        overrides += ["metric.jacobian_spectral.enabled=true", "metric.jacobian_spectral.num_samples=2"]
    overrides += args.overrides
    sys.argv = [str(repo / "src.py"), "--config", args.config, *overrides]
    namespace = runpy.run_path(str(repo / "src.py"), run_name="__main__")
    runner = namespace["runner"]

    state = {"vi_model": {k: v.detach().cpu().clone() for k, v in runner.vi_model.state_dict().items()}}
    reverse = getattr(runner, "reverse_model", None)
    if reverse is not None and hasattr(reverse, "state_dict"):
        state["reverse_model"] = {k: v.detach().cpu().clone() for k, v in reverse.state_dict().items()}
    torch.save(state, output / "state.pt")

    def digest(tensor) -> str:
        return hashlib.sha256(tensor.cpu().numpy().tobytes()).hexdigest()

    report = {
        "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "config": args.config,
        "seed": args.seed,
        "numpy_seed": args.numpy_seed,
        "epochs": args.epochs,
        "warmup_epochs_override": args.warmup_epochs,
        "torch_version": torch.__version__,
        "device": str(runner.device),
        "run_path": str(runner.save_path),
        "torch_cpu_rng_digest": digest(torch.get_rng_state()),
        "torch_cuda_rng_digests": [digest(s) for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
        "resolved_config": OmegaConf.to_container(runner.config, resolve=True),
        "scalars": rows,
    }
    (output / "capture.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "commit": report["commit"], "epochs": args.epochs, "scalars": len(rows)}))


if __name__ == "__main__":
    main()
