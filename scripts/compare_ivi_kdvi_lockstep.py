#!/usr/bin/env python
"""Lockstep numerical parity harness: IVI (run_ivi.py) vs KDVI (kdvi.py).

Goal
----
IVI (`IVI-via-mcmc-distillation/run_ivi.py::ImVIDrift`) and KDVI
(`runner/kdvi.py::KDVIRunner`) implement the *same* MALA-distillation
objective. This harness drives BOTH through identical optimization steps,
force-syncing their initial parameters, and diffs every intermediate tensor
(sampled z, MCMC-refined z', bandwidth h, K_xx/K_yx means, loss, per-parameter
gradients, and post-step parameters) to locate the FIRST point of divergence.

Two RNG regimes are supported:
  - mode="reseed": re-seed the global RNG to the SAME value immediately before
    each implementation's step. This feeds both the identical noise stream so
    we isolate *algorithmic* differences (kernel distance, MALA formula, ...).
  - mode="global":  seed once, then let both run continuously. This tests RNG
    *phase* alignment (draw order/shape) on top of algorithmic parity.

This script is the measurement instrument and houses ALL temporary verbose
dumps so the production code stays clean. It is CPU-only and uses a tiny step
count.

Usage
-----
    .venv/bin/python scripts/compare_ivi_kdvi_lockstep.py --steps 5 --mode reseed
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
IVI_DIR = REPO_ROOT / "IVI-via-mcmc-distillation"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

KDVI_CONFIG = REPO_ROOT / "configs" / "kdvi_8_gaussians_small.yaml"

# IVI learn(...) hyper-parameters for 8_gaussians_small (run_ivi.py::main).
IVI_LR = 0.001
IVI_DRIFT_STEPSZ = 0.01          # KDVI tau = 2 * this = 0.02
IVI_BATCH_SIZE = 128
IVI_WARMUP_INTERVAL = 50000      # anneal_coef = min(1, 0.1 + i/50000)
IVI_ANNEAL_FREQ = 5000
IVI_ANNEAL_RATE = 0.75


# --------------------------------------------------------------------------
# Import the project (KDVI-side) modules first so they cache the *project*
# `models` namespace package in sys.modules.
# --------------------------------------------------------------------------
from omegaconf import OmegaConf  # noqa: E402
from runner.runners import Runners  # noqa: E402
from utils.mmd import mmd_ivi_drift  # noqa: E402
from utils.kernels import LaplaceL2Kernel  # noqa: E402
from utils.mcmc_kernels import mala_transition, mala_transition_ivi  # noqa: E402
from utils.annealing import annealing  # noqa: E402


def _load_ivi_module():
    """Import IVI's run_ivi.py in-process.

    run_ivi.py does ``from models import GMM`` which must resolve to the LOCAL
    ``IVI-via-mcmc-distillation/models.py`` (which defines GMM), NOT the
    project ``models/`` namespace package that KDVI uses. We temporarily swap
    ``sys.modules['models']`` to the local module for the duration of the
    import, then restore it. KDVI modules are already imported above, so their
    ``models.*`` references are already bound and unaffected.
    """
    # 1. Load the local IVI models.py under a private name.
    local_models_path = IVI_DIR / "models.py"
    spec_lm = importlib.util.spec_from_file_location(
        "_ivi_local_models", str(local_models_path))
    ivi_local_models = importlib.util.module_from_spec(spec_lm)
    spec_lm.loader.exec_module(ivi_local_models)

    # 2. Temporarily install it as `models` so run_ivi's top-level
    #    `from models import GMM` resolves locally.
    saved_models = sys.modules.get("models")
    saved_target_models = sys.modules.get("models.target_models")
    sys.modules["models"] = ivi_local_models
    try:
        spec_ri = importlib.util.spec_from_file_location(
            "_ivi_run_ivi", str(IVI_DIR / "run_ivi.py"))
        run_ivi = importlib.util.module_from_spec(spec_ri)
        spec_ri.loader.exec_module(run_ivi)
    finally:
        # 3. Restore the project `models` namespace package.
        if saved_models is not None:
            sys.modules["models"] = saved_models
        else:
            sys.modules.pop("models", None)
        if saved_target_models is not None:
            sys.modules["models.target_models"] = saved_target_models
    return run_ivi


# --------------------------------------------------------------------------
# Builders
# --------------------------------------------------------------------------
def build_ivi(run_ivi, target_name: str):
    """Build the IVI ImVIDrift model on the project target (no training)."""
    target_model = run_ivi.build_target(target_name)
    model = run_ivi.ImVIDrift(target_model, hidden_units=256, latent_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=IVI_LR)
    return model, optimizer, run_ivi.maximum_mean_discrepancy


def build_kdvi(target_name: str, seed: int):
    """Build the KDVIRunner from the parity config (no training)."""
    cfg = OmegaConf.load(str(KDVI_CONFIG))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    over = OmegaConf.from_dotlist([
        f"seed={seed}",
        f"target_type={target_name}",
        "train.epochs=1",
        "train.plot.freq=10000000",
        "train.sample.freq=10000000",
        "train.checkpoint.enabled=false",
        "train.log.metric_log_freq=10000000",
        f"output.results_dir=results/ivi_kdvi_lockstep/{timestamp}",
        f"output.tb_dir=tb_logs/ivi_kdvi_lockstep/{timestamp}",
    ])
    cfg = OmegaConf.merge(cfg, over)
    cfg.config_path = str(KDVI_CONFIG)
    cfg.device = "cpu"
    runner = Runners[cfg.runner_type](config=cfg)
    return runner


# --------------------------------------------------------------------------
# Parameter force-sync: IVI Transform.model.* -> KDVI vi_model.net.*
# Both are nn.Sequential([Linear(2,256),ELU,Linear(256,256),ELU,
#                         Linear(256,256),ELU,Linear(256,4)]).
# --------------------------------------------------------------------------
def sync_params_ivi_to_kdvi(ivi_model, kdvi_runner) -> None:
    src_sd = ivi_model.transform.model.state_dict()
    dst = kdvi_runner.vi_model.net
    missing = dst.load_state_dict(src_sd, strict=True)
    # load_state_dict returns NamedTuple(missing_keys, unexpected_keys)
    if missing.missing_keys or missing.unexpected_keys:
        raise RuntimeError(
            f"param sync mismatch: missing={missing.missing_keys} "
            f"unexpected={missing.unexpected_keys}")


def assert_params_equal(ivi_model, kdvi_runner) -> float:
    src = dict(ivi_model.transform.model.state_dict())
    dst = dict(kdvi_runner.vi_model.net.state_dict())
    assert set(src) == set(dst), (set(src) ^ set(dst))
    max_diff = 0.0
    for k in src:
        d = (src[k] - dst[k]).abs().max().item()
        max_diff = max(max_diff, d)
    return max_diff


# --------------------------------------------------------------------------
# Diff utilities
# --------------------------------------------------------------------------
def maxabs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.detach() - b.detach()).abs().max().item()


def banner(msg: str) -> None:
    print("=" * 78)
    print(msg)
    print("=" * 78)


# --------------------------------------------------------------------------
# Single-step drivers (replicating each pipeline using its OWN production
# sub-functions, so the comparison stays faithful).
# --------------------------------------------------------------------------
def ivi_step(model, optimizer, mmd_fn, i: int, capture: dict) -> None:
    """One IVI optimization step (mirrors ImVIDrift.drift_loss + learn loop)."""
    anneal_coef = min(1.0, 0.1 + i * 1.0 / IVI_WARMUP_INTERVAL)
    stepsz = IVI_DRIFT_STEPSZ / anneal_coef

    samp_x = model.sample(IVI_BATCH_SIZE)
    next_x, accept_rate = model.mala(samp_x, stepsz, anneal_coef)
    loss = mmd_fn(next_x.detach(), samp_x)

    optimizer.zero_grad()
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in model.transform.model.named_parameters()}
    optimizer.step()

    capture["anneal"] = anneal_coef
    capture["z"] = samp_x.detach().clone()
    capture["z_refined"] = next_x.detach().clone()
    capture["accept_rate"] = float(accept_rate)
    capture["loss"] = float(loss.detach())
    capture["grads"] = grads
    capture["params_after"] = {
        n: p.detach().clone()
        for n, p in model.transform.model.named_parameters()}


def kdvi_step(runner, optimizer, i: int, capture: dict) -> None:
    """One KDVI optimization step (mirrors KDVIRunner._compute_loss_and_step)."""
    vi = runner.vi_model
    beta = annealing(t=i, warm_up_interval=runner.anneal_steps,
                     anneal=runner.use_annealing, scheme=runner.anneal_scheme)
    # coupled step-size schedule: tau / beta
    current_step_size = runner.mcmc_step_size / max(beta, 1e-6)

    epsilon = vi.sample_epsilon(num=runner.training_batch_size)
    z, _ = vi.forward(epsilon)

    # IVI-exact MALA: pass RAW score/logp (annealing applied internally via
    # anneal_coef=beta) and stepsz = current_step_size/2 (== IVI's
    # drift_stepsz/anneal = 0.01/beta), so noise scale sqrt(2*stepsz) ==
    # sqrt(current_step_size) and drift stepsz*anneal == drift_stepsz.
    raw_score_fn = lambda zz: runner.target_model.score(zz)
    raw_logp_fn = lambda zz: runner.target_model.logp(zz)
    out = mala_transition_ivi(
        z_init=z.detach(), score_fn=raw_score_fn, logp_fn=raw_logp_fn,
        stepsz=current_step_size / 2.0, anneal_coef=beta,
        n_steps=runner.mcmc_steps)
    z_refined = out.z

    loss, info = mmd_ivi_drift(
        x=z, y=z_refined, kernel=runner.mmd_kernel,
        fit_bandwidth_on=runner.fit_bandwidth_on)

    optimizer.zero_grad()
    loss.backward()
    grads = {n: (p.grad.detach().clone() if p.grad is not None else None)
             for n, p in vi.net.named_parameters()}
    optimizer.step()

    capture["anneal"] = beta
    capture["z"] = z.detach().clone()
    capture["z_refined"] = z_refined.detach().clone()
    capture["accept_rate"] = float(out.accept_rate)
    capture["h"] = float(runner.mmd_kernel.h)
    capture["k_xx_mean"] = info["k_xx_mean"]
    capture["k_xy_mean"] = info["k_xy_mean"]
    capture["loss"] = float(loss.detach())
    capture["grads"] = grads
    capture["params_after"] = {
        n: p.detach().clone() for n, p in vi.net.named_parameters()}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", default="8_gaussians_small")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--mode", choices=["reseed", "global"], default="reseed")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="Max-abs tolerance for declaring a phase 'matched'.")
    args = ap.parse_args()

    # Determinism: match IVI (ImVIDrift.__init__ calls set_num_threads(1)).
    torch.set_num_threads(1)

    banner(f"BUILD  target={args.target} seed={args.seed} mode={args.mode}")

    # --- Build KDVI first (caches project `models`), then IVI. ---
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    kdvi = build_kdvi(args.target, args.seed)
    kdvi_opt = kdvi.optimizer_vi

    run_ivi = _load_ivi_module()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    ivi_model, ivi_opt, ivi_mmd = build_ivi(run_ivi, args.target)

    # --- Force-sync initial parameters (IVI -> KDVI) and assert identical. ---
    sync_params_ivi_to_kdvi(ivi_model, kdvi)
    init_diff = assert_params_equal(ivi_model, kdvi)
    print(f"[init] max-abs param diff after force-sync: {init_diff:.3e}")
    if init_diff != 0.0:
        print("[init] WARNING: initial parameters not byte-identical.")
    else:
        print("[init] OK: initial parameters byte-identical.")

    # --- Lockstep loop ---
    first_div = None
    for i in range(1, args.steps + 1):
        ivi_cap: dict = {}
        kdvi_cap: dict = {}

        if args.mode == "reseed":
            torch.manual_seed(1000 + i)
            ivi_step(ivi_model, ivi_opt, ivi_mmd, i, ivi_cap)
            torch.manual_seed(1000 + i)
            kdvi_step(kdvi, kdvi_opt, i, kdvi_cap)
        else:  # global
            ivi_step(ivi_model, ivi_opt, ivi_mmd, i, ivi_cap)
            kdvi_step(kdvi, kdvi_opt, i, kdvi_cap)

        d_anneal = abs(ivi_cap["anneal"] - kdvi_cap["anneal"])
        d_z = maxabs(ivi_cap["z"], kdvi_cap["z"])
        d_zr = maxabs(ivi_cap["z_refined"], kdvi_cap["z_refined"])
        d_acc = abs(ivi_cap["accept_rate"] - kdvi_cap["accept_rate"])
        d_loss = abs(ivi_cap["loss"] - kdvi_cap["loss"])
        # grads
        d_grad = 0.0
        for n in ivi_cap["grads"]:
            gi, gk = ivi_cap["grads"][n], kdvi_cap["grads"][n]
            if gi is None or gk is None:
                continue
            d_grad = max(d_grad, (gi - gk).abs().max().item())
        # params after step
        d_par = 0.0
        for n in ivi_cap["params_after"]:
            d_par = max(d_par, (ivi_cap["params_after"][n]
                                - kdvi_cap["params_after"][n]).abs().max().item())

        print(f"\n--- step {i} (anneal ivi={ivi_cap['anneal']:.6f} "
              f"kdvi={kdvi_cap['anneal']:.6f}) ---")
        print(f"  d_anneal   = {d_anneal:.3e}")
        print(f"  d_z        = {d_z:.3e}")
        print(f"  d_z_refined= {d_zr:.3e}  (ivi acc={ivi_cap['accept_rate']:.4f}"
              f" kdvi acc={kdvi_cap['accept_rate']:.4f}, d_acc={d_acc:.3e})")
        print(f"  d_loss     = {d_loss:.3e}  "
              f"(ivi={ivi_cap['loss']:.6e} kdvi={kdvi_cap['loss']:.6e})")
        print(f"  d_grad_max = {d_grad:.3e}")
        print(f"  d_param_max= {d_par:.3e}")

        if first_div is None:
            for name, val in [("z", d_z), ("z_refined", d_zr),
                              ("loss", d_loss), ("grad", d_grad),
                              ("param", d_par)]:
                if val > args.atol:
                    first_div = (i, name, val)
                    break

    banner("SUMMARY")
    if first_div is None:
        print(f"No divergence above atol={args.atol:.1e} across "
              f"{args.steps} steps. IVI and KDVI are in lockstep.")
    else:
        i, name, val = first_div
        print(f"FIRST DIVERGENCE: step {i}, phase '{name}', max-abs={val:.3e}")


if __name__ == "__main__":
    main()
