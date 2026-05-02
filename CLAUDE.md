# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StatComp is a PyTorch research codebase for Semi-Implicit Variational Inference (SIVI) experiments. The repo supports multiple runner variants (`SIVI`, `UIVI`, `RSIVI`, `AISIVI`, `DSIVI`, `KSIVI`) and several toy, high-dimensional, and Bayesian regression targets.

## Commands

### Running Experiments
```bash
# Run an experiment with a YAML config
python src.py --config configs/sivi_banana.yaml

# Override config values via CLI
python src.py --config configs/sivi_banana.yaml train.epochs=20000 train.vi.lr=0.001

# HMC baseline
python mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000

# One-time data preparation (before first use of data-dependent targets)
python prepare_data.py
```

### Output Layout
- Results: `results/{runner_type}/{target_type}/{timestamp}/`
- TensorBoard logs: `tb_logs/{runner_type}/{target_type}/{timestamp}/`
- Campaign reports: `campaigns/*/generated_reports/`

Keep all experiment outputs inside the existing `results/` folder and all TensorBoard logs inside `tb_logs/`. Do not spill generated files into unrelated directories; create a new subfolder under the corresponding existing folder when needed.

There is no formal test suite, linting, or formatting configuration.

## Architecture

### Entry Point & Config System

`src.py` loads a main OmegaConf YAML, merges CLI dotlist overrides, sets `CUDA_VISIBLE_DEVICES` from `cuda_visible_devices` when `use_cuda=true`, chooses the device, instantiates a runner from `runner/runners.py`, and calls `runner.learn()`.

Configs are composed from:
- Main experiment configs: `configs/{runner}_{target}.yaml`
- Target configs: `configs/targets/`
- Reverse model configs: `configs/reverse_models/`
- VI model configs: `configs/vi_models/`

### Core Abstractions (in `models/`)

1. **Target Distribution** (`target_models.py`): Defines `logp(X)` (unnormalized log-density) and `score(X)` (nabla_X log p). Includes toy 2-D targets, high-dimensional targets, and data-dependent BNN/LR targets. Data-dependent targets are wrapped by `DataBoundTarget` (`models/data_bound_target.py`) to present the standard `logp(X)`/`score(X)` interface with bound dataset.
2. **VI Model** (`vi_model.py`): The variational family q_phi(z|epsilon). Provides `sample_epsilon()`, `forward(epsilon)` -> z + negative score, and `sampling()`.
3. **Reverse Model** (`reverse_model.py`): The reverse conditional q_psi(epsilon|z). Provides `log_prob(epsilon,z)`, `sample(z)`, `fit(epsilon,z)`. Implementations include GMM, Conditional RealNVP, and denoising models.

### Runner System (in `runner/`)

`base_runner.py` encapsulates the full training loop: sampling, loss computation, metric evaluation (KL via ITE, Wasserstein-2, ELBO), TensorBoard logging, checkpointing, and visualization. `base_reverse_runner.py` extends it for methods that train a reverse model.

Six variant runners inherit from these bases:

| Runner | File | Reverse Model | Description |
|--------|------|--------------|-------------|
| `SIVI` | `sivi.py` | None (prior) | Standard semi-implicit VI |
| `UIVI` | `uivi.py` | HMC | Unbiased VI via HMC targeting q_phi(epsilon\|z)q(epsilon) |
| `RSIVI` | `rsivi.py` | Learned (GMM/RealNVP) | Reverse model learned via gradient |
| `AISIVI` | `aisivi.py` | Learned + Annealed | Annealed importance sampling |
| `DSIVI` | `dsivi.py` | Diffusion model | Diffusion-based reverse model |
| `KSIVI` | `ksivi.py` | None | Kernel Stein discrepancy minimization (no reverse model needed) |

Runner types are registered in `runner/runners.py` and selected via `runner_type` in config.

### Supporting Modules

- **`utils/`**: HMC sampler (`mcmc.py`), sliced Wasserstein distance (`metrics.py`), kernel functions for KSIVI (`kernels.py`), batch Jacobian (`batch_jacobian.py`), annealing schedules (`annealing.py`), EMA (`ema.py`), structured logging (`logging.py`), dataset loaders (`datasets.py`), density estimation (`density_estimation.py`), expected log-marginal (`expected_log_marginal.py`), TensorBoard extraction helpers (`extract_tensorboard_run.py`).
- **`ite/`**: Information-Theoretic Estimators for KL divergence estimation during evaluation.
- **`visualization/`**: Post-hoc visualization pipeline for benchmark comparison (KL curves, scatter plots, density estimation). Expects standardized PyTorch sample files.

### Data-Dependent Targets

Data-dependent targets are wrapped by `models/data_bound_target.py` so they still expose the standard `logp(X)` and `score(X)` interface after dataset binding.

Run once before first use:
```bash
python prepare_data.py
```

This generates pre-processed `.pt` files under `data/` (subdirectories: `waveform/`, `boston/`, `concrete/`, `power/`, `protein/`, `winered/`, `yacht/`).

### Available Targets

| Target | Type | z_dim | Config | Baseline |
|--------|------|-------|--------|----------|
| `banana` | Toy 2D | 2 | `configs/targets/banana.yaml` | `baselines/exact/banana_exact_100k_20260408.pt` |
| `multimodal` | Toy 2D | 2 | `configs/targets/multimodal.yaml` | `baselines/exact/multimodal_exact_100k_20260408.pt` |
| `x_shaped` | Toy 2D | 2 | `configs/targets/x_shaped.yaml` | `baselines/exact/x_shaped_exact_100k_20260408.pt` |
| `student_uc` | Toy 2D | 2 | `configs/targets/student_uc.yaml` | `baselines/exact/student_uc_exact_100k_20260408.pt` |
| `8_gaussians` | Toy 2D | 2 | `configs/targets/8_gaussians.yaml` | `baselines/exact/8_gaussians_exact_100k_20260408.pt` |
| `Langevin_post` | High-dim | 100 | `configs/targets/Langevin_post.yaml` | `baselines/hmc/Langevin_post.pt` |
| `LRwaveform` | Data-dep | 22 | `configs/targets/LRwaveform.yaml` | None (ELBO only) |
| `Bnn_boston` | Data-dep | 751 | `configs/targets/Bnn_boston.yaml` | None (ELBO only) |
| `Bnn_concrete` | Data-dep | 501 | `configs/targets/Bnn_concrete.yaml` | None (ELBO only) |
| `Bnn_power` | Data-dep | 301 | `configs/targets/Bnn_power.yaml` | None (ELBO only) |
| `Bnn_protein` | Data-dep | 551 | `configs/targets/Bnn_protein.yaml` | None (ELBO only) |
| `Bnn_winered` | Data-dep | 651 | `configs/targets/Bnn_winered.yaml` | None (ELBO only) |
| `Bnn_yacht` | Data-dep | 401 | `configs/targets/Bnn_yacht.yaml` | None (ELBO only) |

## Environments

### Local (CPU-only, for development/testing)
- **Python**: 3.14.2 (venv managed by uv 0.10.12, in `.venv/`)
- **PyTorch**: 2.9.0+cpu
- **OS**: Linux x86_64
- **IMPORTANT**: Always run via `.venv/bin/python` (or activate with `source .venv/bin/activate`). System Python does not have project dependencies installed.
- **Note**: `triton==3.5.0` is pinned in `requirements.txt` but may need to be omitted on platforms without available wheels (e.g., Windows).

### Remote GPU Server (for training)
- **Host**: `ssh -p 48236 root@connect.nmb1.seetacloud.com`
- **Code path**: `~/ruivi/` (same repo, `vince-dev` branch)
- **Conda env**: `ruivi` (PyTorch 2.9.0+cu126)
- **OS**: Ubuntu 22.04.1 LTS
- **Workflow**:
  - Push locally, pull on remote. Use `tmux` for long runs.
  - Do not copy code/config/script files directly to the remote server. All code changes go through git.
  - Generated report artifacts (figures, tables) should be produced on the remote from pushed code, then committed and pushed from the remote repo so the local checkout pulls them back through git.
  - Keep direct changes on the remote minimal: running experiments, inspecting logs, generating report artifacts from committed code, committing those artifacts, and managing runtime artifacts under `results/`, `tb_logs/`, or `campaigns/*/runtime`.
