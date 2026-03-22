# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

StatComp is a PyTorch research codebase implementing multiple Semi-Implicit Variational Inference (SIVI) algorithms. It provides a modular framework for comparing five SIVI variants (SIVI, UIVI, RSIVI, AISIVI, DSIVI) across various target distributions with comprehensive experiment management and evaluation.

## Commands

### Environment Setup
```bash
conda create -n stat_comp python=3.14
conda activate stat_comp
pip install -r requirements.txt
```

### Running Experiments
```bash
# Run an experiment with a YAML config
python src.py --config configs/sivi_banana.yaml

# Override config values via CLI
python src.py --config configs/sivi_banana.yaml train.epochs=20000 train.vi.lr=0.001

# HMC baseline
python mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000

# Visualization pipeline
bash visualization.sh
```

### Output Locations
- Results: `results/{runner_type}/{target_type}/{timestamp}/`
- TensorBoard logs: `tb_logs/{runner_type}/{target_type}/{timestamp}/`

There is no formal test suite, linting, or formatting configuration.

## Architecture

### Entry Point & Config System

`src.py` parses YAML configs via OmegaConf, instantiates the appropriate runner from a registry (`runner/runners.py`), and calls `runner.learn()`. Configs are hierarchical: a main config (e.g. `configs/sivi_banana.yaml`) references sub-configs for target, reverse model, and VI model from `configs/targets/`, `configs/reverse_models/`, and `configs/vi_models/`.

### Three Core Abstractions (in `models/`)

1. **Target Distribution** (`target_models.py`): Defines `logp(X)` (unnormalized log-density) and `score(X)` (∇_X log p). Implementations: Banana, Multimodal, X-shaped, StudentT, BNN, Logistic Regression, Langevin post-processing. Data-dependent targets (BNN, LR) are wrapped by `DataBoundTarget` (`models/data_bound_target.py`) to present the standard `logp(X)`/`score(X)` interface with bound dataset.
2. **VI Model** (`vi_model.py`): The variational family q_φ(z|ε). Provides `sample_epsilon()`, `forward(ε)` → z + negative score, and `sampling()`.
3. **Reverse Model** (`reverse_model.py`): The reverse conditional q_ψ(ε|z). Provides `log_prob(ε,z)`, `sample(z)`, `fit(ε,z)`. Implementations include GMM, Conditional RealNVP, and denoising models.

### Runner System (in `runner/`)

`base_runner.py` encapsulates the full training loop: sampling, loss computation, metric evaluation (KL via ITE, Wasserstein-2, ELBO), TensorBoard logging, checkpointing, and visualization. `base_reverse_runner.py` extends it for methods that train a reverse model.

Five variant runners inherit from these bases:

| Runner | Reverse Model | Description |
|--------|--------------|-------------|
| `sivi.py` | None (prior) | Standard semi-implicit VI |
| `uivi.py` | HMC | Unbiased VI via HMC targeting q_φ(ε\|z)q(ε) |
| `rsivi.py` | Learned (GMM/RealNVP) | Reverse model learned via gradient |
| `aisivi.py` | Learned + Annealed | Annealed importance sampling |
| `dsivi.py` | Diffusion model | Diffusion-based reverse model |

Runner types are registered in `runner/runners.py` and selected via `runner_type` in config.

### Supporting Modules

- **`ite/`**: Information-Theoretic Estimators toolbox for KL divergence estimation during evaluation.
- **`utils/`**: HMC sampler (`mcmc.py`), sliced Wasserstein distance (`metrics.py`), batch Jacobian (`batch_jacobian.py`), annealing schedules (`annealing.py`), EMA (`ema.py`), structured logging (`logging.py`), dataset loaders (`datasets.py`).
- **`visualization/`**: Post-hoc visualization pipeline for benchmark comparison (KL curves, scatter plots, density estimation). Expects standardized PyTorch sample files.

### Data-Dependent Targets

Data-dependent targets (`LRwaveform`, `Bnn_boston`) require a dataset bound to their `logp`/`score` calls. This is handled by the `DataBoundTarget` wrapper in `models/data_bound_target.py`, which the runner constructs automatically via `_build_target_model()`.

**Data preparation** (one-time, before first use):
```bash
python prepare_data.py
```
This generates pre-processed `.pt` files under `data/waveform/` and `data/boston/`.

**Available targets and their configs:**

| Target | Type | z_dim | Config | Baseline |
|--------|------|-------|--------|----------|
| `banana` | Toy 2D | 2 | `configs/targets/banana.yaml` | `baselines/hmc/banana.pt` |
| `multimodal` | Toy 2D | 2 | `configs/targets/multimodal.yaml` | `baselines/hmc/multimodal.pt` |
| `x_shaped` | Toy 2D | 2 | `configs/targets/x_shaped.yaml` | `baselines/hmc/x_shaped.pt` |
| `student_uc` | Toy 2D | 2 | `configs/targets/student_uc.yaml` | `baselines/hmc/student_uc.pt` |
| `Langevin_post` | High-dim | 100 | `configs/targets/Langevin_post.yaml` | `baselines/hmc/Langevin_post.pt` |
| `LRwaveform` | Data-dep | 22 | `configs/targets/LRwaveform.yaml` | None (ELBO only) |
| `Bnn_boston` | Data-dep | 751 | `configs/targets/Bnn_boston.yaml` | None (ELBO only) |

## Current Debug Environment

- **Python**: 3.14.2 (venv managed by uv 0.10.12, in `.venv/`)
- **PyTorch**: 2.9.0+cpu (CUDA not available in this environment)
- **OS**: Linux 5.4.241-1-tlinux4-0017.16 x86_64
- **GPU**: None detected (CPU-only)
