# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

StatComp is a PyTorch research codebase for Semi-Implicit Variational Inference (SIVI) experiments. The repo supports multiple runner variants (`SIVI`, `UIVI`, `RSIVI`, `AISIVI`, `DSIVI`, `KSIVI`) and several toy and Bayesian regression targets.

## Environment

### Local Windows Environment

- Python: 3.14.3
- Virtual environment: `.venv\` managed with `uv`
- PyTorch: 2.9.0+cpu
- OS: Windows 10 x86_64
- Always use `.\.venv\Scripts\python.exe` or activate `.\.venv\Scripts\activate`
- `triton==3.5.0` is pinned in `requirements.txt` but is intentionally omitted locally because Windows wheels are unavailable

### Local Setup

```powershell
uv venv .venv --python 3.14 --seed
.\.venv\Scripts\activate
$pkgs = Get-Content requirements.txt | Where-Object { $_ -and ($_ -notmatch '^#') -and ($_ -notmatch '^--extra-index-url') -and ($_ -notmatch '^triton==') }
uv pip install --python .\.venv\Scripts\python.exe --index-url https://pypi.org/simple --extra-index-url https://download.pytorch.org/whl/cpu --index-strategy unsafe-best-match $pkgs
```

### Remote GPU Environment

- Host: `ssh -p 18321 root@connect.bjb2.seetacloud.com`
- Repo path: `~/ruivi/`
- Branch: typically `vince-dev`
- Conda env: `ruivi`
- OS: Ubuntu 22.04.1 LTS
- Driver: 580.105.08
- PyTorch: 2.9.0+cu126
- Use `tmux` for long-running remote experiments
- Required remote workflow for code/config/script changes: make the change locally, test locally when feasible, commit locally, push the branch, then sync the remote repo with git (`git pull`/`git fetch` + checkout) before running remote jobs. Do not copy code/config/script files directly to the remote server as the primary workflow.
- Keep direct changes made on the remote server minimal. Remote-only actions should be limited to running experiments, inspecting logs/results, and managing runtime artifacts under `results/`, `tb_logs/`, or `campaigns/*/runtime`.

## Core Commands

### Single Experiment Runs

```powershell
.\.venv\Scripts\python.exe src.py --config configs\sivi_banana.yaml
.\.venv\Scripts\python.exe src.py --config configs\sivi_banana.yaml train.epochs=20000 train.vi.lr=0.001
.\.venv\Scripts\python.exe mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000
.\.venv\Scripts\python.exe prepare_data.py
```

## Output Layout

- Standard run outputs: `results/{runner_type}/{target_type}/{timestamp}/`
- Standard TensorBoard logs: `tb_logs/{runner_type}/{target_type}/{timestamp}/`
- Keep all experiment outputs inside the existing `results/` folder and all TensorBoard logs inside the existing `tb_logs/` folder. Do not spill generated files into unrelated directories, especially on the remote server; create a new subfolder under the corresponding existing folder when needed.

There is no formal test suite, lint, or formatter configuration in the repo.

## Architecture

### Entry Point and Config System

`src.py` loads a main OmegaConf YAML, merges CLI dotlist overrides, sets `CUDA_VISIBLE_DEVICES` from `cuda_visible_devices` when `use_cuda=true`, chooses the device, instantiates a runner from `runner/runners.py`, and calls `runner.learn()`.

Configs are typically composed from:

- main experiment configs such as `configs/sivi_banana.yaml`
- target configs in `configs/targets/`
- reverse model configs in `configs/reverse_models/`
- VI model configs in `configs/vi_models/`

### Core Abstractions

1. Target distribution in `models/target_models.py`
   - Defines `logp(X)` and `score(X)`.
   - Includes toy targets and BNN/LR-style data-dependent targets.
2. Variational family in `models/vi_model.py`
   - Defines epsilon sampling and forward/sampling methods for `q_phi(z | epsilon)`.
3. Reverse model in `models/reverse_model.py`
   - Defines `log_prob(epsilon, z)`, `sample(z)`, and `fit(epsilon, z)`.

### Runner System

- `runner/base_runner.py`
  - Shared training loop, metrics, logging, checkpointing, and visualization.
- `runner/base_reverse_runner.py`
  - Adds reverse-model training behavior.
- Registered runners in `runner/runners.py`
  - `SIVI`
  - `UIVI`
  - `RSIVI`
  - `AISIVI`
  - `DSIVI`
  - `KSIVI`

### Supporting Modules

- `utils/`
  - HMC, metrics, annealing, EMA, logging, datasets, TensorBoard extraction helpers.
- `ite/`
  - Information-theoretic estimators used for KL-style evaluation.
- `visualization/`
  - Benchmark visualization utilities.

## Data-Dependent Targets

Data-dependent targets are wrapped by `models/data_bound_target.py` so they still expose the standard `logp(X)` and `score(X)` interface after dataset binding.

Run once before first use:

```powershell
.\.venv\Scripts\python.exe prepare_data.py
```

Important prepared datasets:

- `LRwaveform`
- `Bnn_boston`
- `Bnn_concrete`
- `Bnn_power`
- `Bnn_protein`
- `Bnn_winered`
- `Bnn_yacht`

## Target Notes

Target configs currently present in `configs/targets/` include:

- `banana`
- `multimodal`
- `x_shaped`
- `student_uc`
- `8_gaussians`
- `Langevin_post`
- `LRwaveform`
- `Bnn_boston`
- `Bnn_concrete`
- `Bnn_power`
- `Bnn_protein`
- `Bnn_winered`
- `Bnn_yacht`
