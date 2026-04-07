# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Project Overview

StatComp is a PyTorch research codebase for Semi-Implicit Variational Inference (SIVI) experiments. The repo supports multiple runner variants (`SIVI`, `UIVI`, `RSIVI`, `AISIVI`, `DSIVI`, `KSIVI`), several toy and Bayesian regression targets, and a campaign-style benchmarking workflow built around generated configs, manifests, per-GPU queues, artifact syncing, and post-hoc reporting.

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

- Host: `ssh -p 44817 root@region-41.seetacloud.com`
- Repo path: `~/ruivi/`
- Branch: typically `vince-dev`
- Conda env: `ruivi`
- OS: Ubuntu 22.04.1 LTS
- Driver: 580.105.08
- PyTorch: 2.9.0+cu126
- Use `tmux` for long-running remote queues
- Push locally, then pull remotely before launching campaign workers

## Core Commands

### Single Experiment Runs

```powershell
.\.venv\Scripts\python.exe src.py --config configs\sivi_banana.yaml
.\.venv\Scripts\python.exe src.py --config configs\sivi_banana.yaml train.epochs=20000 train.vi.lr=0.001
.\.venv\Scripts\python.exe mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000
.\.venv\Scripts\python.exe prepare_data.py
```

### Campaign Generation and Monitoring

```powershell
.\.venv\Scripts\python.exe scripts\generate_grid_benchmark.py --num-gpus 2
.\.venv\Scripts\python.exe scripts\refresh_grid_target_configs.py --target LRwaveform
.\.venv\Scripts\python.exe scripts\fetch_grid_benchmark_artifacts.py
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase official
.\.venv\Scripts\python.exe scripts\summarize_grid_benchmark.py --phase official
.\.venv\Scripts\python.exe scripts\manual_check_grid_benchmark.py
.\.venv\Scripts\python.exe scripts\reevaluate_grid_checkpoints.py
.\.venv\Scripts\python.exe scripts\render_grid_benchmark_report.py
.\.venv\Scripts\python.exe scripts\reset_grid_target_progress.py --target banana --phase official
```

### Remote Queue Workers

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ruivi
python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0
python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0
python scripts/run_grid_queue.py --phase official --queue gpu1 --gpu 1
```

`run_grid_queue.py` launches one sequential worker per queue. It reads the manifest, filters entries by `queue_name`, and injects `cuda_visible_devices=<gpu>` into the `src.py` launch command.

## Output Layout

- Standard run outputs: `results/{runner_type}/{target_type}/{timestamp}/`
- Standard TensorBoard logs: `tb_logs/{runner_type}/{target_type}/{timestamp}/`
- Campaign outputs:
  - `campaigns/{campaign_slug}/manifest.json`
  - `campaigns/{campaign_slug}/manifest.csv`
  - `campaigns/{campaign_slug}/smoke_manifest.json`
  - `campaigns/{campaign_slug}/queue_gpu*.txt`
  - `campaigns/{campaign_slug}/runtime/`
  - `campaigns/{campaign_slug}/generated_reports/`
  - `configs/generated/{campaign_slug}/`

There is no formal test suite, lint, or formatter configuration in the repo.

## Current Campaign System

The current benchmark flow is centered on `scripts/grid_benchmark_common.py` plus the scripts in `scripts/`.

### Current Checked-In Campaign

- Current campaign slug: `grid_benchmark_20260330`
- Campaign directory: `campaigns/grid_benchmark_20260330`
- Generated configs: `configs/generated/grid_benchmark_20260330`
- Top-level manual log: `grid_benchmark_2026-03-30.md`

### Script Roles

- `scripts/grid_benchmark_common.py`
  - Single source of truth for the current campaign slug, target list, variant list, metric budgets, output roots, and helper functions.
- `scripts/generate_grid_benchmark.py`
  - Generates campaign configs and metadata.
  - Writes `manifest.json`, `manifest.csv`, `smoke_manifest.json`, `queue_gpu*.txt`, the campaign `README.md`, and the Markdown monitoring template.
  - Balances runs across queues using estimated cost.
  - Supports `--num-gpus N` when regenerating queue assignments.
- `scripts/run_grid_queue.py`
  - Runs one queue sequentially on one GPU.
  - Records `started`, `completed`, `failed`, and `worker_error` events under `campaigns/.../runtime/`.
  - Stores queue state in `*_current.json` and console logs in `runtime/console_logs/`.
- `scripts/fetch_grid_benchmark_artifacts.py`
  - Pulls compact runtime artifacts, `run.log`, `full_config.yaml`, and extracted TensorBoard summaries from the remote host.
- `scripts/show_grid_status.py`
  - Reports queue-level progress from runtime JSONL logs.
- `scripts/summarize_grid_benchmark.py`
  - Reads extracted TensorBoard CSVs and writes `generated_reports/{phase}_completed_runs.csv` and `.md`.
- `scripts/refresh_grid_target_configs.py`
  - Recomputes metric toggles and budgets for generated configs for one target, and updates manifest metadata.
- `scripts/reset_grid_target_progress.py`
  - Removes runtime records and artifacts for one target so that a subset can be rerun cleanly.
- `scripts/reevaluate_grid_checkpoints.py`
  - Reevaluates completed official checkpoints and writes aggregate reevaluation outputs under `generated_reports/`.
- `scripts/render_grid_benchmark_report.py`
  - Produces the detailed benchmark report from manifests, runtime logs, and completed-run summaries.
- `scripts/manual_check_grid_benchmark.py`
  - Convenience wrapper for a fetch + status + summarize pass, then prints the manual follow-up steps.

## Recommended Workflow

1. Prepare or refresh data if needed with `prepare_data.py`.
2. Regenerate the benchmark campaign with `scripts/generate_grid_benchmark.py --num-gpus N` when targets, variants, budgets, or queue counts change.
3. If a single target's generated configs need metric or budget fixes, use `scripts/refresh_grid_target_configs.py --target ...`.
4. Push local changes, pull on the remote server, activate `ruivi`, and launch one `scripts/run_grid_queue.py` worker per queue inside `tmux`.
5. During the run, use `scripts/fetch_grid_benchmark_artifacts.py`, `scripts/show_grid_status.py`, and `scripts\summarize_grid_benchmark.py` locally to monitor progress.
6. Keep `grid_benchmark_2026-03-30.md` updated with manual check notes, failures, and interventions.
7. If a target or variant needs to be rerun, use `scripts/reset_grid_target_progress.py`, optionally regenerate or refresh configs, and restart the relevant queue worker.
8. After completion, run `scripts/summarize_grid_benchmark.py`, `scripts/reevaluate_grid_checkpoints.py`, and `scripts/render_grid_benchmark_report.py`.

## GPU and Queue Behavior

### What Is Hardcoded

- Many base YAML configs still default `cuda_visible_devices: "0"`.
- The current checked-in campaign artifacts were generated for two queues, so the repo currently contains `queue_gpu0.txt` and `queue_gpu1.txt`.

### What Actually Controls Campaign GPU Placement

- The campaign runner does not trust the YAML default GPU for queue runs.
- `scripts/run_grid_queue.py` launches `src.py` with an override like `cuda_visible_devices=1`, so queue execution is determined by the queue worker command, not by the baked-in YAML default.
- The manifest stores both `queue_name` and `queue_gpu` for each run.

### Flexibility Notes

- The campaign generator now supports `scripts/generate_grid_benchmark.py --num-gpus N`.
- Queue/status/report/reset helpers discover queue names from manifest/runtime data instead of assuming exactly `gpu0` and `gpu1`.
- If the remote environment changes to one GPU or more than two GPUs, regenerate the campaign with the desired queue count and launch one worker per queue.
- For ad hoc single runs outside the campaign system, the config default still matters unless you override `cuda_visible_devices=...` on the command line.

## Architecture

### Entry Point and Config System

`src.py` loads a main OmegaConf YAML, merges CLI dotlist overrides, sets `CUDA_VISIBLE_DEVICES` from `cuda_visible_devices` when `use_cuda=true`, chooses the device, instantiates a runner from `runner/runners.py`, and calls `runner.learn()`.

Configs are typically composed from:

- main experiment configs such as `configs/sivi_banana.yaml`
- target configs in `configs/targets/`
- reverse model configs in `configs/reverse_models/`
- VI model configs in `configs/vi_models/`

Campaign generation writes fully materialized configs to `configs/generated/{campaign_slug}/`.

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

The current grid benchmark campaign uses the 12 targets listed in `scripts/grid_benchmark_common.py`; `8_gaussians` exists in configs but is not part of the current benchmark manifest.
