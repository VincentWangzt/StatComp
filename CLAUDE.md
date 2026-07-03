# CLAUDE.md

PyTorch research codebase for Semi-Implicit Variational Inference (SIVI) experiments. See README.md for full architecture docs.

## Commands

```bash
python src.py --config configs/sivi_banana.yaml                        # run experiment
python src.py --config configs/sivi_banana.yaml train.epochs=20000     # with CLI overrides
python mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000  # HMC baseline
python prepare_data.py                                                 # one-time data prep (before data-dependent targets)
```

## Config System

Configs are composed from 4 layers — main experiment configs pull in sub-configs:
- Main: `configs/{runner}_{target}.yaml`
- Targets: `configs/targets/`
- Reverse models: `configs/reverse_models/`
- VI models: `configs/vi_models/`

## Output Layout

Keep all experiment outputs inside these existing folders — create subfolders when needed but don't spill generated files elsewhere:
- Results: `results/{runner_type}/{target_type}/{timestamp}/`
- W&B + live metrics: inside `results/{runner_type}/{target_type}/{timestamp}/`

## Environment (Local)

- Python 3.14.2, venv in `.venv/` (managed by uv)
- Always run via `.venv/bin/python` — system Python lacks project dependencies
- `triton==3.5.0` is pinned but may need omitting on platforms without wheels

## Environment (Remote GPU)

- Host: `ssh -p 48236 root@connect.nmb1.seetacloud.com`
- Code path: `~/ruivi/` (same repo, `distillation` branch)
- Conda env: `ruivi` (PyTorch 2.9.0+cu126, Ubuntu 22.04)
- Use `tmux` for long-running experiments

### Remote Workflow Rules

- Push locally, pull on remote — don't copy code/config/script files directly, all code changes go through git.
- Generated report artifacts (figures, tables): produce on remote from committed code, commit there, pull locally.
- Keep direct remote changes minimal: running experiments, inspecting logs, generating artifacts, committing those artifacts, managing `results/`, `tb_logs/`, or `campaigns/

## Notes

- No test suite, linting, or formatting configuration exists.
- `src.py` sets `CUDA_VISIBLE_DEVICES` from config's `cuda_visible_devices` when `use_cuda=true`.
- Explicitly instruct the exploration subagents to place their output artifacts in `/tmp` instead of the current working directory.
