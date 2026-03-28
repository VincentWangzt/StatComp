# KSIVI Baseline Log

This document records fresh baseline experiments for KSIVI in:

- `D:\PKU\Programming\StatComp\project-ksivi-debug`
- `D:\PKU\Programming\StatComp\KSIVI`

Rules for this log:

- Record only observations from fresh runs started during this debugging campaign.
- Record the exact command, redirected log path, TensorBoard/result path, and current git commit when relevant.
- Do not use `experiment_results.md` as evidence here.

## Run Index

| ID | Repo | Target | Config/Script | Status | Output Log | Notes |
|---|---|---|---|---|---|---|
| B001 | current | banana | `configs/ksivi_banana.yaml` with `train.epochs=2000`, metrics every 500, plots/samples/checkpoints disabled | completed | `results/ksivi_debug/logs/project_banana_baseline.log` | Run dir `results/ksivi_debug/KSIVI/banana/20260328_123306`; TB summary extracted separately |
| B002 | official | banana | `sivistein_2d.KernelSIVI` with `configs/banana.yml` and `num_epochs=20` (2000 inner iters) | completed | `D:\PKU\Programming\StatComp\KSIVI\expkernelSIVI\debug_logs\official_banana_baseline_rerun.log` | Official log path `D:\PKU\Programming\StatComp\KSIVI\expkernelSIVI\debug_logs\banana_baseline_rerun\final.log` |

## Baseline Observations

- `B001` current repo, banana, step 2000:
  - loss `7.1259`
  - KL `2.2191`
  - W2 `0.5806`
  - KSD `4.9868`
  - average `z` norm near `2.54`
  - kernel bandwidth near `0.52`
- `B002` official repo, banana, inner iter 2000:
  - loss `0.0554`
  - `compu_targetscore` mean `-0.3317`
  - `neg_score_implicit` mean `2.7264`
  - combined score mean `-0.4059`
- Fresh baseline conclusion:
  - The current repo and official repo are not in the same training regime on banana even before high-dimensional targets are considered.
  - The difference is too large to treat as noise or metric-format drift.
