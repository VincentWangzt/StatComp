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
| B003 | official | LRwaveform | manual short probe in official repo, 200 optimization steps on `datasets/waveform.mat` | completed | command-only probe | Loss dropped from `3259.42` to `50.66`; sample norm stayed near `1.46` |
| B004 | current | LRwaveform | manual short probe in current repo on prepared waveform data, 200 optimization steps | completed | command-only probe | Diverged badly; loss remained `190211.55`, sample norm `6.13` |
| B005 | current | LRwaveform | short run after switching KSIVI to official waveform `.mat` split, 200 epochs | completed | `results/ksivi_debug/logs/project_lrwaveform_patchcheck_v3.log` | KSD `5.99`; no immediate explosion |
| B006 | current | LRwaveform | repaired long run, 1000 epochs on official waveform `.mat` split | completed | `results/ksivi_debug/logs/project_lrwaveform_repaired_v2.log` | KSD improved to `1.45` at 1000 |
| B007 | current | LRwaveform | repaired long run, 5000 epochs on official waveform `.mat` split | completed | `results/ksivi_debug/logs/project_lrwaveform_long.log` | Not fully stable; KSD worsened after 1000 before partially recovering |
| B008 | current | Bnn_boston | repaired short run with VI warm start, 1000 epochs | completed | `results/ksivi_debug/logs/project_bnn_boston_repaired_v2.log` | RMSE improved to `10.52`, NLL to `5.53` |
| B009 | current | Bnn_boston | repaired long run with VI warm start, 5000 epochs | completed | `results/ksivi_debug/logs/project_bnn_boston_long.log` | RMSE `4.20`, NLL `2.75`, KSD `17.71`; strong convergence trend |

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
- Fresh LRwaveform conclusion:
  - The prepared waveform dataset in this repo is not the same split as the official KSIVI waveform dataset.
  - Current prepared data had `4000` training examples, while the official waveform `.mat` training split had `400`.
  - Switching the repaired KSIVI run to the official waveform data source removed the immediate LR explosion.
- Fresh Bnn_boston conclusion:
  - The repaired Boston path with warm start is no longer catastrophically broken.
  - Boston metrics improve steadily with longer budgets, so the current evidence points to a convergence problem rather than a structural failure.
