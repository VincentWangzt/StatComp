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
| B010 | official | Bnn_boston | `sivistein_bnn.py` with `configs/kernel_sivi_boston.yml` short baseline rerun | completed | `D:\PKU\Programming\StatComp\KSIVI\expkernelSIVI\debug_logs\bnn_boston_baseline_rerun\final.log` | Ran after installing `scikit-learn` into the shared local `uv` environment |
| B011 | current | Bnn_boston | official-raw split verification probe | completed | `results/ksivi_debug/logs/verify_boston_official_split_v2.txt` | Current repo Boston loader now matches the official split, dev split, and normalization exactly |
| B012 | current | Bnn_boston | repaired short rerun on exact official split, 1000 epochs | completed | `results/ksivi_debug/logs/project_bnn_boston_officialsplit_1000.log` | TB extract in `results/ksivi_debug/logs/extract_bnn_boston_officialsplit_1000_tb.txt` |
| B013 | current | banana | repaired rerun after restoring kernel-bandwidth gradients, 2000 epochs | completed | `results/ksivi_debug/logs/project_banana_kernelgrad_2000.log` | TB extract in `results/ksivi_debug/logs/extract_banana_kernelgrad_2000_tb.txt` |
| B014 | current | LRwaveform | repaired rerun after restoring kernel-bandwidth gradients, 5000 epochs | completed | `results/ksivi_debug/logs/project_lrwaveform_kernelgrad_5000.log` | TB extract in `results/ksivi_debug/logs/extract_lrwaveform_kernelgrad_5000_tb.txt` |
| B015 | current | Bnn_boston | repaired rerun after restoring kernel-bandwidth gradients, 1000 epochs | completed | `results/ksivi_debug/logs/project_bnn_boston_kernelgrad_1000.log` | TB extract in `results/ksivi_debug/logs/extract_bnn_boston_kernelgrad_1000_tb.txt` |
| B016 | current | Bnn_boston | repaired long rerun after restoring kernel-bandwidth gradients, 5000 epochs | completed | `results/ksivi_debug/logs/project_bnn_boston_kernelgrad_5000.log` | TB extract in `results/ksivi_debug/logs/extract_bnn_boston_kernelgrad_5000_tb.txt` |
| B017 | current | LRwaveform | repaired long rerun after restoring kernel-bandwidth gradients, 20000 epochs | completed | `results/ksivi_debug/logs/project_lrwaveform_kernelgrad_20000.log` | TB extract in `results/ksivi_debug/logs/extract_lrwaveform_kernelgrad_20000_tb.txt` |
| B018 | current | multimodal / x_shaped / student_uc | repaired toy reruns after restoring kernel-bandwidth gradients, 2000 epochs | completed | `results/ksivi_debug/logs/project_multimodal_kernelgrad_2000.log`, `results/ksivi_debug/logs/project_xshaped_kernelgrad_2000.log`, `results/ksivi_debug/logs/project_student_kernelgrad_2000.log` | Combined TB extract in `results/ksivi_debug/logs/extract_toy_kernelgrad_2000_tb.txt` |
| B019 | current | multimodal / student_uc | extended repaired toy reruns, 10000 epochs | completed | `results/ksivi_debug/logs/project_multimodal_kernelgrad_10000.log`, `results/ksivi_debug/logs/project_student_kernelgrad_10000.log` | Combined TB extract in `results/ksivi_debug/logs/extract_multimodal_student_kernelgrad_10000_tb.txt` |

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
  - The official short baseline is now available locally after installing `scikit-learn`.
  - Current repaired Boston results collected before `B011` should not be treated as official-parity evidence because the official raw split path was still mismatched at that point.
  - `B011` confirms the current repo can now reproduce the official Boston preprocessing path exactly, so the next Boston reruns are directly comparable.
- Fresh rerun conclusions after `B011` and the kernel-bandwidth-gradient fix:
  - `B012` moved current Boston much closer to the official short baseline immediately: RMSE `2.94` and NLL `2.56` at step `1000`, versus official RMSE `2.64` and NLL `2.51`.
  - `B016` shows Boston continuing to improve and then flatten: RMSE `2.93 -> 2.73 -> 2.64 -> 2.62 -> 2.61`, NLL `2.53 -> 2.49 -> 2.47 -> 2.47 -> 2.47`.
  - `B014` and `B017` show that restoring kernel-bandwidth gradients fixed the long-run LR instability: KSD improved monotonically from `1.91` at `1000` to `0.39` at `5000`, then to `0.065` at `20000`, with bandwidth staying bounded near `0.86`.
  - `B013` keeps banana in the repaired regime after the kernel fix: step-`2000` loss `0.0581`, KL `0.4083`, KSD `0.0507`.
  - `B018` and `B019` show `x_shaped` is healthy and `multimodal` improves steadily with more budget, while `student_uc` still has unresolved mismatch signs because its KSD worsens as KL/W2 improve.
