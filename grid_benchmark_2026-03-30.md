# Grid Benchmark 2026-03-30

## Expected Format

Use this file as the canonical manual campaign log for the 2026-03-30 grid run.

Expected top-level sections:
- `Campaign Header`: fixed campaign metadata, launch commit, remote environment, queue plan.
- `Progress Table`: current counts for pending, running, completed, failed.
- `Monitoring Log`: one row per manual check or important immediate intervention.
- `Failure Log`: one row per failed/interrupted run with cause and resolution.
- `Per-Target Summary Tables`: grouped by target, populated during manual checks from extracted metrics plus brief notes.
- `End-of-Campaign Summary`: aggregate outcomes, caveats, and final completion note.

Expected per-run fields when recording finished runs in the summary tables:
- `run_id`, `target`, `variant`, `anneal`
- `config_path`, special overrides, result path, TensorBoard path
- total runtime, average epoch time
- final metrics: ELBO, KL, W2, KSD, MMD, Fisher, RMSE, test LLK, NLL, proxy L2, reverse-model metrics where available
- best metric values and the epochs they occurred
- run status and anomaly notes

Update policy:
- update this file at each 2-hour manual check and at failure investigations
- use only results from this campaign, not historical benchmark numbers
- keep narrative notes concise and put raw metrics in tables
- commit locally after each manual check with new progress
- do not rely on remote-only state; sync artifacts locally before updating

## Current Status

This file is initialized and already tracking the smoke validation and official launch state below.

## Campaign Header

- Commit SHA: `2763607`
- Remote environment: 2x RTX 3080, Python 3.14.2, PyTorch 2.9.0+cu126
- Official run count: 216
- Queue plan: GPU0 + GPU1 independent single-GPU queues

## Progress Table

| Status | Count |
|--------|-------|
| Pending | 168 |
| Running | 2 |
| Completed | 43 |
| Failed | 3 |

## Monitoring Log

| Time | Check Type | Notes |
|------|------------|-------|
| 2026-03-30 16:58 CST | Smoke validation | 5/5 smoke runs completed successfully on remote after the Gaussian KSD memory-safety patch. BNN smoke runs `official_on_bnn_yacht_uivi` and `official_on_bnn_yacht_dsivi_bs4096_rbs2048` no longer OOM at `metric.ksd.num_samples=2000`. |
| 2026-03-30 17:01 CST | Official launch | Started `grid_official_gpu0` and `grid_official_gpu1` tmux queues on remote after pushing local prep commits and pulling remotely. |
| 2026-03-30 17:03 CST | Immediate check | `gpu0` running `official_on_banana_sivi`; `gpu1` running `official_on_banana_aisivi`. Both GPUs active; no failures or worker errors. |
| 2026-03-30 17:04 CST | Immediate check | `official_on_banana_aisivi` advanced through metric logging cleanly under high memory use on GPU1; no early OOM observed. |
| 2026-03-30 17:22 CST | Manual check | Official queues healthy. Completed runs: `official_on_banana_sivi`, `official_on_banana_aisivi`, `official_on_banana_dsivi_default`. Active runs: `official_on_banana_uivi` on GPU0 and `official_off_banana_uivi` on GPU1. `nvidia-smi` showed both GPUs active at about 5.8 GiB used with no failures or worker errors. |
| 2026-03-30 18:32 CST | Monitoring check | Campaign reached 19 completed official runs. GPU1 continued normally into `official_off_multimodal_uivi`. GPU0 queue halted on `official_off_banana_rsivi` after a numerical failure in the RSIVI reverse model with annealing off. |
| 2026-03-30 19:16 CST | Manual check | Campaign recovered after the RSIVI failure. GPU0 resumed past the investigated failed run and progressed to `official_on_multimodal_uivi`; GPU1 progressed to `official_off_multimodal_dsivi_bs4096_rbs4096`. Totals at this checkpoint: 26 completed, 1 failed, 2 running, 0 worker errors. |
| 2026-03-30 20:33 CST | Monitoring check | Both queues later hit additional RealNVP-based failures and paused again. GPU0 failed on `official_off_multimodal_rsivi`; GPU1 failed on `official_on_student_uc_aisivi`. At investigation time the campaign stood at 43 completed, 3 failed, 0 worker errors. |
| 2026-03-30 22:55 CST | Failure recovery | Both queues were resumed past the investigated failures using the queue runner's continue control. GPU0 restarted at `official_on_x_shaped_sivi`; GPU1 restarted at `official_off_student_uc_sivi`. |

## Failure Log

| Time | Run ID | GPU | Issue | Resolution |
|------|--------|-----|-------|------------|
| 2026-03-30 18:32 CST | `official_off_banana_rsivi` | 0 | Training crashed at epoch 371. The reverse `ConditionalRealNVP` began producing non-finite samples after the loss exploded under annealing-off training; runtime error: `Failed to obtain finite samples from RealNVP after 3 attempts.` | Queue stopped as intended. Failure recorded locally and investigated from remote logs. GPU0 was later resumed past this investigated failed run using the queue runner's `--continue-past-failed` control so the remaining queue could continue without erasing the failure record. |
| 2026-03-30 20:33 CST | `official_off_multimodal_rsivi` | 0 | Training crashed at epoch 1502 after non-finite VI loss and repeated non-finite reverse-model samples from `ConditionalRealNVP`. Runtime error again ended with `Failed to obtain finite samples from RealNVP after 3 attempts.` | Failure recorded locally after log inspection. Queue will resume past this investigated failed run using the same continue control. |
| 2026-03-30 20:33 CST | `official_on_student_uc_aisivi` | 1 | Training crashed at epoch 3383 when AISIVI’s reverse `ConditionalRealNVP` produced non-finite samples on three consecutive retries. | Failure recorded locally after log inspection. Queue will resume past this investigated failed run using the same continue control. |

## Per-Target Summary Tables

Update manually using script-generated summaries at each 2-hour manual check.

## End-of-Campaign Summary

Pending.
