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

This file is initialized and tracking the live 216-run campaign, including manual checks, immediate investigations, and queue recovery actions.

## Campaign Header

- Commit SHA: `2763607`
- Remote environment: 2x RTX 3080, Python 3.14.2, PyTorch 2.9.0+cu126
- Official run count: 216
- Queue plan: GPU0 + GPU1 independent single-GPU queues

## Progress Table

| Status | Count |
|--------|-------|
| Pending | 127 |
| Running | 1 |
| Completed | 78 |
| Failed | 11 |

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
| 2026-03-30 23:26 CST | Failure recovery | GPU1 encountered another AISIVI student-target failure on `official_off_student_uc_aisivi`, was investigated, and then resumed past that run. Campaign state after recovery: 48 completed, 4 failed, 2 running. |
| 2026-03-31 00:01 CST | Failure investigation | Both queues paused again. GPU0 failed on `official_off_x_shaped_rsivi` with another `ConditionalRealNVP` non-finite sampling crash at epoch 246. GPU1 failed on `official_on_langevin_post_aisivi` during reverse warmup when `calculate_rev_KSD()` triggered a CUDA OOM. Campaign state at investigation time: 58 completed, 6 failed, 0 worker errors. |
| 2026-03-31 00:26 CST | Immediate failure check | GPU0 failed again on `official_on_student_uc_rsivi` with the same `ConditionalRealNVP` non-finite sampling crash, while GPU1 remained healthy on `official_on_langevin_post_dsivi_default`. Campaign state at investigation time: 60 completed, 7 failed, 1 running, 0 worker errors. |
| 2026-03-31 00:49 CST | Immediate failure check | GPU0 later failed on `official_off_student_uc_rsivi` after a NaN/Inf reverse-model loss warning and another `ConditionalRealNVP` non-finite sampling crash. GPU1 completed `official_on_langevin_post_dsivi_default` cleanly and advanced to `official_off_langevin_post_uivi`. Campaign state at investigation time: 66 completed, 8 failed, 1 running, 0 worker errors. |
| 2026-03-31 02:21 CST | Monitoring check | After a one-hour wait, GPU0 had completed `official_on_langevin_post_sivi` and advanced into `official_on_langevin_post_uivi`. GPU1 completed `official_off_langevin_post_uivi` but failed on `official_off_langevin_post_aisivi` during reverse warmup with the same Langevin-post AISIVI OOM pattern seen earlier. Campaign state at investigation time: 68 completed, 9 failed, 1 running, 0 worker errors. |
| 2026-03-31 03:23 CST | Monitoring check | The next one-hour probe found GPU0 had completed `official_on_langevin_post_uivi` but then failed on `official_on_langevin_post_rsivi` after another `ConditionalRealNVP` instability. GPU1 remained healthy on `official_off_langevin_post_ksivi_custom`. Campaign state at investigation time: 69 completed, 10 failed, 1 running, 0 worker errors. |
| 2026-03-31 06:26 CST | Manual check | No new run boundary since the last recovery, but both long `Langevin_post` `KSIVI-custom` runs remained healthy. GPU0 log advanced to about `63680/100000` epochs and GPU1 log advanced to about `84283/100000`, with fresh log timestamps on both queues. Campaign totals stayed at 69 completed, 10 failed, 1 running, 0 worker errors. |
| 2026-03-31 08:27 CST | Manual check | Both long `Langevin_post` `KSIVI-custom` runs completed cleanly during the 2-hour interval. GPU0 advanced into `official_on_langevin_post_ksivi_standard_cg` and GPU1 advanced into `official_off_langevin_post_ksivi_standard_cg`; current log positions were about `5380/100000` on GPU0 and `25892/100000` on GPU1. Campaign totals reached 71 completed, 10 failed, 1 running, 0 worker errors. |
| 2026-03-31 14:33 CST | Monitoring resync | The previous long sleep-probe overran the tool timeout, but the remote campaign had kept progressing. By the resync check, the campaign had advanced to 78 completed and 11 failed. GPU0 had moved through the remaining `Langevin_post` KSIVI and DSIVI variants and was healthy on `official_off_langevin_post_sivi`; GPU1 had advanced into `LRwaveform` and failed late in `official_on_lrwaveform_aisivi`. |

## Failure Log

| Time | Run ID | GPU | Issue | Resolution |
|------|--------|-----|-------|------------|
| 2026-03-30 18:32 CST | `official_off_banana_rsivi` | 0 | Training crashed at epoch 371. The reverse `ConditionalRealNVP` began producing non-finite samples after the loss exploded under annealing-off training; runtime error: `Failed to obtain finite samples from RealNVP after 3 attempts.` | Queue stopped as intended. Failure recorded locally and investigated from remote logs. GPU0 was later resumed past this investigated failed run using the queue runner's `--continue-past-failed` control so the remaining queue could continue without erasing the failure record. |
| 2026-03-30 20:33 CST | `official_off_multimodal_rsivi` | 0 | Training crashed at epoch 1502 after non-finite VI loss and repeated non-finite reverse-model samples from `ConditionalRealNVP`. Runtime error again ended with `Failed to obtain finite samples from RealNVP after 3 attempts.` | Failure recorded locally after log inspection. Queue will resume past this investigated failed run using the same continue control. |
| 2026-03-30 20:33 CST | `official_on_student_uc_aisivi` | 1 | Training crashed at epoch 3383 when AISIVI’s reverse `ConditionalRealNVP` produced non-finite samples on three consecutive retries. | Failure recorded locally after log inspection. Queue will resume past this investigated failed run using the same continue control. |
| 2026-03-30 23:26 CST | `official_off_student_uc_aisivi` | 1 | Training crashed at epoch 1584 after repeated non-finite importance-sampling weights, skipped VI updates, and then non-finite `ConditionalRealNVP` samples on all three retries. | Failure recorded locally after log inspection. GPU1 resumed past this investigated failed run using the same continue control. |
| 2026-03-31 00:00 CST | `official_off_x_shaped_rsivi` | 0 | Training crashed at epoch 246 after `ConditionalRealNVP` sampling returned non-finite values on three consecutive retries during RSIVI reverse sampling. Runtime error ended with `Failed to obtain finite samples from RealNVP after 3 attempts.` | Failure recorded locally after log inspection. GPU0 will resume past this investigated failed run using the same continue control. |
| 2026-03-31 00:00 CST | `official_on_langevin_post_aisivi` | 1 | Reverse warmup crashed at epoch 99 when `calculate_rev_KSD()` attempted a large reverse-model sample and triggered `torch.OutOfMemoryError`, requesting another 3.91 GiB on a 10 GiB RTX 3080. | Failure recorded locally after log inspection. GPU1 will resume past this investigated failed run using the same continue control so the remaining official queue can proceed while preserving the failure record. |
| 2026-03-31 00:26 CST | `official_on_student_uc_rsivi` | 0 | Training crashed at epoch 351 after repeated warnings that `ConditionalRealNVP` sampling had produced non-finite values, ending with `Failed to obtain finite samples from RealNVP after 3 attempts.` | Failure recorded locally after log inspection. GPU0 will resume past this investigated failed run using the same continue control while GPU1 continues its active run. |
| 2026-03-31 00:49 CST | `official_off_student_uc_rsivi` | 0 | Training crashed at epoch 3805 after a `NaN or Inf detected in reverse model loss` warning and three consecutive non-finite `ConditionalRealNVP` sampling retries, ending with `Failed to obtain finite samples from RealNVP after 3 attempts.` | Failure recorded locally after log inspection. GPU0 will resume past this investigated failed run using the same continue control while GPU1 continues its active run. |
| 2026-03-31 02:21 CST | `official_off_langevin_post_aisivi` | 1 | Reverse warmup crashed at epoch 99 when `calculate_rev_KSD()` attempted a large reverse-model sample and again triggered `torch.OutOfMemoryError`, requesting another 3.91 GiB on a 10 GiB RTX 3080. | Failure recorded locally after log inspection. GPU1 will resume past this investigated failed run using the same continue control while GPU0 continues its active run. |
| 2026-03-31 03:23 CST | `official_on_langevin_post_rsivi` | 0 | Training crashed at epoch 932 after long stalls, repeated `ConditionalRealNVP` non-finite sampling warnings, and a final `Failed to obtain finite samples from RealNVP after 3 attempts.` runtime error. | Failure recorded locally after log inspection. GPU0 will resume past this investigated failed run using the same continue control while GPU1 continues its active run. |
| 2026-03-31 14:33 CST | `official_on_lrwaveform_aisivi` | 1 | Training crashed late at epoch 9957 after repeated `NaN or Inf detected in reverse model loss` warnings and three consecutive non-finite `ConditionalRealNVP` sampling retries, ending with `Failed to obtain finite samples from RealNVP after 3 attempts.` | Failure recorded locally after log inspection. GPU1 will resume past this investigated failed run using the same continue control while GPU0 continues its active run. |

## Per-Target Summary Tables

Update manually using script-generated summaries at each 2-hour manual check.

## End-of-Campaign Summary

Pending.
