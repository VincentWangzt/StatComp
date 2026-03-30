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
| Pending | 211 |
| Running | 2 |
| Completed | 3 |
| Failed | 0 |

## Monitoring Log

| Time | Check Type | Notes |
|------|------------|-------|
| 2026-03-30 16:58 CST | Smoke validation | 5/5 smoke runs completed successfully on remote after the Gaussian KSD memory-safety patch. BNN smoke runs `official_on_bnn_yacht_uivi` and `official_on_bnn_yacht_dsivi_bs4096_rbs2048` no longer OOM at `metric.ksd.num_samples=2000`. |
| 2026-03-30 17:01 CST | Official launch | Started `grid_official_gpu0` and `grid_official_gpu1` tmux queues on remote after pushing local prep commits and pulling remotely. |
| 2026-03-30 17:03 CST | Immediate check | `gpu0` running `official_on_banana_sivi`; `gpu1` running `official_on_banana_aisivi`. Both GPUs active; no failures or worker errors. |
| 2026-03-30 17:04 CST | Immediate check | `official_on_banana_aisivi` advanced through metric logging cleanly under high memory use on GPU1; no early OOM observed. |
| 2026-03-30 17:22 CST | Manual check | Official queues healthy. Completed runs: `official_on_banana_sivi`, `official_on_banana_aisivi`, `official_on_banana_dsivi_default`. Active runs: `official_on_banana_uivi` on GPU0 and `official_off_banana_uivi` on GPU1. `nvidia-smi` showed both GPUs active at about 5.8 GiB used with no failures or worker errors. |

## Failure Log

| Time | Run ID | GPU | Issue | Resolution |
|------|--------|-----|-------|------------|

## Per-Target Summary Tables

Update manually using script-generated summaries at each 2-hour manual check.

## End-of-Campaign Summary

Pending.
