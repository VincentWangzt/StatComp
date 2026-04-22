# Langevin/BNN Knob Grid 2026-04-22

## Campaign Header

- Campaign launch commit SHA: `91027db`
- Final report commit SHA: `2f43e8b`
- Remote worktree: `~/ruivi`
- Remote artifact root: `/root/autodl-tmp`
- Remote GPU: RTX 3080 queue `gpu0`
- Official run count: 18
- Smoke run count: 4

## Progress Table

| Status | Count |
|--------|-------|
| Pending | 0 |
| Running | 0 |
| Completed | 18 |
| Failed | 0 |

## Monitoring Log

| Time | Check Type | Notes |
|------|------------|-------|
| 2026-04-22 22:07 CST | Smoke launch | Started 4-run smoke queue in `tmux` session `lbg_smoke_20260422`. |
| 2026-04-22 22:09 CST | Smoke complete | All 4 smoke runs completed successfully. |
| 2026-04-22 22:11 CST | Official launch | Started 18-run official queue in `tmux` session `lbg_official_20260422`. |
| 2026-04-22 22:28 CST | Official progress | AISIVI default completed; AISIVI CG-global substitution running. |
| 2026-04-22 23:08 CST | Official progress | KSIVI annealing comparison completed; DSIVI Langevin LR sweep running. |
| 2026-04-22 23:49 CST | Official progress | Langevin portion completed; DSIVI BNN batch-size runs started. |
| 2026-04-23 00:20 CST | Official complete | All 18 official runs completed successfully; GPU idle. |

## Failure Log

| Time | Run ID | Issue | Resolution |
|------|--------|-------|------------|
| - | - | None | - |

## Final Report

Generated summaries:

- `campaigns/langevin_bnn_knob_grid_20260422/generated_reports/official_completed_runs.csv`
- `campaigns/langevin_bnn_knob_grid_20260422/generated_reports/knob_grid_report.md`

### Q1 AISIVI Langevin VI Model

Higher ELBO and KDE ELM are better. The current default AISIVI VI model clearly outperformed the KSIVI-style Conditional Gaussian Global substitution.

| Variant | ELBO final | ELBO best | KDE ELM final | KDE ELM best | Minutes |
|---------|------------|-----------|---------------|--------------|---------|
| default `ConditionalGaussianGlobalUniform` | -210.110 | -210.110 | 70.7098 | 70.7106 | 17.1 |
| substituted `ConditionalGaussianGlobal` | -719.886 | -719.886 | 27.6905 | 27.6905 | 17.4 |

### Q2 KSIVI Langevin Annealing

Turning annealing on with 25K anneal steps improved KSIVI on both headline metrics.

| Variant | ELBO final | ELBO best | KDE ELM final | KDE ELM best | Minutes |
|---------|------------|-----------|---------------|--------------|---------|
| anneal off default | -239.997 | -239.997 | 51.3335 | 51.3335 | 10.4 |
| anneal on, 25K steps | -204.325 | -204.325 | 67.5671 | 67.5671 | 10.6 |

### Q3 Langevin Learning Rate

`1e-3` beat `2e-4` for both DSIVI and KSIVI on ELBO and KDE ELM. Across these four runs, KSIVI at `1e-3` had the best KDE ELM, while DSIVI at `1e-3` had the best final ELBO.

| Method | VI LR | ELBO final | ELBO best | KDE ELM final | KDE ELM best | Minutes |
|--------|-------|------------|-----------|---------------|--------------|---------|
| DSIVI | 2e-4 | -657.733 | -657.093 | 32.2632 | 32.2632 | 9.8 |
| DSIVI | 1e-3 | -190.298 | -190.093 | 74.7319 | 74.7319 | 9.7 |
| KSIVI | 2e-4 | -239.997 | -239.997 | 51.3335 | 51.3335 | 10.5 |
| KSIVI | 1e-3 | -201.302 | -191.182 | 75.0513 | 75.0669 | 10.4 |

### Q4 DSIVI BNN Batch Size

Batch 128 was faster on every BNN target. Metric impact was mixed: it improved Boston and Concrete, was worse on Power and Winered, and was essentially tied/slightly worse on Protein.

| Target | Batch | Test LLK final | Test LLK best | RMSE final | RMSE best | NLL final | NLL best | Minutes |
|--------|-------|----------------|---------------|------------|-----------|-----------|----------|---------|
| Bnn_boston | 1024 | -2.52063 | -2.50490 | 2.88790 | 2.81678 | 2.52063 | 2.50490 | 3.7 |
| Bnn_boston | 128 | -2.49928 | -2.49502 | 2.79080 | 2.77051 | 2.49928 | 2.49502 | 2.7 |
| Bnn_concrete | 1024 | -3.29707 | -3.29619 | 6.41488 | 6.41064 | 3.29707 | 3.29619 | 3.3 |
| Bnn_concrete | 128 | -3.23140 | -3.23140 | 6.08422 | 6.08422 | 3.23140 | 3.23140 | 2.7 |
| Bnn_power | 1024 | -2.78492 | -2.77805 | 3.89488 | 3.86296 | 2.78492 | 2.77805 | 3.0 |
| Bnn_power | 128 | -2.90444 | -2.82523 | 4.40985 | 3.99780 | 2.90444 | 2.82523 | 2.6 |
| Bnn_protein | 1024 | -2.92902 | -2.92902 | 4.50261 | 4.50261 | 2.92902 | 2.92902 | 4.3 |
| Bnn_protein | 128 | -2.92935 | -2.92935 | 4.50481 | 4.50481 | 2.92935 | 2.92935 | 2.8 |
| Bnn_winered | 1024 | -0.980729 | -0.920431 | 0.640557 | 0.607016 | 0.980729 | 0.920431 | 3.6 |
| Bnn_winered | 128 | -1.02909 | -0.930509 | 0.666322 | 0.613260 | 1.02909 | 0.930509 | 2.8 |
