# KDVI 8-Gaussians MCMC Sweep — Summary

Complete recipe groups: **120 / 120**.
Metrics are final epoch-100K means and sample standard deviations across seeds 0, 1, and 7.

## Winners

- **KL-ITE:** `sgld-coupled-step0p05-k10-ann50k` — 0.064768 ± 0.004337
- **W2:** `sgld-coupled-step0p20-k1to20-ann50k` — 0.071161 ± 0.001053

## KL/W2 Pareto front

| Recipe | KL-ITE mean ± std | W2 mean ± std |
|---|---:|---:|
| `sgld-coupled-step0p05-k10-ann50k` | 0.064768 ± 0.004337 | 0.074617 ± 0.028245 |
| `sgld-coupled-step0p20-k1to20-ann50k` | 0.067193 ± 0.013463 | 0.071161 ± 0.001053 |

## Incomplete recipe groups

None.
