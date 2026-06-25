# KDVI 8-Gaussians LR Sweep Summary

Complete recipe groups: **29 / 30**.
Metrics are final epoch-100K means and sample standard deviations across seeds 0, 1, and 7.

## Winners

- **KL-ITE:** `lr1em3-steplr5000-gamma0p9` - 0.043687 +/- 0.007083
- **W2:** `lr2em3-steplr5000-gamma0p75` - 0.095087 +/- 0.013721

## KL/W2 Pareto front

| Recipe | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `lr1em3-steplr5000-gamma0p9` | 0.043687 +/- 0.007083 | 0.323196 +/- 0.038178 |
| `lr2em3-steplr5000-gamma0p85` | 0.054636 +/- 0.012749 | 0.189905 +/- 0.101323 |
| `lr2em3-steplr5000-gamma0p75` | 0.057475 +/- 0.014113 | 0.095087 +/- 0.013721 |

## Incomplete recipe groups

| Recipe | Complete seeds |
|---|---|
| `lr2em3-steplr5000-gamma1p0` | 0,7 |
