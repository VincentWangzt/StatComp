# KDVI VI Architecture Sweep Summary

Complete architecture groups: **52 / 54**.
Metrics are means and sample standard deviations across seeds 0, 1, and 7.

## Winners

- **Final KL-ITE:** `eps64-h256-l3-silu` - 0.057906 +/- 0.025660
- **Final W2:** `eps64-h256-l3-elu` - 0.108395 +/- 0.049530

## Final KL/W2 Pareto Front

| Recipe | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `eps64-h256-l3-silu` | 0.057906 +/- 0.025660 | 0.273318 +/- 0.082947 |
| `eps16-h512-l4-silu` | 0.066043 +/- 0.024893 | 0.223747 +/- 0.098831 |
| `eps16-h512-l3-elu` | 0.067397 +/- 0.010976 | 0.169912 +/- 0.027813 |
| `eps64-h256-l3-elu` | 0.067857 +/- 0.022007 | 0.108395 +/- 0.049530 |

## Incomplete Architecture Groups

| Recipe | Complete seeds |
|---|---|
| `eps16-h512-l4-elu` | 1,7 |
| `eps64-h512-l3-elu` | 0,1 |
