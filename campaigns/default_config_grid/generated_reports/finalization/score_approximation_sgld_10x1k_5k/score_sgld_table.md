# DIVI (DSIVI checkpoint)–SGLD Score-Approximation Analysis

All values are mean ± sample standard deviation across seeds 42, 43, 45, 49, 50. Values use fixed-point notation.

The reference is the average of 10 independent SGLD-group score means; each group averages 1,000 terminal epsilon particles. Within-SGLD L2 is calculated across those group means.

> A small within-SGLD L2 measures Monte Carlo agreement between groups; it does not by itself establish mixing or remove common finite-horizon and fixed-step bias.

## Diagnostic warnings

- x_shaped / seed 42: sgld_score_drift_step_2500_to_5000_l2/gold_mcse=6.84477 > 4

| Target | Epoch | DIVI (DSIVI)–SGLD L2 | Within-SGLD L2 | Golden-score MCSE L2 | DIVI–SGLD per-coordinate MSE |
|---|---:|---:|---:|---:|---:|
| 8_gaussians | 10000 | 191.21436195 ± 38.49866632 | 7.76600739 ± 3.42563386 | 0.86288971 ± 0.38062598 | 95.60718098 ± 19.24933316 |
| x_shaped | 10000 | 720.68614064 ± 1360.92435074 | 309.05962772 ± 501.48399384 | 34.33995864 ± 55.72044376 | 360.34307032 ± 680.46217537 |
