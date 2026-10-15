# Canonical KSIVI X-shaped evaluation

Source manifest: `campaigns/ksivi_x_shaped/manifest.json`.
Seeds 42--51; 50,000 iterations each. Batches: [42, 43, 44], [45, 46, 47], [48, 49, 50], [51].
One RTX 4090, at most three concurrent seeds. Per-run wall-clock times reflect shared GPU execution.
Constant conditional-variance initialization 1.313; mean-network LR 0.001; variance-head LR 0.00001.
The saved runner configuration uses a two-dimensional standard Gaussian mixing prior.
Linear annealing over 25,000 steps; Gaussian kernel; scheduler gamma 0.9 every 1,000 steps.
Evaluation uses each run's saved configuration and final checkpoint:
5,000 VI samples and 20 batches of 2,048 auxiliary samples for KL-style (-ELBO);
10,000 samples and 5,000 projections for sliced W2.
The refreshed grids keep the preselected seed 42 for KSIVI on x_shaped.

| Seed | KL-style (-ELBO) | Sliced W2 | Wall-clock (s) |
| --- | ---: | ---: | ---: |
| 42 | 0.082895 | 0.350274 | 418.06 |
| 43 | 0.031754 | 0.139769 | 442.09 |
| 44 | 0.083892 | 0.314781 | 418.07 |
| 45 | 0.025561 | 0.116084 | 418.04 |
| 46 | 0.016979 | 0.122174 | 432.15 |
| 47 | 0.058975 | 0.235532 | 422.09 |
| 48 | 0.091657 | 0.320627 | 416.11 |
| 49 | 0.087565 | 0.325378 | 410.04 |
| 50 | 0.035901 | 0.163600 | 430.18 |
| 51 | 0.020770 | 0.130212 | 362.03 |

Ten-seed KL-style mean ± SE: 0.053595 ± 0.009662.
Ten-seed sliced W2 mean ± SE: 0.221843 ± 0.030811.
All checkpoints and saved samples are finite; all optimizer states have exactly 50,000 updates.
No evaluation errors or warnings. Only the KSIVI/x_shaped metric group is replaced.
