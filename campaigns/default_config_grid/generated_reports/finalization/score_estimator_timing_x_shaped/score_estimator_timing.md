# Native score-estimator latency

Latency is synchronized end-to-end wall time on NVIDIA GeForce RTX 3090. Each cell uses 10 untimed warm-up calls followed by 100 timed calls and reports mean ± sample standard deviation.
Checkpoint loading, input `(z, epsilon)` generation, optimizer work, model fitting, diagnostics, and logging are excluded. AISIVI loads its trained reverse-flow checkpoint; it is not trained or refit by this benchmark.

## Batch size 1

| Method | Mean ± SD (ms) | Amortized ms / z | z / second |
|---|---:|---:|---:|
| SIVI | 0.9441 ± 0.0887 | 0.944054 | 1059.3 |
| UIVI | 201.098 ± 26.283 | 201.098459 | 5.0 |
| AISIVI | 5.489 ± 0.148 | 5.489481 | 182.2 |
| DIVI (DSIVI) | 0.1296 ± 0.0024 | 0.129640 | 7713.7 |

## Batch size 128

| Method | Mean ± SD (ms) | Amortized ms / z | z / second |
|---|---:|---:|---:|
| SIVI | 5.316 ± 0.183 | 0.041531 | 24078.3 |
| UIVI | 204.523 ± 23.199 | 1.597834 | 625.8 |
| AISIVI | 8.096 ± 0.222 | 0.063250 | 15810.3 |
| DIVI (DSIVI) | 0.1339 ± 0.0031 | 0.001046 | 955839.3 |

## Estimator boundaries and checkpoints

| Method | Timed estimator | Native auxiliaries | Checkpoint |
|---|---|---:|---|
| SIVI | prior sampling + mixture logsumexp + autograd score | 4097 | `results/default_config_grid/SIVI/x_shaped/20260428_043837/checkpoints/epoch_10000` |
| UIVI | posterior HMC + conditional-score mean | 5 | `results/default_config_grid/UIVI/x_shaped/20260428_051601/checkpoints/epoch_10000` |
| AISIVI | reverse-flow sampling + importance mixture + autograd score | 1024 | `results/default_config_grid/AISIVI/x_shaped/20260428_055416/checkpoints/epoch_10000` |
| DIVI (DSIVI) | score-network forward pass | 0 | `results/default_config_grid/DSIVI/x_shaped/20260504_125719/checkpoints/epoch_10000` |

UIVI uses 5 burn-in transitions, 5 retained transitions, and 5 leapfrog steps per transition.
