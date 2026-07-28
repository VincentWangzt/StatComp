# Shared-DSIVI score approximation

At each seed and training stage, all method-native estimators use that cell's x_shaped DSIVI variational checkpoint.
The posterior-HMC reference uses 20 chains, 1,000 burn-in transitions, and 5,000 retained samples per chain. Per-chain score means are persisted at the paths in checkpoint_metrics.csv.

Values are mean ± sample standard deviation across seeds 42, 43, 45, 49, 50.

## Method–HMC L2 over training

| Checkpoint | SIVI | UIVI | AISIVI | DSIVI |
|---:|---:|---:|---:|---:|
| 2,000 (20%) | 1.5438e+02 ± 1.5264e+02 | 1.1887e+03 ± 1.1394e+03 | 4.5529e+01 ± 7.6908e+01 | 8.9904e-01 ± 9.0793e-01 |
| 4,000 (40%) | 1.1330e+02 ± 6.0589e+01 | 8.8981e+02 ± 5.3571e+02 | 1.3478e+01 ± 1.8378e+01 | 8.0206e-01 ± 8.6722e-01 |
| 6,000 (60%) | 7.8763e+01 ± 3.9973e+01 | 8.8087e+02 ± 7.0376e+02 | 7.2486e+00 ± 4.6709e+00 | 4.0713e-01 ± 4.7135e-01 |
| 8,000 (80%) | 8.1909e+01 ± 6.1060e+01 | 9.7244e+02 ± 8.0178e+02 | 2.0203e+01 ± 1.5426e+01 | 3.0264e-01 ± 3.0688e-01 |
| 10,000 (100%) | 1.0779e+02 ± 4.9088e+01 | 8.9924e+02 ± 6.1104e+02 | 6.8810e+00 ± 6.9901e+00 | 3.3116e-01 ± 5.4839e-01 |

## HMC internal L2 over training

| Checkpoint | HMC internal L2 | HMC mean MCSE L2 | Post-burn acceptance | Score R-hat p95 | Quality |
|---:|---:|---:|---:|---:|---:|
| 2,000 (20%) | 1.2115e+00 ± 8.5442e-01 | 6.3764e-02 ± 4.4969e-02 | 98.23 ± 1.20% | 1.0047e+00 ± 1.5383e-03 | 5/5 pass |
| 4,000 (40%) | 7.3456e-01 ± 7.3330e-01 | 3.8661e-02 ± 3.8595e-02 | 98.64 ± 0.62% | 1.0059e+00 ± 1.9233e-03 | 5/5 pass |
| 6,000 (60%) | 6.4447e-01 ± 6.9737e-01 | 3.3920e-02 ± 3.6704e-02 | 98.78 ± 0.76% | 1.0068e+00 ± 2.3093e-03 | 5/5 pass |
| 8,000 (80%) | 5.7692e-01 ± 5.2846e-01 | 3.0364e-02 ± 2.7814e-02 | 98.68 ± 0.94% | 1.0080e+00 ± 2.9811e-03 | 5/5 pass |
| 10,000 (100%) | 6.2316e-01 ± 6.7189e-01 | 3.2798e-02 ± 3.5362e-02 | 98.67 ± 0.88% | 1.0073e+00 ± 2.7610e-03 | 5/5 pass |

## Native UIVI acceptance over training

| Checkpoint | Average acceptance |
|---:|---:|
| 2,000 (20%) | 13.73 ± 6.00% |
| 4,000 (40%) | 12.01 ± 7.40% |
| 6,000 (60%) | 17.32 ± 13.11% |
| 8,000 (80%) | 17.57 ± 13.03% |
| 10,000 (100%) | 17.16 ± 13.13% |

Method–HMC L2 is the mean over the 1,024 fixed z values of the squared Euclidean difference between the method score and the mean of the 20 HMC chain-score estimates.
HMC internal L2 is the mean squared deviation of individual HMC chain-score estimates from their 20-chain mean.
UIVI acceptance is averaged over all native UIVI HMC transitions and z values, including its five burn-in transitions.
