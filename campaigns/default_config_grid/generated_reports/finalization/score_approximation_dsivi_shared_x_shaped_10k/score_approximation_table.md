# Shared-DSIVI score approximation

All method-native estimators use the same x_shaped DSIVI variational checkpoint: `results/default_config_grid/DSIVI/x_shaped/20260504_125719/checkpoints/epoch_10000`.
The posterior-HMC reference uses 20 chains, 1,000 burn-in transitions, and 5,000 retained samples per chain. Its saved per-chain score means are in `/root/ruivi/results/default_config_grid/score_approximation_dsivi_shared_x_shaped_10k/reference_cache/x_shaped/seed_42/epoch_10000/hmc_2228085552679786.pt`.

| Method | Method–HMC L2 | HMC internal L2 | Native auxiliaries | UIVI acceptance |
|---|---:|---:|---:|---:|
| SIVI | 5.412715e+01 | 1.512297e-01 | 4097 | — |
| UIVI | 3.070058e+02 | 1.512297e-01 | 5 | 38.44% |
| AISIVI | 1.368480e+00 | 1.512297e-01 | 1024 | — |
| DSIVI | 7.480492e-02 | 1.512297e-01 | 0 | — |

Method–HMC L2 is the mean over the 1,024 fixed z values of the squared Euclidean difference between the method score and the mean of the 20 HMC chain-score estimates.
HMC internal L2 is the mean squared deviation of individual HMC chain-score estimates from their 20-chain mean.
UIVI acceptance is averaged over all native UIVI HMC transitions and z values, including its five burn-in transitions.
