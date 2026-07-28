# Shared-checkpoint score accuracy and latency

The 2K–10K columns report Method–HMC L2, \(N^{-1}\sum_{i=1}^{N}\lVert s_{\mathrm{method}}(z_i)-\bar{s}_{\mathrm{HMC}}(z_i)\rVert_2^2\) with \(N=1{,}024\), as mean ± sample standard deviation over seeds 42, 43, 45, 49, and 50; at each stage every method uses the matching `x_shaped` DSIVI variational checkpoint, and \(\bar{s}_{\mathrm{HMC}}\) uses 20 chains with 1,000 burn-in and 5,000 retained samples per chain. AISIVI’s reverse flow is refit for 10,000 steps at each cell. The final \(t_{128}\) column is synchronized score-estimation-only latency in milliseconds for a batch of 128 on an RTX 3090, reported as mean ± sample standard deviation over 100 calls after 10 warm-ups at each method’s seed-42 10K native checkpoint; checkpoint loading, input generation, training/refitting, optimizer work, diagnostics, and logging are excluded. The timed native budgets are 4,097 SIVI mixture components, 5 UIVI retained HMC transitions after 5 burn-in transitions with 5 leapfrog steps each, 1,024 AISIVI importance samples, and one DSIVI score-network forward pass. Lower is better throughout.

| Method | 2K | 4K | 6K | 8K | 10K | \(t_{128}\) (ms) |
|---|---:|---:|---:|---:|---:|---:|
| SIVI | 1.5438e+02 ± 1.5264e+02 | 1.1330e+02 ± 6.0589e+01 | 7.8763e+01 ± 3.9973e+01 | 8.1909e+01 ± 6.1060e+01 | 1.0779e+02 ± 4.9088e+01 | 5.316 ± 0.183 |
| UIVI | 1.1887e+03 ± 1.1394e+03 | 8.8981e+02 ± 5.3571e+02 | 8.8087e+02 ± 7.0376e+02 | 9.7244e+02 ± 8.0178e+02 | 8.9924e+02 ± 6.1104e+02 | 204.523 ± 23.199 |
| AISIVI | 4.5529e+01 ± 7.6908e+01 | 1.3478e+01 ± 1.8378e+01 | 7.2486e+00 ± 4.6709e+00 | 2.0203e+01 ± 1.5426e+01 | 6.8810e+00 ± 6.9901e+00 | 8.096 ± 0.222 |
| DSIVI | 8.9904e-01 ± 9.0793e-01 | 8.0206e-01 ± 8.6722e-01 | 4.0713e-01 ± 4.7135e-01 | 3.0264e-01 ± 3.0688e-01 | 3.3116e-01 ± 5.4839e-01 | 0.1339 ± 0.0031 |
