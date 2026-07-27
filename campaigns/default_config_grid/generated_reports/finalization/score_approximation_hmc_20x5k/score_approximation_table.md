# Score-Approximation Analysis

All values are mean ± sample standard deviation across seeds 42, 43, 45, 49, 50.
The reference internal L2 is calculated across posterior-HMC chain means.

## Data-quality flags

- Native score unavailable: 8_gaussians / AISIVI / seed 42 / epoch 6000: RuntimeError: Failed to obtain finite samples from RealNVP after 3 attempts.
- Native score unavailable: 8_gaussians / AISIVI / seed 42 / epoch 8000: FloatingPointError: AISIVI reverse checkpoint contains 80 non-finite parameter tensors.
- Native score unavailable: 8_gaussians / AISIVI / seed 42 / epoch 10000: FloatingPointError: AISIVI reverse checkpoint contains 80 non-finite parameter tensors.
- Native score unavailable: 8_gaussians / AISIVI / seed 43 / epoch 10000: RuntimeError: Failed to obtain finite samples from RealNVP after 3 attempts.
- Native score unavailable: 8_gaussians / AISIVI / seed 49 / epoch 8000: RuntimeError: Failed to obtain finite samples from RealNVP after 3 attempts.
- Native score unavailable: 8_gaussians / AISIVI / seed 49 / epoch 10000: FloatingPointError: AISIVI reverse checkpoint contains 80 non-finite parameter tensors.
- HMC diagnostic warning: x_shaped / AISIVI / seed 42 / epoch 10000: hmc_post_burn_acceptance_min=0.0144 >= 0.05 failed
- HMC diagnostic warning: x_shaped / AISIVI / seed 49 / epoch 8000: hmc_post_burn_acceptance_min=0.035 >= 0.05 failed
- HMC diagnostic warning: 8_gaussians / AISIVI / seed 49 / epoch 8000: hmc_score_rhat_p95=1.93839 <= 1.1 failed; hmc_epsilon_rhat_p95=2.2251 <= 2 failed
- HMC diagnostic warning: 8_gaussians / AISIVI / seed 49 / epoch 10000: hmc_score_rhat_p95=1.95631 <= 1.1 failed; hmc_epsilon_rhat_p95=2.19405 <= 2 failed; hmc_post_burn_acceptance_min=0.0162 >= 0.05 failed

| Target | Method | Stage | Epoch | Method vs HMC q | Method vs target p | HMC q vs target p | HMC-chain internal L2 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 8_gaussians | AISIVI | 20% | 2000 | 4.4195e-01 ± 6.0245e-01 | 1.1351e+01 ± 1.2399e+00 | 1.1061e+01 ± 9.7559e-01 | 6.2270e-02 ± 2.1567e-02 |
| 8_gaussians | AISIVI | 40% | 4000 | 7.5258e-01 ± 3.7219e-01 | 5.8972e+00 ± 6.3608e-01 | 5.8419e+00 ± 3.8806e-01 | 1.6591e-01 ± 4.0891e-02 |
| 8_gaussians | AISIVI | 60% | 6000 | 1.1435e+00 ± 4.5730e-01 (n=4/5) | 4.3343e+00 ± 5.0595e-01 (n=4/5) | 8.9160e+01 ± 1.8893e+02 | 2.7349e-01 ± 5.8580e-02 |
| 8_gaussians | AISIVI | 80% | 8000 | 1.8768e+00 ± 4.6277e-01 (n=3/5) | 3.8962e+00 ± 7.2167e-01 (n=3/5) | 1.2468e+04 ± 2.7626e+04 | 2.8969e+00 ± 5.7939e+00 |
| 8_gaussians | AISIVI | 100% | 10000 | 1.0290e+00 ± 2.8144e-01 (n=2/5) | 3.4395e+00 ± 1.7303e-01 (n=2/5) | 1.3310e+04 ± 2.9368e+04 | 3.2394e+00 ± 6.5947e+00 |
| 8_gaussians | DSIVI | 20% | 2000 | 1.9057e+00 ± 2.4703e+00 | 1.0658e+01 ± 4.8302e-01 | 1.2642e+01 ± 2.3457e+00 | 2.5755e-01 ± 3.4336e-01 |
| 8_gaussians | DSIVI | 40% | 4000 | 1.3847e+00 ± 1.2497e+00 | 5.4881e+00 ± 6.8270e-01 | 6.8254e+00 ± 1.5578e+00 | 2.8753e-01 ± 2.2025e-01 |
| 8_gaussians | DSIVI | 60% | 6000 | 1.9971e+00 ± 1.4075e+00 | 3.8106e+00 ± 6.5196e-01 | 5.9375e+00 ± 1.7962e+00 | 4.3209e-01 ± 2.3909e-01 |
| 8_gaussians | DSIVI | 80% | 8000 | 1.4056e+00 ± 6.1565e-01 | 3.2520e+00 ± 7.6544e-01 | 4.8188e+00 ± 6.0601e-01 | 4.0553e-01 ± 1.2880e-01 |
| 8_gaussians | DSIVI | 100% | 10000 | 1.6082e+00 ± 5.7678e-01 | 2.6848e+00 ± 5.0945e-01 | 4.2667e+00 ± 4.5991e-01 | 4.0952e-01 ± 1.2193e-01 |
| 8_gaussians | SIVI | 20% | 2000 | 8.2152e-02 ± 5.4444e-02 | 1.1922e+01 ± 2.5652e-01 | 1.2102e+01 ± 2.2151e-01 | 4.0320e-02 ± 1.7594e-02 |
| 8_gaussians | SIVI | 40% | 4000 | 1.7820e-01 ± 1.0194e-01 | 6.7851e+00 ± 4.1076e-01 | 6.9123e+00 ± 4.6381e-01 | 7.7780e-02 ± 3.1508e-02 |
| 8_gaussians | SIVI | 60% | 6000 | 2.8439e-01 ± 1.4222e-01 | 5.1416e+00 ± 6.1410e-01 | 5.2009e+00 ± 5.5578e-01 | 1.5215e-01 ± 5.9584e-02 |
| 8_gaussians | SIVI | 80% | 8000 | 4.2073e-01 ± 1.9958e-01 | 4.4598e+00 ± 8.6434e-01 | 4.6437e+00 ± 8.9037e-01 | 2.4601e-01 ± 7.2580e-02 |
| 8_gaussians | SIVI | 100% | 10000 | 4.7358e-01 ± 3.4189e-01 | 4.0636e+00 ± 1.1050e+00 | 4.2749e+00 ± 1.0623e+00 | 2.8647e-01 ± 8.3612e-02 |
| 8_gaussians | UIVI | 20% | 2000 | 1.2129e+00 ± 1.9047e-01 | 1.8606e+01 ± 1.4514e+00 | 1.7539e+01 ± 1.4395e+00 | 6.0100e-03 ± 2.3395e-03 |
| 8_gaussians | UIVI | 40% | 4000 | 1.8516e+00 ± 5.2440e-01 | 1.4530e+01 ± 9.4827e-01 | 1.2580e+01 ± 7.7645e-01 | 6.6479e-03 ± 3.8950e-03 |
| 8_gaussians | UIVI | 60% | 6000 | 1.9941e+00 ± 3.4877e-01 | 1.3355e+01 ± 2.5854e-01 | 1.1209e+01 ± 1.5677e-01 | 7.9691e-03 ± 6.8051e-03 |
| 8_gaussians | UIVI | 80% | 8000 | 2.0773e+00 ± 5.2302e-01 | 1.3024e+01 ± 6.0651e-01 | 1.0911e+01 ± 6.4671e-01 | 8.7523e-03 ± 9.4835e-03 |
| 8_gaussians | UIVI | 100% | 10000 | 2.0322e+00 ± 6.0676e-01 | 1.2395e+01 ± 6.0138e-01 | 1.0829e+01 ± 3.2476e-01 | 7.8351e-03 ± 8.8915e-03 |
| x_shaped | AISIVI | 20% | 2000 | 1.7170e+03 ± 3.7534e+03 | 1.7048e+03 ± 3.7164e+03 | 1.9547e+01 ± 3.2506e+01 | 1.6263e+00 ± 1.1971e+00 |
| x_shaped | AISIVI | 40% | 4000 | 2.8041e+02 ± 3.6029e+02 | 2.8265e+02 ± 3.6217e+02 | 1.6434e+00 ± 1.0137e+00 | 2.5929e+00 ± 3.5727e+00 |
| x_shaped | AISIVI | 60% | 6000 | 5.7958e+02 ± 9.5887e+02 | 5.8107e+02 ± 9.6133e+02 | 9.0666e-01 ± 7.2667e-01 | 1.6143e+00 ± 1.2201e+00 |
| x_shaped | AISIVI | 80% | 8000 | 1.0016e+02 ± 6.2242e+01 | 1.0048e+02 ± 6.2395e+01 | 7.5231e-01 ± 5.5766e-01 | 1.0569e+00 ± 7.2873e-01 |
| x_shaped | AISIVI | 100% | 10000 | 8.7674e+01 ± 8.4645e+01 | 8.8381e+01 ± 8.6784e+01 | 8.8281e-01 ± 6.8386e-01 | 1.3078e+00 ± 8.7793e-01 |
| x_shaped | DSIVI | 20% | 2000 | 8.9904e-01 ± 9.0793e-01 | 3.6940e+00 ± 4.4813e-01 | 4.0004e+00 ± 1.3177e+00 | 1.2115e+00 ± 8.5442e-01 |
| x_shaped | DSIVI | 40% | 4000 | 8.0206e-01 ± 8.6722e-01 | 4.1110e-01 ± 1.0895e-01 | 8.7963e-01 ± 8.1374e-01 | 7.3456e-01 ± 7.3330e-01 |
| x_shaped | DSIVI | 60% | 6000 | 4.0713e-01 ± 4.7135e-01 | 1.4796e-01 ± 6.2954e-02 | 3.7226e-01 ± 4.7422e-01 | 6.4447e-01 ± 6.9737e-01 |
| x_shaped | DSIVI | 80% | 8000 | 3.0264e-01 ± 3.0688e-01 | 1.4205e-01 ± 8.5538e-02 | 2.4645e-01 ± 3.0152e-01 | 5.7692e-01 ± 5.2846e-01 |
| x_shaped | DSIVI | 100% | 10000 | 3.3116e-01 ± 5.4839e-01 | 7.9085e-02 ± 3.6935e-02 | 3.1789e-01 ± 5.4037e-01 | 6.2316e-01 ± 6.7189e-01 |
| x_shaped | SIVI | 20% | 2000 | 4.2888e-02 ± 6.5733e-02 | 4.4792e+00 ± 2.6528e+00 | 4.3995e+00 ± 2.5990e+00 | 8.7017e-03 ± 3.5715e-03 |
| x_shaped | SIVI | 40% | 4000 | 2.8119e-02 ± 2.1848e-02 | 5.4626e-01 ± 6.8398e-01 | 5.1372e-01 ± 6.4973e-01 | 1.8901e-02 ± 1.4726e-03 |
| x_shaped | SIVI | 60% | 6000 | 4.6126e-02 ± 6.2425e-02 | 2.9826e-01 ± 5.5440e-01 | 2.5094e-01 ± 4.8408e-01 | 2.4430e-02 ± 2.6735e-03 |
| x_shaped | SIVI | 80% | 8000 | 3.4660e-02 ± 4.0255e-02 | 2.7047e-01 ± 4.9510e-01 | 2.4006e-01 ± 4.6430e-01 | 2.4536e-02 ± 3.3398e-03 |
| x_shaped | SIVI | 100% | 10000 | 2.6443e-02 ± 1.9625e-02 | 1.9033e-01 ± 3.3950e-01 | 1.6540e-01 ± 3.2275e-01 | 2.3642e-02 ± 2.5883e-03 |
| x_shaped | UIVI | 20% | 2000 | 1.3263e+00 ± 2.9895e-01 | 5.0624e+00 ± 5.0470e-01 | 3.7796e+00 ± 4.4489e-01 | 1.1624e-02 ± 2.5604e-03 |
| x_shaped | UIVI | 40% | 4000 | 2.6242e+00 ± 1.1679e+00 | 3.2650e+00 ± 1.5119e+00 | 5.9695e-01 ± 3.2904e-01 | 2.2671e-02 ± 6.7250e-03 |
| x_shaped | UIVI | 60% | 6000 | 2.7486e+00 ± 3.2691e-01 | 3.0229e+00 ± 6.1812e-01 | 2.6528e-01 ± 3.2150e-01 | 3.3292e-02 ± 1.5610e-02 |
| x_shaped | UIVI | 80% | 8000 | 2.7677e+00 ± 3.8620e-01 | 3.0523e+00 ± 7.1733e-01 | 2.9761e-01 ± 3.2740e-01 | 3.1673e-02 ± 1.4050e-02 |
| x_shaped | UIVI | 100% | 10000 | 2.7615e+00 ± 5.3563e-01 | 3.0396e+00 ± 7.9649e-01 | 2.9405e-01 ± 3.7044e-01 | 3.2197e-02 ± 1.7754e-02 |
