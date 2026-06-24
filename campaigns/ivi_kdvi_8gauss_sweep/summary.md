# IVI/KDVI 8-Gaussian Sweep

Updated: 2026-06-22T19:31:28.204838+00:00

Progress: **76/76** runs with a final KL ITE. Collapse is defined as **final KL ITE > 1**.
All setups use the checked-in 100k exact reference sample for their target. IVI uses a 32-dimensional latent input and KDVI uses a 32-dimensional epsilon input; all other training hyperparameters remain at the current IVI/KDVI defaults.

## Aggregate by setup

| Method slug | Target | Expected | Observed | Collapsed | Non-collapsed | Mean KL (non-collapsed) |
|---|---|---:|---:|---:|---:|---:|
| `ivi_default` | `8_gaussians` | 10 | 10 | 0 | 10 | 0.124283 |
| `ivi_default` | `8_gaussians_small` | 10 | 10 | 0 | 10 | 0.184770 |
| `kdvi_default` | `8_gaussians` | 10 | 10 | 0 | 10 | 0.099108 |
| `kdvi_default` | `8_gaussians_small` | 10 | 10 | 0 | 10 | 0.134919 |
| `kdvi_mala_mcmcsteps1_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.104066 |
| `kdvi_mala_mcmcsteps1_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.068833 |
| `kdvi_mala_mcmcsteps2_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.067962 |
| `kdvi_mala_mcmcsteps2_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.052114 |
| `kdvi_mala_mcmcsteps5_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.100585 |
| `kdvi_mala_mcmcsteps5_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.067495 |
| `kdvi_sgld_mcmcsteps1_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.144092 |
| `kdvi_sgld_mcmcsteps1_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.060808 |
| `kdvi_sgld_mcmcsteps2_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.131721 |
| `kdvi_sgld_mcmcsteps2_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.060270 |
| `kdvi_sgld_mcmcsteps5_stepsize0p50` | `8_gaussians` | 3 | 3 | 3 | 0 | — |
| `kdvi_sgld_mcmcsteps5_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.063524 |

## Per-run final KL ITE

| Run ID | Status | Final step | Final KL ITE | Collapsed | Metric source |
|---|---|---:|---:|:---:|---|
| `ivi_default__8_gaussians__seed00` | completed | 100000 | 0.126352 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed01` | completed | 100000 | 0.140058 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed02` | completed | 100000 | 0.115579 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed03` | completed | 100000 | 0.113693 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed04` | completed | 100000 | 0.127473 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed05` | completed | 100000 | 0.126990 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed06` | completed | 100000 | 0.103072 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed07` | completed | 100000 | 0.125085 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed08` | completed | 100000 | 0.144581 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed09` | completed | 100000 | 0.119949 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed00` | completed | 100000 | 0.202847 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed01` | completed | 100000 | 0.192893 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed02` | completed | 100000 | 0.217472 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed03` | completed | 100000 | 0.155199 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed04` | completed | 100000 | 0.218357 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed05` | completed | 100000 | 0.179565 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed06` | completed | 100000 | 0.142807 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed07` | completed | 100000 | 0.176453 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed08` | completed | 100000 | 0.174679 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed09` | completed | 100000 | 0.187431 | no | kl_ite.csv |
| `kdvi_default__8_gaussians__seed00` | completed | 100000 | 0.077444 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed01` | completed | 100000 | 0.128485 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed02` | completed | 100000 | 0.091468 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed03` | completed | 100000 | 0.086609 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed04` | completed | 100000 | 0.092315 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed05` | completed | 100000 | 0.086189 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed06` | completed | 100000 | 0.117198 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed07` | completed | 100000 | 0.092596 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed08` | completed | 100000 | 0.121398 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed09` | completed | 100000 | 0.097380 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed00` | completed | 100000 | 0.144575 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed01` | completed | 100000 | 0.131317 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed02` | completed | 100000 | 0.159295 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed03` | completed | 100000 | 0.124967 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed04` | completed | 100000 | 0.127091 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed05` | completed | 100000 | 0.110407 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed06` | completed | 100000 | 0.166827 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed07` | completed | 100000 | 0.139897 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed08` | completed | 100000 | 0.143636 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed09` | completed | 100000 | 0.101180 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.077413 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.123191 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.111593 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.042676 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.081925 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.081897 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.074536 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.075679 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.053672 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.081050 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.043817 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.031475 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.106587 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.072549 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.122618 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.055181 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.083505 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.063798 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.152308 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.118660 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.161308 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.066676 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.047278 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.068469 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.105550 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.140767 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.148847 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.070925 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.043551 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.066334 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 1.184805 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 1.209606 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 1.169399 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.062039 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.069700 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.058833 | no | tensorboard:metric/vi_model/kl_ite |
