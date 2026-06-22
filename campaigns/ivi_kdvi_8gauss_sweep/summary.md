# IVI/KDVI 8-Gaussian Sweep

Updated: 2026-06-22T13:00:12.786047+00:00

Progress: **76/76** runs with a final KL ITE. Collapse is defined as **final KL ITE > 1**.
All setups use the checked-in 100k exact reference sample for their target; the default IVI/KDVI comparison uses a 32-dimensional latent/epsilon input, and all other training hyperparameters remain at the current IVI/KDVI defaults. The KDVI MCMC ablations retain their original 2-dimensional epsilon input.

## Aggregate by setup

| Method slug | Target | Expected | Observed | Collapsed | Non-collapsed | Mean KL (non-collapsed) |
|---|---|---:|---:|---:|---:|---:|
| `ivi_default` | `8_gaussians` | 10 | 10 | 0 | 10 | 0.124283 |
| `ivi_default` | `8_gaussians_small` | 10 | 10 | 0 | 10 | 0.184770 |
| `kdvi_default` | `8_gaussians` | 10 | 10 | 0 | 10 | 0.100762 |
| `kdvi_default` | `8_gaussians_small` | 10 | 10 | 0 | 10 | 0.128864 |
| `kdvi_mala_mcmcsteps1_stepsize0p50` | `8_gaussians` | 3 | 3 | 1 | 2 | 0.534808 |
| `kdvi_mala_mcmcsteps1_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.696888 |
| `kdvi_mala_mcmcsteps2_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.685831 |
| `kdvi_mala_mcmcsteps2_stepsize0p10` | `8_gaussians` | 3 | 3 | 1 | 2 | 0.123891 |
| `kdvi_mala_mcmcsteps5_stepsize0p50` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.414266 |
| `kdvi_mala_mcmcsteps5_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.398117 |
| `kdvi_sgld_mcmcsteps1_stepsize0p50` | `8_gaussians` | 3 | 3 | 2 | 1 | 0.202278 |
| `kdvi_sgld_mcmcsteps1_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.106444 |
| `kdvi_sgld_mcmcsteps2_stepsize0p50` | `8_gaussians` | 3 | 3 | 3 | 0 | — |
| `kdvi_sgld_mcmcsteps2_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.354545 |
| `kdvi_sgld_mcmcsteps5_stepsize0p50` | `8_gaussians` | 3 | 3 | 3 | 0 | — |
| `kdvi_sgld_mcmcsteps5_stepsize0p10` | `8_gaussians` | 3 | 3 | 0 | 3 | 0.950386 |

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
| `kdvi_default__8_gaussians__seed00` | completed | 100000 | 0.071312 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed01` | completed | 100000 | 0.125387 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed02` | completed | 100000 | 0.086291 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed03` | completed | 100000 | 0.116127 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed04` | completed | 100000 | 0.086952 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed05` | completed | 100000 | 0.081017 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed06` | completed | 100000 | 0.120970 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed07` | completed | 100000 | 0.116255 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed08` | completed | 100000 | 0.090743 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed09` | completed | 100000 | 0.112568 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed00` | completed | 100000 | 0.129319 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed01` | completed | 100000 | 0.146687 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed02` | completed | 100000 | 0.131603 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed03` | completed | 100000 | 0.101494 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed04` | completed | 100000 | 0.121090 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed05` | completed | 100000 | 0.119159 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed06` | completed | 100000 | 0.155098 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed07` | completed | 100000 | 0.139808 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed08` | completed | 100000 | 0.146597 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed09` | completed | 100000 | 0.097783 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.982737 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 1.035457 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.086879 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.975568 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.142021 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps1_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.973076 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.977707 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.959060 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.120725 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 1.014015 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.123056 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps2_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.124725 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 0.144880 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 0.944409 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.153508 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.115369 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.964591 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_mala_mcmcsteps5_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.114390 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 1.100313 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 1.033512 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 0.202278 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.097553 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.102170 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps1_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.119609 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 1.118433 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 1.096458 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 1.068561 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.061815 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.914267 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps2_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.087553 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p50__8_gaussians__seed00` | completed | 100000 | 1.999820 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p50__8_gaussians__seed01` | completed | 100000 | 2.055434 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p50__8_gaussians__seed02` | completed | 100000 | 1.196228 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p10__8_gaussians__seed00` | completed | 100000 | 0.962325 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p10__8_gaussians__seed01` | completed | 100000 | 0.931268 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_sgld_mcmcsteps5_stepsize0p10__8_gaussians__seed02` | completed | 100000 | 0.957566 | no | tensorboard:metric/vi_model/kl_ite |
