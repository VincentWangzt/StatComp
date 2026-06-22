# IVI/KDVI 8-Gaussian Sweep

Updated: 2026-06-22T03:49:28.242465+00:00

Progress: **76/76** runs with a final KL ITE. Collapse is defined as **final KL ITE > 1**.
All setups use the checked-in 100k exact reference sample for their target; training hyperparameters remain at the current IVI/KDVI defaults.

## Aggregate by setup

| Method slug | Target | Expected | Observed | Collapsed | Non-collapsed | Mean KL (non-collapsed) |
|---|---|---:|---:|---:|---:|---:|
| `ivi_default` | `8_gaussians` | 10 | 10 | 5 | 5 | 0.152642 |
| `ivi_default` | `8_gaussians_small` | 10 | 10 | 5 | 5 | 0.203136 |
| `kdvi_default` | `8_gaussians` | 10 | 10 | 1 | 9 | 0.229587 |
| `kdvi_default` | `8_gaussians_small` | 10 | 10 | 6 | 4 | 0.167039 |
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
| `ivi_default__8_gaussians__seed00` | completed | 100000 | 0.176172 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed01` | completed | 100000 | 1.104971 | yes | kl_ite.csv |
| `ivi_default__8_gaussians__seed02` | completed | 100000 | 1.153110 | yes | kl_ite.csv |
| `ivi_default__8_gaussians__seed03` | completed | 100000 | 0.159296 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed04` | completed | 100000 | 0.158171 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed05` | completed | 100000 | 1.111891 | yes | kl_ite.csv |
| `ivi_default__8_gaussians__seed06` | completed | 100000 | 1.153502 | yes | kl_ite.csv |
| `ivi_default__8_gaussians__seed07` | completed | 100000 | 0.134095 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed08` | completed | 100000 | 0.135477 | no | kl_ite.csv |
| `ivi_default__8_gaussians__seed09` | completed | 100000 | 1.092814 | yes | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed00` | completed | 100000 | 1.335360 | yes | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed01` | completed | 100000 | 0.192412 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed02` | completed | 100000 | 0.214659 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed03` | completed | 100000 | 1.323364 | yes | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed04` | completed | 100000 | 1.322251 | yes | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed05` | completed | 100000 | 0.209310 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed06` | completed | 100000 | 1.365712 | yes | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed07` | completed | 100000 | 1.363537 | yes | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed08` | completed | 100000 | 0.180655 | no | kl_ite.csv |
| `ivi_default__8_gaussians_small__seed09` | completed | 100000 | 0.218642 | no | kl_ite.csv |
| `kdvi_default__8_gaussians__seed00` | completed | 100000 | 1.011827 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed01` | completed | 100000 | 0.986830 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed02` | completed | 100000 | 0.108936 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed03` | completed | 100000 | 0.135111 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed04` | completed | 100000 | 0.153541 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed05` | completed | 100000 | 0.124197 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed06` | completed | 100000 | 0.152391 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed07` | completed | 100000 | 0.140003 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed08` | completed | 100000 | 0.112298 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians__seed09` | completed | 100000 | 0.152977 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed00` | completed | 100000 | 0.142887 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed01` | completed | 100000 | 1.204120 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed02` | completed | 100000 | 0.205807 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed03` | completed | 100000 | 0.198138 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed04` | completed | 100000 | 1.161830 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed05` | completed | 100000 | 1.223947 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed06` | completed | 100000 | 1.125833 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed07` | completed | 100000 | 0.121322 | no | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed08` | completed | 100000 | 1.131071 | yes | tensorboard:metric/vi_model/kl_ite |
| `kdvi_default__8_gaussians_small__seed09` | completed | 100000 | 1.225810 | yes | tensorboard:metric/vi_model/kl_ite |
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
