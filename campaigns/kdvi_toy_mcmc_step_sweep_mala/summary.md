# KDVI Toy MCMC Step-Size MALA Sweep Summary

Complete target-step groups: **41 / 49**.
Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.

## banana

### Winners

- **KL-ITE:** `KDVI-banana-mcmcstep5em2-mala` - 0.063482 +/- 0.002274
- **W2:** `KDVI-banana-mcmcstep5em3-mala` - 0.258346 +/- 0.016672

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `5e-2` | 0.063482 +/- 0.002274 | 0.372434 +/- 0.036747 |
| `5e-3` | 0.064017 +/- 0.006232 | 0.258346 +/- 0.016672 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | incomplete | 1,7 | 2.313130 +/- 3.213296 | 2779769137528832.000000 +/- 3931187214559435.500000 |
| `5e-2` | complete | 0,1,7 | 0.063482 +/- 0.002274 | 0.372434 +/- 0.036747 |
| `2e-2` | complete | 0,1,7 | 0.077067 +/- 0.007883 | 0.324365 +/- 0.006092 |
| `1e-2` | complete | 0,1,7 | 0.084745 +/- 0.012577 | 0.260100 +/- 0.012363 |
| `5e-3` | complete | 0,1,7 | 0.064017 +/- 0.006232 | 0.258346 +/- 0.016672 |
| `2e-3` | complete | 0,1,7 | 0.083710 +/- 0.013906 | 0.317796 +/- 0.031095 |
| `1e-3` | complete | 0,1,7 | 0.116440 +/- 0.008211 | 0.319385 +/- 0.018438 |

## x_shaped

### Winners

- **KL-ITE:** `KDVI-x_shaped-mcmcstep1em1-mala` - 0.007526 +/- 0.014989
- **W2:** `KDVI-x_shaped-mcmcstep5em2-mala` - 0.084430 +/- 0.015884

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `1e-1` | 0.007526 +/- 0.014989 | 0.090986 +/- 0.010013 |
| `5e-2` | 0.029157 +/- 0.014128 | 0.084430 +/- 0.015884 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.007526 +/- 0.014989 | 0.090986 +/- 0.010013 |
| `5e-2` | complete | 0,1,7 | 0.029157 +/- 0.014128 | 0.084430 +/- 0.015884 |
| `2e-2` | complete | 0,1,7 | 0.047837 +/- 0.014519 | 0.097244 +/- 0.014442 |
| `1e-2` | complete | 0,1,7 | 0.044945 +/- 0.028100 | 0.119373 +/- 0.021262 |
| `5e-3` | complete | 0,1,7 | 0.066986 +/- 0.005585 | 0.146530 +/- 0.008145 |
| `2e-3` | complete | 0,1,7 | 0.089566 +/- 0.007649 | 0.203868 +/- 0.027747 |
| `1e-3` | complete | 0,1,7 | 0.110411 +/- 0.011645 | 0.241189 +/- 0.033304 |

## multimodal

### Winners

- **KL-ITE:** `KDVI-multimodal-mcmcstep2em2-mala` - 0.010107 +/- 0.012267
- **W2:** `KDVI-multimodal-mcmcstep1em1-mala` - 0.058661 +/- 0.008350

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `2e-2` | 0.010107 +/- 0.012267 | 0.095150 +/- 0.001241 |
| `1e-1` | 0.010296 +/- 0.008941 | 0.058661 +/- 0.008350 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.010296 +/- 0.008941 | 0.058661 +/- 0.008350 |
| `5e-2` | complete | 0,1,7 | 0.022651 +/- 0.009206 | 0.068551 +/- 0.007633 |
| `2e-2` | complete | 0,1,7 | 0.010107 +/- 0.012267 | 0.095150 +/- 0.001241 |
| `1e-2` | complete | 0,1,7 | 0.037977 +/- 0.038113 | 0.107066 +/- 0.008243 |
| `5e-3` | complete | 0,1,7 | 0.046929 +/- 0.022674 | 0.134823 +/- 0.003646 |
| `2e-3` | complete | 0,1,7 | 0.087711 +/- 0.018598 | 0.172803 +/- 0.015164 |
| `1e-3` | complete | 0,1,7 | 0.101491 +/- 0.016706 | 0.203775 +/- 0.053598 |

## 8_gaussians

### Winners

- **KL-ITE:** `KDVI-8_gaussians-mcmcstep5em2-mala` - 0.064555 +/- 0.011356
- **W2:** `KDVI-8_gaussians-mcmcstep2em2-mala` - 0.089130 +/- 0.002427

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `5e-2` | 0.064555 +/- 0.011356 | 0.142684 +/- 0.073081 |
| `2e-2` | 0.065935 +/- 0.006088 | 0.089130 +/- 0.002427 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.077103 +/- 0.003226 | 0.124672 +/- 0.039892 |
| `5e-2` | complete | 0,1,7 | 0.064555 +/- 0.011356 | 0.142684 +/- 0.073081 |
| `2e-2` | complete | 0,1,7 | 0.065935 +/- 0.006088 | 0.089130 +/- 0.002427 |
| `1e-2` | complete | 0,1,7 | 0.087966 +/- 0.010507 | 0.182928 +/- 0.057516 |
| `5e-3` | complete | 0,1,7 | 0.130588 +/- 0.008597 | 0.135123 +/- 0.021257 |
| `2e-3` | complete | 0,1,7 | 0.226718 +/- 0.048869 | 0.192469 +/- 0.051691 |
| `1e-3` | complete | 0,1,7 | 0.294036 +/- 0.017678 | 0.231280 +/- 0.060634 |

## 8_gaussians_small

### Winners

- **KL-ITE:** `KDVI-8_gaussians_small-mcmcstep5em3-mala` - 0.099373 +/- 0.010561
- **W2:** `KDVI-8_gaussians_small-mcmcstep5em3-mala` - 0.075266 +/- 0.027310

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `5e-3` | 0.099373 +/- 0.010561 | 0.075266 +/- 0.027310 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 3.302254 +/- 0.023517 | 0.247434 +/- 0.002554 |
| `5e-2` | complete | 0,1,7 | 2.537384 +/- 0.053720 | 0.269703 +/- 0.000777 |
| `2e-2` | complete | 0,1,7 | 0.815647 +/- 0.574533 | 0.252945 +/- 0.181098 |
| `1e-2` | complete | 0,1,7 | 0.436988 +/- 0.601310 | 0.146890 +/- 0.177511 |
| `5e-3` | complete | 0,1,7 | 0.099373 +/- 0.010561 | 0.075266 +/- 0.027310 |
| `2e-3` | complete | 0,1,7 | 0.120066 +/- 0.008893 | 0.098923 +/- 0.007314 |
| `1e-3` | complete | 0,1,7 | 0.111562 +/- 0.005561 | 0.107019 +/- 0.029782 |

## student_uc

No complete step-size groups yet.

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | incomplete | none | - +/- - | - +/- - |
| `5e-2` | incomplete | none | - +/- - | - +/- - |
| `2e-2` | incomplete | none | - +/- - | - +/- - |
| `1e-2` | incomplete | none | - +/- - | - +/- - |
| `5e-3` | incomplete | none | - +/- - | - +/- - |
| `2e-3` | incomplete | none | - +/- - | - +/- - |
| `1e-3` | incomplete | none | - +/- - | - +/- - |

## Langevin_post

### Winners

- **KDE ELM:** `KDVI-Langevin_post-mcmcstep1em2-mala` - 74.350456 +/- 0.219364
- **W2:** `KDVI-Langevin_post-mcmcstep1em2-mala` - 0.014912 +/- 0.000923

### ELM/W2 Pareto Front

| Step size | KDE ELM mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `1e-2` | 74.350456 +/- 0.219364 | 0.014912 +/- 0.000923 |

### All Step Sizes

| Step size | Status | Seeds | KDE ELM | W2 |
|---:|---|---|---:|---:|
| `1e-2` | complete | 0,1,7 | 74.350456 +/- 0.219364 | 0.014912 +/- 0.000923 |
| `5e-3` | complete | 0,1,7 | 71.145177 +/- 0.353383 | 0.021296 +/- 0.000733 |
| `2e-3` | complete | 0,1,7 | 70.542903 +/- 1.562500 | 0.025514 +/- 0.008526 |
| `1e-3` | complete | 0,1,7 | 67.765195 +/- 5.146205 | 0.037110 +/- 0.019696 |
| `5e-4` | complete | 0,1,7 | 58.229720 +/- 13.296461 | 0.034550 +/- 0.006075 |
| `2e-4` | complete | 0,1,7 | 11.589353 +/- 9.994312 | 0.035966 +/- 0.001234 |
| `1e-4` | complete | 0,1,7 | 26.539283 +/- 6.322241 | 0.040290 +/- 0.003726 |

## Incomplete Groups

| Recipe | Complete seeds |
|---|---|
| `KDVI-banana-mcmcstep1em1-mala` | 1,7 |
| `KDVI-student_uc-mcmcstep1em1-mala` | none |
| `KDVI-student_uc-mcmcstep5em2-mala` | none |
| `KDVI-student_uc-mcmcstep2em2-mala` | none |
| `KDVI-student_uc-mcmcstep1em2-mala` | none |
| `KDVI-student_uc-mcmcstep5em3-mala` | none |
| `KDVI-student_uc-mcmcstep2em3-mala` | none |
| `KDVI-student_uc-mcmcstep1em3-mala` | none |
