# KDVI Toy MCMC Step-Size Sweep Summary

Complete target-step groups: **41 / 49**.
Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.

## banana

### Winners

- **KL-ITE:** `KDVI-banana-mcmcstep2em2` - 0.046878 +/- 0.017134
- **W2:** `KDVI-banana-mcmcstep2em3` - 0.278187 +/- 0.022734

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `2e-2` | 0.046878 +/- 0.017134 | 0.490339 +/- 0.055512 |
| `1e-2` | 0.061670 +/- 0.024723 | 0.319870 +/- 0.039275 |
| `5e-3` | 0.062349 +/- 0.003909 | 0.281522 +/- 0.018038 |
| `2e-3` | 0.111456 +/- 0.004447 | 0.278187 +/- 0.022734 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 1.513071 +/- 1.425807 | 10828.706365 +/- 18751.991147 |
| `5e-2` | complete | 0,1,7 | 0.730663 +/- 0.857712 | 603022.961164 +/- 1044464.667462 |
| `2e-2` | complete | 0,1,7 | 0.046878 +/- 0.017134 | 0.490339 +/- 0.055512 |
| `1e-2` | complete | 0,1,7 | 0.061670 +/- 0.024723 | 0.319870 +/- 0.039275 |
| `5e-3` | complete | 0,1,7 | 0.062349 +/- 0.003909 | 0.281522 +/- 0.018038 |
| `2e-3` | complete | 0,1,7 | 0.111456 +/- 0.004447 | 0.278187 +/- 0.022734 |
| `1e-3` | complete | 0,1,7 | 0.089426 +/- 0.000719 | 0.340045 +/- 0.077699 |

## x_shaped

### Winners

- **KL-ITE:** `KDVI-x_shaped-mcmcstep1em1` - 0.009474 +/- 0.013005
- **W2:** `KDVI-x_shaped-mcmcstep1em1` - 0.077865 +/- 0.025749

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `1e-1` | 0.009474 +/- 0.013005 | 0.077865 +/- 0.025749 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.009474 +/- 0.013005 | 0.077865 +/- 0.025749 |
| `5e-2` | complete | 0,1,7 | 0.013671 +/- 0.028412 | 0.097252 +/- 0.004307 |
| `2e-2` | complete | 0,1,7 | 0.033616 +/- 0.005959 | 0.108363 +/- 0.019030 |
| `1e-2` | complete | 0,1,7 | 0.047695 +/- 0.018002 | 0.118488 +/- 0.019662 |
| `5e-3` | complete | 0,1,7 | 0.071684 +/- 0.027444 | 0.161738 +/- 0.037089 |
| `2e-3` | complete | 0,1,7 | 0.095348 +/- 0.008418 | 0.196817 +/- 0.020485 |
| `1e-3` | complete | 0,1,7 | 0.110219 +/- 0.011945 | 0.264421 +/- 0.006101 |

## multimodal

### Winners

- **KL-ITE:** `KDVI-multimodal-mcmcstep1em1` - 0.008164 +/- 0.009587
- **W2:** `KDVI-multimodal-mcmcstep1em1` - 0.054765 +/- 0.004785

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `1e-1` | 0.008164 +/- 0.009587 | 0.054765 +/- 0.004785 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.008164 +/- 0.009587 | 0.054765 +/- 0.004785 |
| `5e-2` | complete | 0,1,7 | 0.012311 +/- 0.010795 | 0.074305 +/- 0.013340 |
| `2e-2` | complete | 0,1,7 | 0.029443 +/- 0.012477 | 0.091821 +/- 0.007167 |
| `1e-2` | complete | 0,1,7 | 0.035979 +/- 0.017877 | 0.111840 +/- 0.028316 |
| `5e-3` | complete | 0,1,7 | 0.045832 +/- 0.028322 | 0.125896 +/- 0.023998 |
| `2e-3` | complete | 0,1,7 | 0.077046 +/- 0.006076 | 0.181723 +/- 0.034036 |
| `1e-3` | complete | 0,1,7 | 0.089931 +/- 0.008746 | 0.191276 +/- 0.011266 |

## 8_gaussians

### Winners

- **KL-ITE:** `KDVI-8_gaussians-mcmcstep5em2` - 0.053259 +/- 0.014683
- **W2:** `KDVI-8_gaussians-mcmcstep5em2` - 0.095087 +/- 0.013721

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `5e-2` | 0.053259 +/- 0.014683 | 0.095087 +/- 0.013721 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.060949 +/- 0.015839 | 0.130806 +/- 0.026725 |
| `5e-2` | complete | 0,1,7 | 0.053259 +/- 0.014683 | 0.095087 +/- 0.013721 |
| `2e-2` | complete | 0,1,7 | 0.075633 +/- 0.016117 | 0.138288 +/- 0.049588 |
| `1e-2` | complete | 0,1,7 | 0.099436 +/- 0.013803 | 0.138147 +/- 0.020267 |
| `5e-3` | complete | 0,1,7 | 0.122005 +/- 0.033188 | 0.132723 +/- 0.035781 |
| `2e-3` | complete | 0,1,7 | 0.189543 +/- 0.024764 | 0.284953 +/- 0.090783 |
| `1e-3` | complete | 0,1,7 | 0.406918 +/- 0.283438 | 0.766269 +/- 0.889654 |

## 8_gaussians_small

### Winners

- **KL-ITE:** `KDVI-8_gaussians_small-mcmcstep5em3` - 0.113294 +/- 0.013390
- **W2:** `KDVI-8_gaussians_small-mcmcstep5em3` - 0.082450 +/- 0.029462

### KL/W2 Pareto Front

| Step size | KL-ITE mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `5e-3` | 0.113294 +/- 0.013390 | 0.082450 +/- 0.029462 |

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | incomplete | none | - +/- - | - +/- - |
| `5e-2` | complete | 0,1,7 | 2.569511 +/- 0.056189 | 0.302774 +/- 0.002893 |
| `2e-2` | complete | 0,1,7 | 1.337853 +/- 0.013267 | 0.214682 +/- 0.008541 |
| `1e-2` | complete | 0,1,7 | 0.737560 +/- 0.556189 | 0.210558 +/- 0.149112 |
| `5e-3` | complete | 0,1,7 | 0.113294 +/- 0.013390 | 0.082450 +/- 0.029462 |
| `2e-3` | complete | 0,1,7 | 0.115295 +/- 0.007866 | 0.103411 +/- 0.035776 |
| `1e-3` | complete | 0,1,7 | 0.132261 +/- 0.012385 | 0.120839 +/- 0.040096 |

## student_uc

No complete step-size groups yet.

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | incomplete | 1 | 5.546073 +/- - | 4713971957366784.000000 +/- - |
| `5e-2` | incomplete | none | - +/- - | - +/- - |
| `2e-2` | incomplete | none | - +/- - | - +/- - |
| `1e-2` | incomplete | none | - +/- - | - +/- - |
| `5e-3` | incomplete | none | - +/- - | - +/- - |
| `2e-3` | incomplete | none | - +/- - | - +/- - |
| `1e-3` | incomplete | none | - +/- - | - +/- - |

## Langevin_post

### Winners

- **KDE ELM:** `KDVI-Langevin_post-mcmcstep5em3` - 71.917399 +/- 0.053418
- **W2:** `KDVI-Langevin_post-mcmcstep5em3` - 0.024812 +/- 0.000473

### ELM/W2 Pareto Front

| Step size | KDE ELM mean +/- std | W2 mean +/- std |
|---:|---:|---:|
| `5e-3` | 71.917399 +/- 0.053418 | 0.024812 +/- 0.000473 |

### All Step Sizes

| Step size | Status | Seeds | KDE ELM | W2 |
|---:|---|---|---:|---:|
| `1e-2` | complete | 0,1,7 | -73.457242 +/- 3.721244 | 1.315362 +/- 0.067206 |
| `5e-3` | complete | 0,1,7 | 71.917399 +/- 0.053418 | 0.024812 +/- 0.000473 |
| `2e-3` | complete | 0,1,7 | 70.236722 +/- 0.177456 | 0.031375 +/- 0.001209 |
| `1e-3` | complete | 0,1,7 | 68.871712 +/- 0.144105 | 0.036622 +/- 0.001087 |
| `5e-4` | complete | 0,1,7 | 67.179342 +/- 0.071153 | 0.041945 +/- 0.000351 |
| `2e-4` | complete | 0,1,7 | 65.999809 +/- 0.290106 | 0.045527 +/- 0.001067 |
| `1e-4` | complete | 0,1,7 | 65.713371 +/- 0.298086 | 0.046039 +/- 0.001352 |

## Incomplete Groups

| Recipe | Complete seeds |
|---|---|
| `KDVI-8_gaussians_small-mcmcstep1em1` | none |
| `KDVI-student_uc-mcmcstep1em1` | 1 |
| `KDVI-student_uc-mcmcstep5em2` | none |
| `KDVI-student_uc-mcmcstep2em2` | none |
| `KDVI-student_uc-mcmcstep1em2` | none |
| `KDVI-student_uc-mcmcstep5em3` | none |
| `KDVI-student_uc-mcmcstep2em3` | none |
| `KDVI-student_uc-mcmcstep1em3` | none |
