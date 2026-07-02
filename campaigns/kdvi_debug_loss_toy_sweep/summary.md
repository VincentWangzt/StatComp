# KDVI Debug Loss Toy Sweep Summary

Complete target/loss groups: **17 / 21**.
Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.

## banana

### Winners

- **KL-ITE:** `KDVI-banana-mmd` - 0.043188 +/- 0.006745
- **W2:** `KDVI-banana-mmd_per_dim` - 0.489427 +/- 0.046808

### KL/W2 Pareto Front

| Loss | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `mmd` | 0.043188 +/- 0.006745 | 0.490339 +/- 0.055512 |
| `mmd_per_dim` | 0.053435 +/- 0.026617 | 0.489427 +/- 0.046808 |

### All Losses

| Loss | Status | Seeds | KL-ITE | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | complete | 0,1,7 | 0.043188 +/- 0.006745 | 0.490339 +/- 0.055512 | - +/- - |
| `mmd_per_dim` | complete | 0,1,7 | 0.053435 +/- 0.026617 | 0.489427 +/- 0.046808 | - +/- - |
| `l2` | complete | 0,1,7 | 5.161542 +/- 2.928333 | 2301925273528032.000000 +/- 2069515921964508.250000 | - +/- - |

## x_shaped

### Winners

- **KL-ITE:** `KDVI-x_shaped-mmd` - 0.011793 +/- 0.012533
- **W2:** `KDVI-x_shaped-mmd` - 0.077865 +/- 0.025749

### KL/W2 Pareto Front

| Loss | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `mmd` | 0.011793 +/- 0.012533 | 0.077865 +/- 0.025749 |

### All Losses

| Loss | Status | Seeds | KL-ITE | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | complete | 0,1,7 | 0.011793 +/- 0.012533 | 0.077865 +/- 0.025749 | - +/- - |
| `mmd_per_dim` | complete | 0,1,7 | 0.021319 +/- 0.019895 | 0.106818 +/- 0.005562 | - +/- - |
| `l2` | complete | 0,1,7 | 4.903013 +/- 0.137853 | 1.361264 +/- 0.000482 | - +/- - |

## multimodal

### Winners

- **KL-ITE:** `KDVI-multimodal-mmd_per_dim` - 0.010917 +/- 0.006663
- **W2:** `KDVI-multimodal-mmd` - 0.054765 +/- 0.004785

### KL/W2 Pareto Front

| Loss | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `mmd_per_dim` | 0.010917 +/- 0.006663 | 0.072354 +/- 0.021666 |
| `mmd` | 0.017919 +/- 0.012652 | 0.054765 +/- 0.004785 |

### All Losses

| Loss | Status | Seeds | KL-ITE | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | complete | 0,1,7 | 0.017919 +/- 0.012652 | 0.054765 +/- 0.004785 | - +/- - |
| `mmd_per_dim` | complete | 0,1,7 | 0.010917 +/- 0.006663 | 0.072354 +/- 0.021666 | - +/- - |
| `l2` | incomplete | 1,7 | 5.690197 +/- 0.061634 | 2.241159 +/- 0.035526 | - +/- - |

## 8_gaussians

### Winners

- **KL-ITE:** `KDVI-8_gaussians-mmd` - 0.049521 +/- 0.005434
- **W2:** `KDVI-8_gaussians-mmd` - 0.095087 +/- 0.013721

### KL/W2 Pareto Front

| Loss | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `mmd` | 0.049521 +/- 0.005434 | 0.095087 +/- 0.013721 |

### All Losses

| Loss | Status | Seeds | KL-ITE | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | complete | 0,1,7 | 0.049521 +/- 0.005434 | 0.095087 +/- 0.013721 | - +/- - |
| `mmd_per_dim` | complete | 0,1,7 | 0.051857 +/- 0.034869 | 0.145979 +/- 0.056226 | - +/- - |
| `l2` | complete | 0,1,7 | 5.819778 +/- 0.260549 | 3.983836 +/- 0.088087 | - +/- - |

## 8_gaussians_small

### Winners

- **KL-ITE:** `KDVI-8_gaussians_small-mmd_per_dim` - 0.112167 +/- 0.008268
- **W2:** `KDVI-8_gaussians_small-mmd` - 0.082450 +/- 0.029462

### KL/W2 Pareto Front

| Loss | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `mmd_per_dim` | 0.112167 +/- 0.008268 | 0.083379 +/- 0.027510 |
| `mmd` | 0.117758 +/- 0.014101 | 0.082450 +/- 0.029462 |

### All Losses

| Loss | Status | Seeds | KL-ITE | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | complete | 0,1,7 | 0.117758 +/- 0.014101 | 0.082450 +/- 0.029462 | - +/- - |
| `mmd_per_dim` | complete | 0,1,7 | 0.112167 +/- 0.008268 | 0.083379 +/- 0.027510 | - +/- - |
| `l2` | complete | 0,1,7 | 4.203702 +/- 2.532963 | 26564431446016.656250 +/- 46010944938679.523438 | - +/- - |

## student_uc

No complete loss groups yet.

### All Losses

| Loss | Status | Seeds | KL-ITE | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | incomplete | none | - +/- - | - +/- - | - +/- - |
| `mmd_per_dim` | incomplete | none | - +/- - | - +/- - | - +/- - |
| `l2` | incomplete | 1,7 | 2.656875 +/- 0.029125 | 0.971490 +/- 0.031006 | - +/- - |

## Langevin_post

### Winners

- **KDE ELM:** `KDVI-Langevin_post-mmd` - 74.870326 +/- 0.125467
- **W2:** `KDVI-Langevin_post-mmd` - 0.014249 +/- 0.002006

### ELM/W2 Pareto Front

| Loss | KDE ELM mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `mmd` | 74.870326 +/- 0.125467 | 0.014249 +/- 0.002006 |

### All Losses

| Loss | Status | Seeds | KDE ELM | W2 | Train loss |
|---|---|---|---:|---:|---:|
| `mmd` | complete | 0,1,7 | 74.870326 +/- 0.125467 | 0.014249 +/- 0.002006 | - +/- - |
| `mmd_per_dim` | complete | 0,1,7 | -66581.752604 +/- 15860.387082 | 717203989.333333 +/- 102666919.399355 | - +/- - |
| `l2` | complete | 0,1,7 | -279707.760417 +/- 619.710004 | 0.105740 +/- 0.001569 | - +/- - |

## Incomplete Groups

| Recipe | Complete seeds |
|---|---|
| `KDVI-multimodal-l2` | 1,7 |
| `KDVI-student_uc-mmd` | none |
| `KDVI-student_uc-mmd_per_dim` | none |
| `KDVI-student_uc-l2` | 1,7 |
