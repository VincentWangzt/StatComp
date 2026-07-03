# KDVI Student UC MCMC Step-Size Sweep Summary

Complete MCMC-type/step groups: **14 / 14**.
Metrics are final logged values summarized as means and sample standard deviations across seeds 0, 1, and 7.

## Overall Winners

- **KL-ITE:** `KDVI-student_uc-mcmcstep1em2-mala` - 0.013221 +/- 0.007343
- **W2:** `KDVI-student_uc-mcmcstep1em1-sgld` - 0.579372 +/- 0.021746

## Overall KL/W2 Pareto Front

| Recipe | KL-ITE mean +/- std | W2 mean +/- std |
|---|---:|---:|
| `KDVI-student_uc-mcmcstep1em2-mala` | 0.013221 +/- 0.007343 | 0.651879 +/- 0.022305 |
| `KDVI-student_uc-mcmcstep1em1-mala` | 0.017883 +/- 0.013061 | 0.594562 +/- 0.006381 |
| `KDVI-student_uc-mcmcstep1em1-sgld` | 0.218372 +/- 0.020217 | 0.579372 +/- 0.021746 |

## SGLD

### Winners

- **KL-ITE:** `KDVI-student_uc-mcmcstep2em2-sgld` - 0.019936 +/- 0.013182
- **W2:** `KDVI-student_uc-mcmcstep1em1-sgld` - 0.579372 +/- 0.021746

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.218372 +/- 0.020217 | 0.579372 +/- 0.021746 |
| `5e-2` | complete | 0,1,7 | 0.064497 +/- 0.013146 | 0.615817 +/- 0.015826 |
| `2e-2` | complete | 0,1,7 | 0.019936 +/- 0.013182 | 0.621496 +/- 0.012723 |
| `1e-2` | complete | 0,1,7 | 0.032491 +/- 0.028221 | 0.647454 +/- 0.018026 |
| `5e-3` | complete | 0,1,7 | 0.032134 +/- 0.016593 | 0.661863 +/- 0.021219 |
| `2e-3` | complete | 0,1,7 | 0.044161 +/- 0.016505 | 0.674253 +/- 0.027169 |
| `1e-3` | complete | 0,1,7 | 0.043088 +/- 0.003988 | 0.690922 +/- 0.014061 |

## MALA

### Winners

- **KL-ITE:** `KDVI-student_uc-mcmcstep1em2-mala` - 0.013221 +/- 0.007343
- **W2:** `KDVI-student_uc-mcmcstep1em1-mala` - 0.594562 +/- 0.006381

### All Step Sizes

| Step size | Status | Seeds | KL-ITE | W2 |
|---:|---|---|---:|---:|
| `1e-1` | complete | 0,1,7 | 0.017883 +/- 0.013061 | 0.594562 +/- 0.006381 |
| `5e-2` | complete | 0,1,7 | 0.041170 +/- 0.017079 | 0.622775 +/- 0.020307 |
| `2e-2` | complete | 0,1,7 | 0.032167 +/- 0.011798 | 0.627515 +/- 0.034859 |
| `1e-2` | complete | 0,1,7 | 0.013221 +/- 0.007343 | 0.651879 +/- 0.022305 |
| `5e-3` | complete | 0,1,7 | 0.033089 +/- 0.004456 | 0.664060 +/- 0.018330 |
| `2e-3` | complete | 0,1,7 | 0.041991 +/- 0.023787 | 0.674843 +/- 0.020918 |
| `1e-3` | complete | 0,1,7 | 0.041891 +/- 0.015844 | 0.680538 +/- 0.013092 |

## Incomplete Groups

None.
