# DSIVI Toy VI Architecture Comparison

Complete target-architecture groups: **12 / 12**.
Metrics are final-value means and sample standard deviations across seeds 0, 1, and 7.

| Target | VI architecture | ELBO | KL-ITE | W2 | Seeds |
|---|---|---:|---:|---:|---:|
| `banana` | `default` | -0.026657 +/- 0.017044 | -0.000436 +/- 0.010717 | 0.113730 +/- 0.070547 | 3 |
| `banana` | `eps32_elu_logstd` | -0.044758 +/- 0.013770 | 0.012543 +/- 0.006248 | 0.177744 +/- 0.064554 | 3 |
| `x_shaped` | `default` | -0.030102 +/- 0.013220 | 0.006451 +/- 0.012059 | 0.077133 +/- 0.029111 | 3 |
| `x_shaped` | `eps32_elu_logstd` | -0.011335 +/- 0.001581 | 0.005708 +/- 0.012420 | 0.057824 +/- 0.005260 | 3 |
| `multimodal` | `default` | -0.000982 +/- 0.000340 | -0.004763 +/- 0.012609 | 0.027412 +/- 0.004044 | 3 |
| `multimodal` | `eps32_elu_logstd` | -0.007322 +/- 0.002718 | 0.007455 +/- 0.006885 | 0.054292 +/- 0.006729 | 3 |
| `8_gaussians` | `default` | -0.235997 +/- 0.149142 | 0.142106 +/- 0.074252 | 0.239817 +/- 0.092143 | 3 |
| `8_gaussians` | `eps32_elu_logstd` | -0.350356 +/- 0.094201 | 0.173952 +/- 0.103204 | 0.241137 +/- 0.041928 | 3 |
| `8_gaussians_small` | `default` | -0.764995 +/- 0.147703 | 0.558255 +/- 0.183958 | 0.263447 +/- 0.041065 | 3 |
| `8_gaussians_small` | `eps32_elu_logstd` | -3.825300 +/- 4.505435 | 1.048712 +/- 0.324973 | 0.296516 +/- 0.193279 | 3 |
| `student_uc` | `default` | -2.683310 +/- 0.002623 | 0.005607 +/- 0.008702 | 0.551579 +/- 0.005278 | 3 |
| `student_uc` | `eps32_elu_logstd` | -2.724949 +/- 0.049683 | 0.062320 +/- 0.030308 | 0.649926 +/- 0.040805 | 3 |

## Incomplete Groups

None.
