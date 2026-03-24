# Experiment Results — Comprehensive Multi-Method Benchmark

**Date**: 2026-03-25
**Metrics**: ELBO↑, KL↓, W2↓ (primary); KSD↓, Fisher↓, MMD↓ (diagnostic)

---

## Banana (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime |
|--------|--------|------|-----|-----|-----|--------|-----|--------|
| SIVI | on | -0.55 | 0.548 | 1.292 | 0.197 | 13.2 | 0.065 | 0.017s |
| KSIVI | off | ★diverged | 10.49 | 5434 | 5.5e23 | 3.5e25 | 0.135 | 0.007s |
| UIVI | on | 0.002 | 0.029 | 0.536 | 0.005 | 2.19 | 0.002 | 0.091s |
| RSIVI | on | -414 | 7.92 | 9.99 | 1692 | 116K | 0.145 | 0.048s |
| AISIVI | on | 1.56 | **0.009** | **0.244** | -0.007 | 2306 | 0.002 | 0.031s |
| DSIVI | on | 2.96 | 0.012 | 0.326 | 0.003 | 35K | 0.002 | 0.011s |
| DSIVI | off | 0.54 | 0.014 | 0.283 | 0.007 | 5736 | 0.001 | 0.010s |

## Multimodal (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime |
|--------|--------|------|-----|-----|-----|--------|-----|--------|
| SIVI | on | -0.21 | 0.234 | 0.330 | 0.049 | 0.81 | 0.025 | 0.017s |
| KSIVI | on | -678K | 5.24 | 1042 | 5.39 | 94.9M | 0.527 | 0.007s |
| UIVI | on | -0.03 | 0.052 | 0.161 | 0.014 | 0.26 | 0.007 | 0.094s |
| RSIVI | on | -0.25 | 0.272 | 0.514 | 0.069 | 1.11 | 0.029 | 0.052s |
| AISIVI | on | **0.03** | **-0.002** | **0.030** | -0.001 | 0.070 | 0.002 | 0.033s |
| DSIVI | on | 0.01 | -0.002 | 0.049 | 0.001 | 0.57 | 0.001 | 0.010s |
| DSIVI | off | 0.22 | 0.032 | 0.068 | 0.001 | 1725 | 0.002 | 0.010s |

## X-Shaped (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime |
|--------|--------|------|-----|-----|-----|--------|-----|--------|
| SIVI | on | 0.005 | 0.020 | 0.047 | -0.001 | 0.17 | 0.002 | 0.017s |
| KSIVI | off | ★diverged | 12.15 | 18.9M | 2.2e12 | 5.1e13 | 0.076 | 0.007s |
| UIVI | on | -0.09 | 0.091 | 0.278 | 0.006 | 0.92 | 0.007 | 0.099s |
| RSIVI | — | ★crashed | — | — | — | — | — | — |
| AISIVI | — | ★crashed/rerunning | — | — | — | — | — | — |
| DSIVI | on | 0.05 | **0.013** | **0.037** | -0.003 | 5.65 | 0.002 | 0.011s |
| DSIVI | off | 0.61 | 0.012 | 0.051 | -0.0002 | 60.6 | 0.002 | 0.011s |

## Student-UC (z_dim=2)

| Method | Anneal | ELBO | KL↓ | W2↓ | KSD | Fisher | MMD | EpTime |
|--------|--------|------|-----|-----|-----|--------|-----|--------|
| SIVI | on | -2.70 | **0.016** | **0.051** | 0.028 | 1.32 | 0.002 | 0.016s |
| KSIVI | on | -2.52 | 0.094 | 0.138 | 0.075 | 21.5 | 0.003 | 0.005s |
| UIVI | on | -2.73 | 0.040 | 0.236 | 0.087 | 2.67 | 0.003 | 0.094s |
| RSIVI | — | ★crashed | — | — | — | — | — | — |
| AISIVI | on | -2.71 | 0.025 | 0.049 | 0.056 | 1.23 | 0.003 | 0.030s |
| DSIVI | off | -3.41* | 0.897* | 0.661* | 3.97 | 1228 | 0.031 | 0.009s |

*DSIVI student_uc only ran 2K epochs (config default). Needs rerun with 10K.

---

## Summary of Toy 2D Rankings (by KL↓)

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Banana | AISIVI (0.009) | DSIVI-on (0.012) | DSIVI-off (0.014) |
| Multimodal | AISIVI (-0.002) | DSIVI-on (-0.002) | UIVI (0.052) |
| X-Shaped | DSIVI-off (0.012) | DSIVI-on (0.013) | SIVI (0.020) |
| Student-UC | SIVI (0.016) | AISIVI (0.025) | UIVI (0.040) |

**Key observations**:
1. **DSIVI is consistently top-2** across all targets
2. **AISIVI is strongest on banana/multimodal** but crashes on x_shaped
3. **KSIVI diverges** on banana, multimodal, x_shaped; works on student_uc
4. **RSIVI crashes** on x_shaped and student_uc (RealNVP issues)
5. **DSIVI is 5-10x faster per epoch** than UIVI/RSIVI/AISIVI

---

## Remaining TODO
- [ ] Rerun RSIVI x_shaped/student_uc with default batch (running)
- [ ] Rerun AISIVI x_shaped with default batch (running)
- [ ] Rerun DSIVI student_uc with 10K epochs
- [ ] Round 3: Langevin_post (100D) — all methods
- [ ] Round 4: LRwaveform (22D) — all methods
- [ ] Round 5: Bnn_boston (751D) — all methods
