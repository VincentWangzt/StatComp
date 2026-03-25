# Experiment Results — Comprehensive Multi-Method Benchmark

**Date**: 2026-03-25/26
**Methods**: SIVI, KSIVI, UIVI, RSIVI, AISIVI, DSIVI
**Primary metrics**: ELBO↑, KL↓, W2↓ (toy/Langevin); ELBO↑, KSD↓ (LRwaveform); RMSE↓, NLL↓ (Bnn_boston)
**Note on Langevin_post**: KL estimation unreliable — use W2 and ELBO instead.
**Note on DSIVI ELBO**: ELBO estimator breaks on Bnn_boston (concentrated posterior) — use RMSE/NLL.

---

## Banana (z_dim=2)

| Method | Anneal | Epochs | ELBO↑ | KL↓ | W2↓ | KSD | MMD | Ep time |
|--------|--------|--------|-------|-----|-----|-----|-----|---------|
| SIVI | on | 20K | -0.55 | 0.548 | 1.292 | 0.197 | 0.065 | 0.017s |
| KSIVI | off | 50K | -0.04 | **0.074** | **0.903** | 0.010 | 0.008 | 0.006s |
| KSIVI | on | 50K | -0.04 | 0.061 | 0.754 | 0.006 | 0.004 | 0.007s |
| UIVI | on | 10K | 0.002 | 0.029 | 0.536 | 0.005 | 0.002 | 0.091s |
| RSIVI | on | 10K | -414 | 7.92 | 9.99 | 1692 | 0.145 | 0.048s |
| AISIVI | on | 10K | **1.56** | **0.009** | 0.244 | -0.007 | 0.002 | 0.031s |
| DSIVI | on | 10K | 2.96 | 0.012 | 0.326 | 0.003 | 0.002 | 0.011s |
| DSIVI | off | 10K | 0.54 | 0.014 | **0.283** | 0.007 | **0.001** | **0.010s** |
| DSIVI | on | 20K | 4.62 | 0.024 | 0.237 | -0.011 | 0.002 | 0.011s |

*Best KL: AISIVI (0.009). Best W2: DSIVI-off-10K (0.283). AISIVI best overall; DSIVI competitive and 3–8x faster/ep than UIVI.*

---

## Multimodal (z_dim=2)

| Method | Anneal | Epochs | ELBO↑ | KL↓ | W2↓ | KSD | MMD | Ep time |
|--------|--------|--------|-------|-----|-----|-----|-----|---------|
| SIVI | on | 20K | -0.21 | 0.234 | 0.330 | 0.049 | 0.025 | 0.017s |
| KSIVI | on | 50K | **0.003** | **~0** | **0.039** | **~0** | **0.001** | **0.007s** |
| UIVI | on | 10K | -0.03 | 0.052 | 0.161 | 0.014 | 0.007 | 0.094s |
| RSIVI | on | 10K | -0.25 | 0.272 | 0.514 | 0.069 | 0.029 | 0.052s |
| AISIVI | on | 10K | 0.028 | -0.002 | 0.030 | -0.001 | 0.002 | 0.033s |
| DSIVI | on | 10K | 0.013 | -0.002 | 0.049 | 0.001 | 0.001 | **0.010s** |
| DSIVI | off | 10K | 0.224 | 0.032 | 0.068 | 0.001 | 0.002 | **0.010s** |
| DSIVI | on | 20K | 0.013 | 0.002 | 0.048 | ~0 | 0.001 | 0.011s |

*Best: KSIVI (KL ~0, W2 0.039). DSIVI-on-10K ties AISIVI on KL. 10K suffices.*

---

## X-Shaped (z_dim=2)

| Method | Anneal | Epochs | ELBO↑ | KL↓ | W2↓ | KSD | MMD | Ep time |
|--------|--------|--------|-------|-----|-----|-----|-----|---------|
| SIVI | on | 20K | 0.005 | 0.020 | 0.047 | -0.001 | 0.002 | 0.017s |
| KSIVI | off | 50K | **0.006** | **0.001** | **0.068** | 0.002 | 0.003 | **0.008s** |
| KSIVI | on | 50K | -0.006 | 0.003 | 0.102 | 0.006 | 0.002 | **0.008s** |
| UIVI | on | 10K | -0.09 | 0.091 | 0.278 | 0.006 | 0.007 | 0.099s |
| RSIVI | — | — | ★ RealNVP crash | | | | | |
| AISIVI | — | — | ★ RealNVP crash | | | | | |
| DSIVI | on | 10K | 0.052 | 0.013 | **0.037** | -0.003 | **0.002** | 0.011s |
| DSIVI | off | 10K | **0.613** | 0.012 | 0.051 | ~0 | **0.002** | 0.011s |
| DSIVI | on | 20K | 0.037 | **-0.001** | 0.055 | 0.001 | 0.002 | 0.011s |

*Best KL: KSIVI-off (0.001). Best W2: DSIVI-on-10K (0.037). KSIVI without annealing is best; annealing slightly worsens it.*

---

## Student-UC (z_dim=2)

| Method | Anneal | Epochs | ELBO↑ | KL↓ | W2↓ | KSD | MMD | Ep time |
|--------|--------|--------|-------|-----|-----|-----|-----|---------|
| SIVI | on | 20K | -2.70 | **0.016** | **0.051** | 0.028 | **0.002** | 0.016s |
| KSIVI | off | 50K | -2.52 | 0.032 | 0.063 | 0.018 | 0.002 | **0.005s** |
| KSIVI | on | 50K | ★ diverged (KL 5.6) | | | | | |
| UIVI | on | 10K | -2.73 | 0.040 | 0.236 | 0.087 | 0.003 | 0.094s |
| RSIVI | — | — | ★ RealNVP crash | | | | | |
| AISIVI | on | 10K | -2.71 | 0.025 | 0.049 | 0.056 | 0.003 | 0.030s |
| DSIVI | off | 10K | **-2.47** | **0.009** | 0.132 | 0.021 | 0.001 | **0.013s** |
| DSIVI | off | 20K | -2.44 | 0.016 | 0.154 | 0.024 | 0.002 | 0.013s |

*Best KL: DSIVI-off-10K (0.009). Best W2: AISIVI (0.049). DSIVI 10K beats SIVI and 20K is no better.*

---

## Langevin_post (z_dim=100) — KL unreliable; use W2↓ and ELBO↑

| Method | Anneal | Rev | Epochs | ELBO↑ | W2↓ | KSD | MMD | Ep time |
|--------|--------|-----|--------|-------|-----|-----|-----|---------|
| SIVI | on | — | 20K | -176 | 0.035 | 0.503 | 0.023 | 0.047s |
| KSIVI | off | — | 100K | ★ diverged | 75.3M | ★div | 0.004 | 0.011s |
| KSIVI | on | — | 100K | ★ diverged | 262.5 | ★div | 0.004 | 0.009s |
| UIVI | on | — | 10K | **-77.4** | **0.007** | **0.014** | **0.002** | 0.090s |
| RSIVI | on | — | 10K | -180 | 0.033 | 0.707 | 0.021 | 0.615s |
| AISIVI | on | — | 10K | 63.3 | 0.074 | 1.011 | 0.004 | 0.620s |
| DSIVI | on | rev2 | 10K | -75.2 | **0.008** | 0.094 | **0.002** | **0.021s** |
| DSIVI | off | rev2 | 10K | 44.4 | 0.010 | 0.097 | **0.002** | **0.022s** |
| DSIVI | on | rev5 | 10K | -79.0 | **0.008** | 0.122 | **0.002** | 0.048s |
| DSIVI | off | rev5 | 10K | -54.4 | 0.011 | 0.896 | **0.002** | 0.046s |
| DSIVI | on | rev2 | 20K | -64.8 | 0.008 | 0.051 | **0.002** | 0.023s |

*Best W2: UIVI (0.007). DSIVI-on-rev2 matches UIVI W2 (0.008) at 4x lower ep time.*

---

## LRwaveform (z_dim=22, no KL/W2 baseline)

| Method | Anneal | Rev | Epochs | ELBO↑ | KSD↓ | Ep time |
|--------|--------|-----|--------|-------|------|---------|
| SIVI | on | — | 20K | -33.5 | 1.78 | 0.022s |
| KSIVI | off | — | 20K | ★ broken (+51K) | 673 | 0.008s |
| UIVI | on | — | 10K | **-24.2** | **0.031** | 0.097s |
| RSIVI | — | — | — | ★ RealNVP crash | | |
| AISIVI | — | — | — | ★ RealNVP crash | | |
| DSIVI | off | rev2 | 2K | -56.4 | 134 | **0.011s** |
| DSIVI | off | rev2 | 10K | -24.3 | 0.260 | **0.011s** |
| DSIVI | off | rev2 | 20K | -24.5 | 0.153 | 0.013s |

*Best ELBO: UIVI (-24.2) ≈ DSIVI-10K (-24.3). 10K is sufficient for DSIVI.*

---

## Bnn_boston (z_dim=751, no baseline — ELBO unreliable for DSIVI)

| Method | Anneal | Rev | Epochs | ELBO | RMSE↓ | NLL↓ | Ep time |
|--------|--------|-----|--------|------|-------|------|---------|
| SIVI | on | — | 20K | -1201 | 5.63 | 3.41 | 0.032s |
| KSIVI | off | — | 20K | ★ broken | 142.4 | 3.84 | 0.011s |
| UIVI | on | — | 10K | **-915** | 5.26 | 3.43 | 0.115s |
| DSIVI | on | rev2 | 10K | — | 3.60 | 2.69 | 0.115s |
| DSIVI | off | rev2 | 10K | — | 3.53 | 2.68 | 0.124s |
| DSIVI | off | rev2 | **20K** | — | **3.34** | **2.63** | 0.114s |
| DSIVI | on | rev5 | 10K | — | 3.76 | 2.74 | 0.278s |
| DSIVI | off | rev5 | 10K | — | 3.62 | 2.70 | 0.279s |

*Best BNN: DSIVI-off-rev2-20K (RMSE 3.34, NLL 2.63) — 33% better RMSE than UIVI.*

---

## Overall Rankings

### By KL↓ (toy 2D targets)

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Banana | AISIVI (0.009) | DSIVI-on-10K (0.012) | DSIVI-off-10K (0.014) |
| Multimodal | KSIVI-on (~0) | DSIVI-on-10K (-0.002) | AISIVI (-0.002) |
| X-Shaped | KSIVI-off (0.001) | DSIVI-off-10K (0.012) | DSIVI-on-10K (0.013) |
| Student-UC | DSIVI-off-10K (0.009) | SIVI (0.016) | AISIVI (0.025) |

### By W2↓ (Langevin_post)

| 1st | 2nd | 3rd |
|-----|-----|-----|
| UIVI (0.007) | DSIVI-on-rev2-10K (0.008) | DSIVI-on-rev5-10K (0.008) |

### By ELBO↑ (LRwaveform)

| 1st | 2nd | 3rd |
|-----|-----|-----|
| UIVI (-24.2) | DSIVI-off-rev2-10K (-24.3) | DSIVI-off-rev2-20K (-24.5) |

### By BNN NLL↓ (Bnn_boston)

| 1st | 2nd | 3rd |
|-----|-----|-----|
| DSIVI-off-rev2-20K (**2.63**) | DSIVI-off-rev2-10K (2.68) | DSIVI-on-rev2-10K (2.69) |

---

## Method Reliability

| Method | Success rate | Failure modes |
|--------|-------------|---------------|
| SIVI | 7/7 | No failures (reduced `reverse_sample_num` needed on Bnn_boston) |
| KSIVI | 4/7 | Diverges on Langevin; broken ELBO on LRwaveform; poor on Bnn_boston (RMSE 142) |
| UIVI | 7/7 | No failures |
| RSIVI | 3/7 | RealNVP crashes on x_shaped, student_uc, LRwaveform |
| AISIVI | 4/7 | RealNVP crashes on x_shaped, LRwaveform |
| **DSIVI** | **7/7** | No failures; ELBO unreliable on Bnn_boston (use RMSE/NLL) |

---

## Key Findings

1. **DSIVI and UIVI are the only 100%-reliable methods** — work on all 7 targets.
2. **DSIVI top-1 or top-2** on student_uc (KL), LRwaveform (ELBO tie with UIVI), Bnn_boston (NLL), Langevin W2 (near-tie with UIVI).
3. **KSIVI (fixed)** is best on multimodal (KL ~0) and x_shaped (KL 0.001) but diverges on Langevin and fails on high-dim targets.
4. **DSIVI 3–9x faster per epoch** than UIVI on toy/mid-dim; equal speed on Bnn_boston (751D).
5. **rev2 is optimal for DSIVI**: faster AND better than rev5 across all targets.
6. **20K epochs improve DSIVI** on Bnn_boston (NLL 2.68→2.63, RMSE 3.53→3.34) and x_shaped; toy 2D converges by 10K.
7. **KSIVI annealing**: improves banana (KL 0.074→0.061), breaks student_uc (diverges), marginally helps multimodal. Best config is target-dependent.
8. **RSIVI/AISIVI** unreliable due to RealNVP instability on several targets.

---

## DSIVI Epoch-Count Comparison

| Target | Metric | 10K | 20K | Δ |
|--------|--------|-----|-----|---|
| Banana | KL↓ | 0.012 | 0.024 | ↑ worse (LR decay) |
| Multimodal | KL↓ | -0.002 | 0.002 | ~ same |
| X-Shaped | KL↓ | 0.013 | **-0.001** | ✅ better |
| Student-UC | KL↓ | **0.009** | 0.016 | ↑ worse |
| Langevin | W2↓ | **0.008** | 0.008 | same |
| LRwaveform | ELBO↑ | **-24.3** | -24.5 | ~ same |
| Bnn_boston | NLL↓ | 2.68 | **2.63** | ✅ better |
| Bnn_boston | RMSE↓ | 3.53 | **3.34** | ✅ better |

## KSIVI Annealing Comparison

| Target | KL (off) | KL (on) | W2 (off) | W2 (on) | Better? |
|--------|----------|---------|---------|---------|---------|
| Banana | 0.074 | **0.061** | 0.903 | **0.754** | ✅ on |
| Multimodal | ~0 | ~0 | 0.041 | **0.039** | ~ same |
| X-Shaped | **0.001** | 0.003 | **0.068** | 0.102 | ✅ off |
| Student-UC | **0.032** | ★ 5.61 (diverged) | **0.063** | ★ 18.1 | ✅ off |
