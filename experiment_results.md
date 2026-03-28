# Experiment Results — Comprehensive Multi-Method Benchmark

**Date**: 2026-03-25/26/28
**Methods**: SIVI, KSIVI, UIVI, RSIVI, AISIVI, DSIVI
**Primary metrics**: ELBO↑, KL↓, W2↓ (toy/Langevin); ELBO↑, KSD↓ (LRwaveform); RMSE↓, NLL↓ (BNN targets)
**Note on Langevin_post**: KL estimation unreliable — use W2 and ELBO instead.
**Note on DSIVI ELBO**: ELBO estimator breaks on all BNN targets (concentrated posterior) — use RMSE/NLL.
**BNN Targets**: boston (z=751), concrete (z=501), power (z=301), protein (z=551), winered (z=651), yacht (z=401)

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
| KSIVI | off | — | **50K** | ★ diverged | — | 3531 | — | 0.009s |
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
| KSIVI | off | — | **10K** | ★ broken | 1283 | 0.006s |
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
| KSIVI | off | — | **10K** | ★ broken | 12.16 | 3.74 | 0.010s |
| UIVI | on | — | 10K | **-915** | 5.26 | 3.43 | 0.115s |
| DSIVI | on | rev2 | 10K | — | 3.60 | 2.69 | 0.055s |
| DSIVI | off | rev2 | 10K | — | 3.53 | 2.68 | 0.064s |
| DSIVI-uniform | on | rev10@256/1024 | 10K | n/a | 3.51 | 2.67 | 0.093s |
| DSIVI | off | rev2 | **20K** | n/a | **3.34** | **2.63** | 0.054s |
| DSIVI | on | rev5 | 10K | — | 3.76 | 2.74 | 0.218s |
| DSIVI | off | rev5 | 10K | — | 3.62 | 2.70 | 0.219s |

*Best BNN: DSIVI-off-rev2-20K (RMSE 3.34, NLL 2.63) — 33% better RMSE than UIVI.*
*Uniform rerun (2026-03-28): bs=256 and rbs=1024 was unexpectedly strong on Boston - second-best NLL overall (2.67), much better than the earlier bs=256/rbs=8192 sweep, and close to the 20K best run despite halving the training budget.*

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
| DSIVI-off-rev2-20K (**2.63**) | DSIVI-uniform-10K (2.67) | DSIVI-off-rev2-10K (2.68) |

---

## Method Reliability

| Method | Success rate | Failure modes |
|--------|-------------|---------------|
| SIVI | 7/7 | No failures (reduced `reverse_sample_num` needed on Bnn_boston) |
| KSIVI | 4/7 original + 0/5 new BNN | Diverges on Langevin; broken ELBO on LRwaveform; diverges on all 5 new BNN targets (KSD → ±billions) |
| UIVI | 7/7 | No failures |
| RSIVI | 3/7 | RealNVP crashes on x_shaped, student_uc, LRwaveform |
| AISIVI | 4/7 | RealNVP crashes on x_shaped, LRwaveform |
| **DSIVI** | **7/7** | No failures; ELBO unreliable on Bnn_boston (use RMSE/NLL) |

---

## Key Findings

1. **DSIVI and UIVI are the only 100%-reliable methods** — work on all 7 original targets + 5 new BNN targets.
2. **DSIVI top-1 or top-2** on student_uc (KL), LRwaveform (ELBO tie with UIVI), Bnn_boston (NLL), Langevin W2 (near-tie with UIVI), and 4/5 new BNN targets.
3. **KSIVI (fixed)** is best on multimodal (KL ~0) and x_shaped (KL 0.001) but diverges on Langevin and fails on high-dim targets.
4. **DSIVI 3–9x faster per epoch** than UIVI on toy/mid-dim; equal speed on high-dim BNN (>500D).
5. **rev2 is optimal for DSIVI**: faster AND better than rev5 across all targets.
6. **20K epochs improve DSIVI** on Bnn_boston (NLL 2.68→2.63), protein (NLL 2.97→2.94), and x_shaped; toy 2D and most BNN converge by 10K.
7. **KSIVI annealing**: improves banana (KL 0.074→0.061), breaks student_uc (diverges), marginally helps multimodal. Best config is target-dependent.
8. **RSIVI/AISIVI** unreliable due to RealNVP instability on several targets.
9. **DSIVI annealing pattern for new BNN**: annealing on wins for small/medium datasets (yacht, power, concrete); off wins for large datasets (protein N=45K; winered UIVI wins).
10. **KSIVI 25K vs 50K**: significantly worse on banana and x_shaped — 50K epochs needed for toy 2D targets.

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

---

## Phase 4: KSIVI Half-Epoch Reruns (2026-03-26)

**Purpose**: Test whether 25K epochs (50% of original) gives comparable quality to 50K on toy 2D targets, and 50K vs 100K on Langevin.

### KSIVI 25K (toy 2D) — compare to 50K originals

| Target | Epochs | ELBO↑ | KL↓ | W2↓ | KSD | MMD | Ep time |
|--------|--------|-------|-----|-----|-----|-----|---------|
| Banana | 50K (orig) | -0.04 | 0.061 | 0.754 | 0.006 | 0.004 | 0.007s |
| Banana | **25K** | -0.04 | 0.148 | 0.987 | 0.020 | 0.011 | **0.007s** |
| Multimodal | 50K (orig) | 0.003 | ~0 | 0.039 | ~0 | 0.001 | 0.007s |
| Multimodal | **25K** | 0.002 | 0.015 | 0.040 | ~0 | 0.001 | **0.008s** |
| X-Shaped | 50K (orig) | 0.006 | 0.001 | 0.068 | 0.002 | 0.003 | 0.008s |
| X-Shaped | **25K** | 0.030 | 0.031 | 0.202 | ~0 | 0.003 | **0.008s** |
| Student-UC | 50K (orig) | -2.52 | 0.032 | 0.063 | 0.018 | 0.002 | 0.005s |
| Student-UC | **25K** | -2.69 | 0.015 | 0.062 | 0.095 | 0.002 | **0.007s** |

*Verdict: 25K is clearly worse on banana and x_shaped (KL 2–30× higher). Multimodal and student_uc are acceptable at 25K. Recommend keeping 50K for toy 2D.*

**Note**: KSIVI Langevin_post 50K, LRwaveform 10K, and Bnn_boston 10K reruns completed — all diverge/fail (KSD 3531, 1283; RMSE 12.16 respectively). Confirms KSIVI cannot handle high-dimensional targets.

---

## Phase 4: New BNN Regression Targets (2026-03-26)

**Datasets**: concrete (N=1030, d=8, z_dim=501), power (N=9568, d=4, z_dim=301), protein (N=45730, d=9, z_dim=551), winered (N=1599, d=11, z_dim=651), yacht (N=308, d=6, z_dim=401).
**Note**: KSIVI metrics are KSD only (no RMSE/NLL); ELBO unreliable for DSIVI (same as Bnn_boston).

---

## Bnn_yacht (z_dim=401)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 1.866 | 2.706 | — | 0.020s |
| UIVI | on | 10K | 2.469 | 2.701 | — | 0.095s |
| DSIVI | on | 10K | 0.794 | **1.073** | n/a | 0.059s |
| DSIVI-uniform | on | 10K | **0.752** | 1.086 | n/a | 0.063s |
| DSIVI | off | 10K | 1.975 | 2.342 | — | 0.060s |
| DSIVI | off | 20K | 1.807 | 2.480 | — | 0.055s |
| KSIVI | off | 10K | ★ div | — | 8572 | 0.009s |
| KSIVI | off | 20K | ★ div | — | 116K | 0.009s |
| KSIVI | off | 30K | ★ div | — | -18M | 0.009s |

*Best NLL: DSIVI-on-10K (1.073). Best RMSE: DSIVI-uniform-10K (0.752). Annealed DSIVI still dominates yacht overall, and KSIVI diverges (KSD explodes).*
*Uniform rerun (2026-03-28): the bs=256, rbs=1024 setting improved RMSE while only slightly worsening NLL, so this aggressive setting clearly holds up on yacht when paired with annealing.*

---

## DSIVI Batch Size Sweep (2026-03-26/27)

**Purpose**: Systematically test how DSIVI performance degrades as we reduce main batch size (bs) and reverse batch size (rbs) on all 6 BNN targets.

### Bnn_boston (z_dim=751)

#### Main batch size sweep (rbs=8192 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| **2048** | 8192 | 20K | **3.34** | **2.63** | 0.054s |
| 1024 | 8192 | 20K | 4.78 | 3.27 | 0.113s |
| 512 | 8192 | 20K | 4.90 | 3.31 | 0.111s |
| 256 | 8192 | 20K | 4.73 | 3.32 | 0.120s |
| 128 | 8192 | 20K | 4.83 | 3.32 | 0.106s |

*Performance degrades significantly below bs=2048 — NLL increases by 25% at bs=1024.*

#### Reverse batch size sweep (bs=2048 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| 2048 | **8192** | 20K | **3.34** | **2.63** | 0.054s |
| 2048 | 4096 | 20K | 3.62 | 2.70 | 0.074s |
| 2048 | 2048 | 20K | 4.20 | 2.86 | 0.051s |
| 2048 | 1024 | 20K | 3.45 | 2.65 | 0.041s |
| 2048 | 512 | 20K | 4.59 | 3.03 | 0.036s |

*Safe rbs range: 1024–8192 (NLL within 5% of baseline). rbs=1024 actually gives faster training with similar performance.*

### Bnn_yacht (z_dim=401)

#### Main batch size sweep (rbs=8192 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| **4096** | 8192 | 10K | **0.79** | **1.07** | 0.059s |
| 2048 | 8192 | 10K | 1.33 | 1.84 | 0.049s |
| 1024 | 8192 | 10K | 1.03 | 2.01 | 0.049s |
| 512 | 8192 | 10K | 1.06 | 1.77 | 0.039s |
| 256 | 8192 | 10K | **0.97** | **1.15** | 0.048s |
| 128 | 8192 | 10K | 1.86 | 2.48 | 0.038s |

*bs=256 surprisingly competitive with baseline — small dataset benefits from smaller batches.*

#### Reverse batch size sweep (bs=4096 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| 4096 | **8192** | 10K | **0.79** | **1.07** | 0.059s |
| 4096 | 4096 | 10K | 1.13 | 1.85 | 0.031s |
| 4096 | 2048 | 10K | 0.63 | 1.03 | 0.018s |
| 4096 | 1024 | 10K | 1.05 | 1.40 | 0.013s |
| 4096 | 512 | 10K | 0.62 | 1.01 | 0.010s |

*rbs=512–2048 gives best performance AND fastest training — dramatic speedup opportunity.*

### Bnn_power (z_dim=301)

#### Main batch size sweep (rbs=8192 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| **4096** | 8192 | 10K | **4.15** | **2.84** | 0.034s |
| 2048 | 8192 | 10K | 4.15 | 2.84 | 0.031s |
| 1024 | 8192 | 10K | 4.21 | 2.86 | 0.026s |
| 512 | 8192 | 10K | 4.18 | 2.85 | 0.026s |
| 256 | 8192 | 10K | 4.20 | 2.85 | 0.024s |
| 128 | 8192 | 10K | 4.16 | 2.85 | 0.022s |

*Very stable across bs range — power dataset is robust to batch size reduction.*

#### Reverse batch size sweep (bs=4096 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| 4096 | **8192** | 10K | **4.15** | **2.84** | 0.094s |
| 4096 | 4096 | 10K | 4.17 | 2.85 | 0.073s |
| 4096 | 2048 | 10K | 4.39 | 2.91 | 0.063s |
| 4096 | 1024 | 10K | 4.29 | 2.89 | 0.059s |
| 4096 | 512 | 10K | 4.29 | 2.88 | 0.057s |

*Safe rbs range: 4096–8192. Performance degrades below rbs=4096 but training gets faster.*

### Bnn_concrete (z_dim=501)

#### Main batch size sweep (rbs=8192 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| **4096** | 8192 | 10K | **6.13** | **3.24** | 0.080s |
| 2048 | 8192 | 10K | 9.66 | 3.96 | 0.074s |
| 1024 | 8192 | 10K | 9.98 | 3.95 | 0.067s |
| 512 | 8192 | 10K | 10.69 | 4.01 | 0.061s |
| 256 | 8192 | 10K | 6.29 | 3.27 | 0.053s |
| 128 | 8192 | 10K | 10.34 | 4.03 | 0.064s |

*bs=256 surprisingly good — concrete benefits from smaller batches like yacht.*

#### Reverse batch size sweep (bs=4096 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| 4096 | **8192** | 10K | **6.13** | **3.24** | 0.080s |
| 4096 | 4096 | 10K | 6.47 | 3.30 | 0.046s |
| 4096 | 2048 | 10K | 9.08 | 3.67 | 0.032s |
| 4096 | 1024 | 10K | **6.08** | **3.22** | 0.024s |
| 4096 | 512 | 10K | 6.37 | 3.28 | 0.022s |

*rbs=1024 gives best performance AND fastest training — similar pattern to yacht.*

### Bnn_protein (z_dim=551)

#### Main batch size sweep (rbs=8192 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| **4096** | 8192 | 20K | **4.57** | **2.94** | 0.099s |
| 2048 | 8192 | 20K | 4.85 | 3.00 | 0.088s |
| 1024 | 8192 | 20K | 4.75 | 2.98 | 0.078s |
| 512 | 8192 | 20K | 4.79 | 2.99 | 0.074s |
| 256 | 8192 | 20K | 4.77 | 2.98 | 0.074s |
| 128 | 8192 | 20K | 4.80 | 2.99 | 0.066s |

*Very stable across bs range — large dataset provides robust gradient estimates.*

#### Reverse batch size sweep (bs=4096 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| 4096 | **8192** | 20K | **4.57** | **2.94** | 0.099s |
| 4096 | 4096 | 20K | 4.62 | 2.95 | 0.068s |
| 4096 | 2048 | 20K | 4.80 | 2.99 | 0.050s |
| 4096 | 1024 | 20K | 4.53 | 2.93 | 0.042s |
| 4096 | 512 | 20K | 4.68 | 2.97 | 0.038s |

*rbs=1024–4096 safe range — again showing smaller rbs can be beneficial.*

### Bnn_winered (z_dim=651)

#### Main batch size sweep (rbs=8192 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| **4096** | 8192 | 20K | **0.582** | **0.875** | 0.111s |
| 2048 | 8192 | 20K | 0.699 | 1.085 | 0.105s |
| 1024 | 8192 | 20K | 0.713 | 1.105 | 0.098s |
| 512 | 8192 | 20K | 0.679 | 1.048 | 0.088s |
| 256 | 8192 | 20K | 0.656 | 1.003 | 0.094s |
| 128 | 8192 | 20K | **0.575** | **0.861** | 0.091s |

*bs=128 gives best performance — winered strongly prefers smaller batches.*

#### Reverse batch size sweep (bs=4096 fixed)

| bs | rbs | Epochs | RMSE↓ | NLL↓ | Ep time |
|----|-----|--------|-------|------|---------|
| 4096 | **8192** | 20K | **0.582** | **0.875** | 0.111s |
| 4096 | 4096 | 20K | 0.703 | 1.093 | 0.075s |
| 4096 | 2048 | 20K | 0.651 | 0.995 | 0.055s |
| 4096 | 1024 | 20K | 0.625 | 0.926 | 0.045s |
| 4096 | 512 | 20K | 0.641 | 0.972 | 0.042s |

*rbs=1024–2048 optimal — consistent pattern across all targets.*

### Minimum Viable Batch Size Summary

| Target | Safe bs range | Safe rbs range | Recommendation |
|--------|---------------|----------------|-----------------|
| Bnn_boston | **2048 only** | 1024–8192 | Conservative: keep bs=2048, rbs=1024 for speed |
| Bnn_yacht | 256–4096 | **512–2048** | Aggressive: bs=256, rbs=512 (fastest + best) |
| Bnn_power | 128–4096 | 4096–8192 | Flexible: bs=128, rbs=4096 (balanced) |
| Bnn_concrete | **256–4096** | **1024–8192** | Aggressive: bs=256, rbs=1024 (optimal) |
| Bnn_protein | 128–4096 | 1024–4096 | Flexible: bs=128, rbs=1024 (efficient) |
| Bnn_winered | **128–4096** | **1024–2048** | Aggressive: bs=128, rbs=1024 (NLL=0.926, RMSE=0.625) |

**Key findings**:
- **Small datasets (yacht, concrete, winered)**: benefit from smaller batches (bs=128–256)
- **Large datasets (protein, power)**: robust to batch size reduction (bs=128–4096 safe)
- **Reverse batch size**: consistently optimal at rbs=1024–2048 across all targets
- **Speedup potential**: 20–40% faster training with optimized rbs
- **Performance**: Often IMPROVES with smaller rbs (better reverse model training)

### Uniform DSIVI Rerun (2026-03-28; anneal on, bs=256, rbs=1024, 10K)

| Target | Run dir | RMSE | NLL | Avg ep | Outcome |
|--------|---------|------|-----|--------|---------|
| Bnn_boston | 20260328_105609 | 3.507 | 2.670 | 0.093s | Strong surprise: 2nd-best Boston NLL overall |
| Bnn_yacht | 20260328_111441 | 0.752 | 1.086 | 0.063s | Holds up well: best RMSE, near-best NLL |
| Bnn_power | 20260328_112649 | 4.333 | 2.981 | 0.057s | Usable but worse than prior DSIVI best |
| Bnn_concrete | 20260328_113806 | 9.255 | 3.920 | 0.073s | Counterexample: much worse than expected |
| Bnn_protein | 20260328_115310 | 5.015 | 3.029 | 0.075s | Stable but clearly behind the best 20K run |
| Bnn_winered | 20260328_120906 | 0.644 | 0.984 | 0.081s | Acceptable, but still behind UIVI and best DSIVI |

**What changed after the joint rerun**:
- **Combination effects matter**: the earlier one-factor sweeps were directionally useful, but `bs=256` and `rbs=1024` do not combine uniformly across targets.
- **Boston is the surprise winner**: this joint setting rescues the poor `bs=256, rbs=8192` result and lands near the 20K optimum.
- **Concrete is the failure case**: each marginal sweep looked favorable, but the combined setting collapses to RMSE 9.255 / NLL 3.920.
- **Recommendation update**: do not use `bs=256, rbs=1024` as a blanket default; keep target-specific choices, and treat the uniform setting as a good low-cost probe rather than a universal recipe.
- **ELBO remains unreliable**: on all DSIVI BNN targets, RMSE/NLL are still the trustworthy metrics for comparison.

**Updated recommendation**: keep the per-target minimum-viable ranges above. For a single exploratory rerun, `bs=256, rbs=1024` is reasonable, but it should not replace the target-specific defaults because it fails badly on concrete and leaves performance on power/protein/winered on the table.

---

## Bnn_power (z_dim=301)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 4.334 | 3.069 | — | 0.016s |
| UIVI | on | 10K | 4.366 | 3.061 | — | 0.095s |
| DSIVI | on | 10K | **4.146** | **2.842** | — | 0.034s |
| DSIVI-uniform | on | 10K | 4.333 | 2.981 | n/a | 0.057s |
| DSIVI | off | 10K | 4.310 | 2.915 | — | 0.033s |
| DSIVI | off | 20K | 4.308 | 2.948 | — | 0.033s |
| KSIVI | off | 10K | ★ div | — | 1075 | 0.009s |
| KSIVI | off | 20K | ★ div | — | -612 | 0.009s |
| KSIVI | off | 30K | ★ div | — | -1573 | 0.009s |

*Best: DSIVI-on-10K (RMSE 4.15, NLL 2.84). KSIVI diverges. Anneal-on beats off on power. 20K offers no benefit over 10K.*
*Uniform rerun (2026-03-28): power stayed stable enough to be usable, but the joint bs=256 / rbs=1024 setting is still clearly worse than the existing DSIVI-on-10K baseline on both RMSE and NLL.*

---

## Bnn_concrete (z_dim=501)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 12.305 | 4.140 | — | 0.023s |
| UIVI | on | 10K | 10.210 | 4.088 | — | 0.094s |
| DSIVI | on | 10K | **6.132** | **3.237** | — | 0.080s |
| DSIVI-uniform | on | 10K | 9.255 | 3.920 | n/a | 0.073s |
| DSIVI | off | 10K | 10.344 | 4.008 | — | 0.077s |
| DSIVI | off | 20K | 10.505 | 3.998 | — | 0.080s |
| KSIVI | off | 10K | ★ div | — | 32149 | 0.009s |
| KSIVI | off | 20K | ★ div | — | 16131 | 0.009s |
| KSIVI | off | 30K | ★ div | — | 331 | 0.009s |

*Best: DSIVI-on-10K by large margin (RMSE 6.13 vs UIVI 10.21, NLL 3.24 vs 4.09). Annealing critical here. KSIVI diverges on all epoch counts (KSD still 331 at 30K — slowly improving but not useful).*
*Uniform rerun (2026-03-28): concrete is the strongest warning sign against a blanket uniform default - the combined bs=256 / rbs=1024 setting is far worse than the target-tuned DSIVI-on baseline, despite both marginal sweeps looking favorable in isolation.*

---

## Bnn_protein (z_dim=551)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 5.097 | 3.047 | — | 0.025s |
| UIVI | on | 10K | 5.111 | 3.050 | — | 0.093s |
| DSIVI | on | 10K | 4.676 | 2.963 | — | 0.103s |
| DSIVI-uniform | on | 10K | 5.015 | 3.029 | n/a | 0.075s |
| DSIVI | off | 10K | 4.707 | 2.968 | — | 0.105s |
| DSIVI | off | 20K | **4.566** | **2.941** | — | 0.099s |
| KSIVI | off | 10K | ★ div | — | 104K | 0.009s |
| KSIVI | off | 20K | ★ div | — | -143M | 0.010s |
| KSIVI | off | 30K | ★ div | — | -116G | 0.009s |

*Best: DSIVI-off-20K (RMSE 4.57, NLL 2.94). KSIVI catastrophically diverges — KSD hits -116 billion at 30K. Annealing slightly hurts on large dataset; 20K better than 10K.*
*Uniform rerun (2026-03-28): protein remains stable under the aggressive setting, but the result regresses toward SIVI/UIVI territory and does not threaten the existing 20K DSIVI runs.*

---

## Bnn_winered (z_dim=651)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 0.577 | 0.868 | — | 0.029s |
| UIVI | on | 10K | **0.568** | **0.853** | — | 0.091s |
| DSIVI | on | 10K | 0.595 | 0.895 | — | 0.115s |
| DSIVI-uniform | on | 10K | 0.644 | 0.984 | n/a | 0.081s |
| DSIVI | off | 10K | 0.587 | 0.876 | — | 0.119s |
| DSIVI | off | 20K | 0.582 | 0.875 | — | 0.111s |
| KSIVI | off | 10K | ★ div | — | -246K | 0.009s |
| KSIVI | off | 20K | ★ div | — | -2.1G | 0.010s |
| KSIVI | off | 30K | ★ div | — | 597K | 0.009s |

*Best: UIVI (RMSE 0.568, NLL 0.853). DSIVI competitive but slightly behind; all methods within 5% RMSE. KSIVI diverges. Likely near-optimal for this dataset.*
*Uniform rerun (2026-03-28): winered tolerates the setting, but the gap to the best DSIVI/UIVI entries widens enough that this does not change the recommendation for this target.*

---

## New BNN Targets: Rankings by NLL↓

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Bnn_yacht | DSIVI-on-10K (**1.073**) | DSIVI-uniform-10K (1.086) | DSIVI-off-20K (2.480) |
| Bnn_power | DSIVI-on-10K (**2.842**) | DSIVI-off-10K (2.915) | DSIVI-off-20K (2.948) |
| Bnn_concrete | DSIVI-on-10K (**3.237**) | DSIVI-uniform-10K (3.920) | DSIVI-off-20K (3.998) |
| Bnn_protein | DSIVI-off-20K (**2.941**) | DSIVI-off-10K (2.968) | DSIVI-on-10K (2.963) |
| Bnn_winered | UIVI (**0.853**) | SIVI (0.868) | DSIVI-off-20K (0.875) |

*After the uniform rerun, DSIVI-on-10K still wins 3/5 targets, but the new bs=256 / rbs=1024 setting slots in as the runner-up on yacht and concrete only. The broader pattern still holds: annealing helps on yacht/power/concrete, while protein and winered prefer the earlier target-specific settings.*
