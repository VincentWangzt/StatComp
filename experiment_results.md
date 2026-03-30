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
| DSIVI-grid (4096/2048) | on | rev2 | 10K | n/a | **2.44** | **2.43** | 0.089s |
| DSIVI | off | rev2 | **20K** | n/a | 3.34 | 2.63 | 0.054s |
| DSIVI | on | rev5 | 10K | — | 3.76 | 2.74 | 0.218s |
| DSIVI | off | rev5 | 10K | — | 3.62 | 2.70 | 0.219s |

*Best BNN: DSIVI-grid (4096/2048) reaches RMSE 2.44 and NLL 2.43, decisively beating the older 20K Boston run and the earlier 10K comparisons.*
*The completed rev2 grid is the important correction here: Boston is no longer the fragile outlier from the earlier partial sweep, and several tuned 10K points land well below the old `2.63` NLL mark.*

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
| DSIVI-grid-10K 4096/2048 (**2.43**) | DSIVI-grid-10K 1024/4096 (2.44) | DSIVI-grid-10K 256/8192 (2.45) |

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
6. **The completed rev2 batch grid changed the 10K/20K story on BNNs**: tuned 10K runs now beat the older Boston 20K result, materially improve yacht/concrete, and nearly match the older best protein run; only protein still clearly prefers the earlier off-20K setup overall.
7. **KSIVI annealing**: improves banana (KL 0.074→0.061), breaks student_uc (diverges), marginally helps multimodal. Best config is target-dependent.
8. **RSIVI/AISIVI** unreliable due to RealNVP instability on several targets.
9. **DSIVI annealing pattern for new BNN after the completed grid**: annealing on now wins clearly on boston, yacht, power, and concrete once batches are tuned; protein still prefers the older off-20K run overall, and winered still belongs to UIVI/SIVI.
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
| DSIVI-grid (4096/2048) | on | 10K | **0.627** | **1.025** | n/a | 0.078s |
| DSIVI | off | 10K | 1.975 | 2.342 | — | 0.060s |
| DSIVI | off | 20K | 1.807 | 2.480 | — | 0.055s |
| KSIVI | off | 10K | ★ div | — | 8572 | 0.009s |
| KSIVI | off | 20K | ★ div | — | 116K | 0.009s |
| KSIVI | off | 30K | ★ div | — | -18M | 0.009s |

*Best NLL/RMSE: DSIVI-grid (4096/2048) at RMSE 0.627 and NLL 1.025. The completed rev2 grid cleanly improves on both the older default annealed run and the one-off uniform rerun.*
*Yacht remains strongly anneal-friendly, but the finished grid shows the winning pair is `4096/2048`, with `256/4096` as the cheap close second rather than the earlier `256/1024` probe.*

---

## DSIVI Reduced Batch Grid (2026-03-29)

**Purpose**: finish the rev2 DSIVI BNN batch search with the reduced grid requested after the earlier partial sweep turned out to be misleading.

**Setup**: annealing on, `train.reverse.epochs=2`, `train.epochs=10000` for all 6 BNN targets, `bs ∈ {256, 1024, 4096}`, `rbs ∈ {1024, 2048, 4096, 8192}`.

**Execution ledger**: 72 total points = 44 newly run + 28 exact 10K reuses. Every reused point was an exact `10K` `rev2` match; no halfway `20K` truncation was needed.

### Sweep Summary

| Target | Best bs/rbs | RMSE↓ | NLL↓ | Avg ep | Fastest bs/rbs | Fastest avg ep |
|--------|-------------|-------|------|--------|----------------|----------------|
| Bnn_boston | 4096/2048 | 2.4448 | 2.4319 | 0.089s | 256/1024 | 0.032s |
| Bnn_yacht | 4096/2048 | 0.6271 | 1.0251 | 0.078s | 256/1024 | 0.027s |
| Bnn_power | 4096/8192 | 4.1458 | 2.8418 | 0.094s | 256/1024 | 0.025s |
| Bnn_concrete | 4096/1024 | 6.0833 | 3.2215 | 0.084s | 256/1024 | 0.029s |
| Bnn_protein | 1024/2048 | 4.6210 | 2.9521 | 0.049s | 256/1024 | 0.032s |
| Bnn_winered | 1024/8192 | 0.5932 | 0.8879 | 0.112s | 256/1024 | 0.031s |

### Bnn_boston (z_dim=751)

| bs | rbs | RMSE↓ | NLL↓ | Avg ep | Source | Run dir |
|----|-----|-------|------|--------|--------|---------|
| 256 | 1024 | 3.2601 | 2.8024 | 0.032s | reused | 20260329_121959 |
| 256 | 2048 | 2.9127 | 2.5275 | 0.047s | reused | 20260329_122703 |
| 256 | 4096 | 3.3187 | 3.1973 | 0.067s | reused | 20260329_123645 |
| 256 | 8192 | 2.5463 | 2.4511 | 0.112s | reused | 20260329_124943 |
| 1024 | 1024 | 2.8303 | 2.5082 | 0.045s | reused | 20260329_131009 |
| 1024 | 2048 | 2.7435 | 2.4924 | 0.056s | reused | 20260329_131926 |
| 1024 | 4096 | 2.5009 | 2.4428 | 0.077s | reused | 20260329_133032 |
| 1024 | 8192 | 2.6290 | 2.4697 | 0.125s | reused | 20260329_134508 |
| 4096 | 1024 | 2.8385 | 2.5116 | 0.078s | new | see runs.tsv |
| 4096 | 2048 | **2.4448** | **2.4319** | 0.089s | new | see runs.tsv |
| 4096 | 4096 | 2.6797 | 2.4780 | 0.114s | new | see runs.tsv |
| 4096 | 8192 | 2.6993 | 2.4812 | 0.167s | new | see runs.tsv |

*Boston was the big correction: the completed rev2 grid beats the older `20K` run by a wide margin, and several 10K points cluster tightly in the `2.43-2.53` NLL band.*

### Bnn_yacht (z_dim=401)

| bs | rbs | RMSE↓ | NLL↓ | Avg ep | Source | Run dir |
|----|-----|-------|------|--------|--------|---------|
| 256 | 1024 | 1.4779 | 1.7807 | 0.027s | new | see runs.tsv |
| 256 | 2048 | 0.9877 | 1.1500 | 0.033s | new | see runs.tsv |
| 256 | 4096 | 0.6844 | 1.0315 | 0.045s | new | see runs.tsv |
| 256 | 8192 | 0.9679 | 1.1473 | 0.108s | reused | 20260326_215651 |
| 1024 | 1024 | 0.9422 | 1.3808 | 0.032s | new | see runs.tsv |
| 1024 | 2048 | 0.7819 | 1.5059 | 0.038s | new | see runs.tsv |
| 1024 | 4096 | 1.0275 | 1.5681 | 0.053s | new | see runs.tsv |
| 1024 | 8192 | 1.0285 | 2.0084 | 0.109s | reused | 20260326_212008 |
| 4096 | 1024 | 1.0518 | 1.3961 | 0.073s | reused | 20260327_025407 |
| 4096 | 2048 | **0.6271** | **1.0251** | 0.078s | reused | 20260327_024000 |
| 4096 | 4096 | 1.1264 | 1.8546 | 0.091s | reused | 20260327_022347 |
| 4096 | 8192 | 0.7939 | 1.0729 | 0.119s | reused | 20260326_022947 |

*Yacht stayed highly non-monotone: `4096/2048` wins, `256/4096` is a very close second, and the `1024/*` band is uniformly worse on NLL.*

### Bnn_power (z_dim=301)

| bs | rbs | RMSE↓ | NLL↓ | Avg ep | Source | Run dir |
|----|-----|-------|------|--------|--------|---------|
| 256 | 1024 | 4.1758 | 2.8523 | 0.025s | new | see runs.tsv |
| 256 | 2048 | 4.2843 | 2.9036 | 0.030s | new | see runs.tsv |
| 256 | 4096 | 4.2212 | 2.8592 | 0.040s | new | see runs.tsv |
| 256 | 8192 | 4.1998 | 2.8546 | 0.084s | reused | 20260326_231934 |
| 1024 | 1024 | 4.1806 | 2.8579 | 0.029s | new | see runs.tsv |
| 1024 | 2048 | 4.1474 | 2.8421 | 0.034s | new | see runs.tsv |
| 1024 | 4096 | 4.2948 | 2.8951 | 0.041s | new | see runs.tsv |
| 1024 | 8192 | 4.2091 | 2.8564 | 0.086s | reused | 20260326_224912 |
| 4096 | 1024 | 4.2864 | 2.8906 | 0.059s | reused | 20260327_034438 |
| 4096 | 2048 | 4.3943 | 2.9128 | 0.063s | reused | 20260327_033309 |
| 4096 | 4096 | 4.1702 | 2.8474 | 0.073s | reused | 20260327_032009 |
| 4096 | 8192 | **4.1458** | **2.8418** | 0.094s | reused | 20260326_041301 |

*Power is the flattest target in the suite: the default `4096/8192` point still wins by a hair, but `1024/2048` is effectively tied while `256/1024` gets within `0.0105` NLL at about one quarter of the epoch cost.*

### Bnn_concrete (z_dim=501)

| bs | rbs | RMSE↓ | NLL↓ | Avg ep | Source | Run dir |
|----|-----|-------|------|--------|--------|---------|
| 256 | 1024 | 9.5832 | 3.8283 | 0.029s | new | see runs.tsv |
| 256 | 2048 | 9.4386 | 3.9324 | 0.034s | new | see runs.tsv |
| 256 | 4096 | 6.4749 | 3.3036 | 0.053s | new | see runs.tsv |
| 256 | 8192 | 6.2919 | 3.2675 | 0.113s | reused | 20260326_181102 |
| 1024 | 1024 | 7.5781 | 3.5575 | 0.036s | new | see runs.tsv |
| 1024 | 2048 | 9.3107 | 3.9315 | 0.044s | new | see runs.tsv |
| 1024 | 4096 | 9.9017 | 3.9396 | 0.061s | new | see runs.tsv |
| 1024 | 8192 | 9.9835 | 3.9547 | 0.127s | reused | 20260326_172710 |
| 4096 | 1024 | **6.0833** | **3.2215** | 0.084s | reused | 20260327_034851 |
| 4096 | 2048 | 9.0770 | 3.6689 | 0.092s | reused | 20260327_033214 |
| 4096 | 4096 | 6.4670 | 3.3017 | 0.106s | reused | 20260327_031315 |
| 4096 | 8192 | 6.1318 | 3.2373 | 0.140s | reused | 20260326_021727 |

*Concrete is the sharpest counterexample to any blanket low-rbs rule: `4096/1024` is best, `4096/8192` is nearly as good, but several neighboring points collapse back into the `3.67-3.95` NLL range.*

### Bnn_protein (z_dim=551)

| bs | rbs | RMSE↓ | NLL↓ | Avg ep | Source | Run dir |
|----|-----|-------|------|--------|--------|---------|
| 256 | 1024 | 4.9670 | 3.0206 | 0.032s | new | see runs.tsv |
| 256 | 2048 | 4.7812 | 2.9835 | 0.041s | new | see runs.tsv |
| 256 | 4096 | 4.8416 | 2.9959 | 0.059s | new | see runs.tsv |
| 256 | 8192 | 4.6580 | 2.9594 | 0.094s | new | see runs.tsv |
| 1024 | 1024 | 4.7421 | 2.9761 | 0.041s | new | see runs.tsv |
| 1024 | 2048 | **4.6210** | **2.9521** | 0.049s | new | see runs.tsv |
| 1024 | 4096 | 4.6493 | 2.9575 | 0.067s | new | see runs.tsv |
| 1024 | 8192 | 4.8151 | 2.9899 | 0.103s | new | see runs.tsv |
| 4096 | 1024 | 4.7160 | 2.9712 | 0.085s | new | see runs.tsv |
| 4096 | 2048 | 4.8174 | 2.9906 | 0.093s | new | see runs.tsv |
| 4096 | 4096 | 4.6894 | 2.9655 | 0.110s | new | see runs.tsv |
| 4096 | 8192 | 4.6755 | 2.9626 | 0.163s | reused | 20260326_041841 |

*Protein prefers the middle of the grid: `1024/2048` is the clear 10K winner, while the older `20K` off-anneal run still holds the overall protein crown by a small margin.*

### Bnn_winered (z_dim=651)

| bs | rbs | RMSE↓ | NLL↓ | Avg ep | Source | Run dir |
|----|-----|-------|------|--------|--------|---------|
| 256 | 1024 | 0.6026 | 0.9064 | 0.031s | new | see runs.tsv |
| 256 | 2048 | 0.6581 | 1.0056 | 0.041s | new | see runs.tsv |
| 256 | 4096 | 0.6137 | 0.9304 | 0.061s | new | see runs.tsv |
| 256 | 8192 | 0.6535 | 1.0015 | 0.107s | new | see runs.tsv |
| 1024 | 1024 | 0.6024 | 0.9063 | 0.042s | new | see runs.tsv |
| 1024 | 2048 | 0.6349 | 0.9662 | 0.048s | new | see runs.tsv |
| 1024 | 4096 | 0.6484 | 0.9868 | 0.070s | new | see runs.tsv |
| 1024 | 8192 | **0.5932** | **0.8879** | 0.112s | new | see runs.tsv |
| 4096 | 1024 | 0.7070 | 1.1030 | 0.071s | new | see runs.tsv |
| 4096 | 2048 | 0.6045 | 0.9145 | 0.080s | new | see runs.tsv |
| 4096 | 4096 | **0.5902** | 0.8907 | 0.101s | new | see runs.tsv |
| 4096 | 8192 | 0.5948 | 0.8946 | 0.175s | reused | 20260326_063600 |

*Winered improves modestly with tuning, but the broader story does not change: the grid narrows the gap to the best DSIVI points, yet UIVI and SIVI still win overall on this target.*

### Updated Takeaways

- **Boston was the main surprise**: the completed `rev2` grid overturns the earlier partial picture and makes `10K` clearly sufficient once batch sizes are tuned.
- **Yacht and concrete are highly non-monotone** in `rbs`: neighboring points can swing from excellent to clearly bad, so one-factor sweeps were not enough.
- **Power is almost flat**: if runtime matters, `256/1024` is a very cheap point that gives up almost nothing.
- **Protein prefers the middle of the grid**: `1024/2048` is the best `10K` annealed point, but the older `off-20K` run is still slightly better overall.
- **Winered only moves a little**: tuning helps DSIVI, but it still does not dislodge UIVI or SIVI on this dataset.

---

## Bnn_power (z_dim=301)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 4.334 | 3.069 | — | 0.016s |
| UIVI | on | 10K | 4.366 | 3.061 | — | 0.095s |
| DSIVI | on | 10K | **4.146** | **2.842** | — | 0.034s |
| DSIVI-grid (1024/2048) | on | 10K | 4.147 | 2.842 | n/a | 0.034s |
| DSIVI | off | 10K | 4.310 | 2.915 | — | 0.033s |
| DSIVI | off | 20K | 4.308 | 2.948 | — | 0.033s |
| KSIVI | off | 10K | ★ div | — | 1075 | 0.009s |
| KSIVI | off | 20K | ★ div | — | -612 | 0.009s |
| KSIVI | off | 30K | ★ div | — | -1573 | 0.009s |

*Best: the annealed 10K DSIVI family still wins power. The completed grid shows `4096/8192` is the absolute best point by a hair, while `1024/2048` is effectively tied and much cheaper.*
*Power is the least sensitive target in the sweep: even the fastest `256/1024` point only gives up about `0.01` NLL relative to the best result.*

---

## Bnn_concrete (z_dim=501)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 12.305 | 4.140 | — | 0.023s |
| UIVI | on | 10K | 10.210 | 4.088 | — | 0.094s |
| DSIVI | on | 10K | **6.132** | **3.237** | — | 0.080s |
| DSIVI-grid (4096/1024) | on | 10K | **6.083** | **3.222** | n/a | 0.084s |
| DSIVI | off | 10K | 10.344 | 4.008 | — | 0.077s |
| DSIVI | off | 20K | 10.505 | 3.998 | — | 0.080s |
| KSIVI | off | 10K | ★ div | — | 32149 | 0.009s |
| KSIVI | off | 20K | ★ div | — | 16131 | 0.009s |
| KSIVI | off | 30K | ★ div | — | 331 | 0.009s |

*Best: DSIVI-grid (4096/1024) improves the already-strong annealed baseline to RMSE 6.08 and NLL 3.22, far ahead of UIVI and all off-anneal runs.*
*Concrete is the clearest warning against blanket rules: both `4096/1024` and `4096/8192` are excellent, but neighboring points in the same grid collapse back toward the `3.7-4.0` NLL band.*

---

## Bnn_protein (z_dim=551)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 5.097 | 3.047 | — | 0.025s |
| UIVI | on | 10K | 5.111 | 3.050 | — | 0.093s |
| DSIVI | on | 10K | 4.676 | 2.963 | — | 0.103s |
| DSIVI-grid (1024/2048) | on | 10K | 4.621 | 2.952 | n/a | 0.049s |
| DSIVI | off | 10K | 4.707 | 2.968 | — | 0.105s |
| DSIVI | off | 20K | **4.566** | **2.941** | — | 0.099s |
| KSIVI | off | 10K | ★ div | — | 104K | 0.009s |
| KSIVI | off | 20K | ★ div | — | -143M | 0.010s |
| KSIVI | off | 30K | ★ div | — | -116G | 0.009s |

*Best overall: DSIVI-off-20K (RMSE 4.57, NLL 2.94). Within the annealed 10K grid, `1024/2048` is the clear winner and now nearly matches the older off-20K result.*
*Protein prefers middle-sized batches: the grid substantially improves the earlier annealed 10K point without changing the overall conclusion that protein is one of the few targets still worth pushing past 10K when annealing is off.*

---

## Bnn_winered (z_dim=651)

| Method | Anneal | Epochs | RMSE↓ | NLL↓ | KSD | Ep time |
|--------|--------|--------|-------|------|-----|---------|
| SIVI | on | 20K | 0.577 | 0.868 | — | 0.029s |
| UIVI | on | 10K | **0.568** | **0.853** | — | 0.091s |
| DSIVI | on | 10K | 0.595 | 0.895 | — | 0.115s |
| DSIVI-grid (1024/8192) | on | 10K | 0.593 | 0.888 | n/a | 0.112s |
| DSIVI | off | 10K | 0.587 | 0.876 | — | 0.119s |
| DSIVI | off | 20K | 0.582 | 0.875 | — | 0.111s |
| KSIVI | off | 10K | ★ div | — | -246K | 0.009s |
| KSIVI | off | 20K | ★ div | — | -2.1G | 0.010s |
| KSIVI | off | 30K | ★ div | — | 597K | 0.009s |

*Best overall: UIVI (RMSE 0.568, NLL 0.853). The completed rev2 grid nudges annealed DSIVI down to NLL 0.888 at `1024/8192`, but that still leaves UIVI and SIVI ahead on winered.*
*Winered moved only modestly under the grid search, so the broad recommendation does not change: DSIVI is competitive, but not the winner here.*

---

## New BNN Targets: Rankings by NLL↓

| Target | 1st | 2nd | 3rd |
|--------|-----|-----|-----|
| Bnn_yacht | DSIVI-grid 4096/2048 (**1.025**) | DSIVI-grid 256/4096 (1.032) | DSIVI-on-10K (1.073) |
| Bnn_power | DSIVI-on-10K (**2.842**) | DSIVI-grid 1024/2048 (2.842) | DSIVI-grid 256/1024 (2.852) |
| Bnn_concrete | DSIVI-grid 4096/1024 (**3.222**) | DSIVI-on-10K (3.237) | DSIVI-grid 256/8192 (3.268) |
| Bnn_protein | DSIVI-off-20K (**2.941**) | DSIVI-grid 1024/2048 (2.952) | DSIVI-grid 1024/4096 (2.958) |
| Bnn_winered | UIVI (**0.853**) | SIVI (0.868) | DSIVI-off-20K (0.875) |

*After the completed rev2 grid, tuned DSIVI is now clearly top on yacht/power/concrete and dramatically better on Boston, while protein still prefers the older off-20K run and winered still belongs to UIVI/SIVI. The main lesson is that the final answer came from the joint grid, not from the earlier one-factor sweep or the one-off uniform rerun.*
