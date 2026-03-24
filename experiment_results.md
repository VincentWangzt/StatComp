# Experiment Results — DSIVI on BNN (Bnn_boston)

## Phase 2: BNN-Focused Results

### UIVI Baselines (converged, 20K epochs)

| Run | Annealing | Ep time | Best RMSE | Best NLL | Ep@best NLL | Wall to best |
|-----|-----------|---------|-----------|----------|-------------|-------------|
| UIVI-A | 5000 | 0.178s | 4.81 | 3.36 | 9500 | 28min |
| UIVI-B | 1000 | 0.177s | 4.74 | 3.35 | 12500 | 37min |

**UIVI BNN ceiling**: RMSE ~4.75, NLL ~3.35. Converged by epoch 10-13K. No improvement beyond.

### DSIVI BNN Experiments (all 20K epochs)

| Run | Anneal | RevEp | Batch | Ep time | Best RMSE | Best NLL | Ep@best | Wall to best | Wall total |
|-----|--------|-------|-------|---------|-----------|----------|---------|-------------|-----------|
| anneal1000+rev2 | 1000 | 2 | 4096 | 0.215s | **3.20** | **2.59** | 9680 | 35min | 72min |
| noanneal+rev2 | off | 2 | 4096 | 0.209s | 3.23 | 2.60 | 11600 | 41min | 70min |
| smallbatch+rev2 | 1000 | 2 | 2048 | **0.146s** | 3.25 | 2.60 | 13520 | 33min | 50min |
| **noanneal+smallbatch** | **off** | **2** | **2048** | **0.148s** | **3.21** | **2.59** | 7370 | **18min** | **50min** |
| anneal1000+rev5 | 1000 | 5 | 4096 | 0.348s | 3.33 | 2.62 | 12450 | 72min | 117min |

### 🏆 Best Configurations

**Best BNN quality (NLL)**: DSIVI anneal1000+rev2 (batch=4096) — NLL **2.59**, RMSE **3.20**

**Fastest to good BNN**: DSIVI noanneal+smallbatch — NLL **2.59** in **18 min** (faster per epoch than UIVI!)

**Best efficiency**: DSIVI noanneal+smallbatch — 0.148s/ep (17% faster than UIVI), same BNN quality

### Head-to-Head Comparisons

**At 30 minutes wall-clock:**

| Method | Config | Epochs done | Best NLL | Best RMSE |
|--------|--------|-------------|----------|-----------|
| UIVI-A | anneal5000 | ~10000 | 3.36 | 4.83 |
| UIVI-B | anneal1000 | ~10000 | 3.35 | 4.74 |
| DSIVI | anneal1000, rev2, b4096 | ~8200 | **2.59** | **3.20** |
| DSIVI | noanneal, rev2, b2048 | ~12100 | **2.59** | **3.21** |

**At 10,000 steps:**

| Method | Config | Wall-time | Best NLL | Best RMSE |
|--------|--------|-----------|----------|-----------|
| UIVI-A | anneal5000 | 30min | 3.36 | 4.83 |
| UIVI-B | anneal1000 | 30min | 3.35 | 4.74 |
| DSIVI | anneal1000, rev2, b4096 | 36min | **2.59** | **3.20** |
| DSIVI | noanneal, rev2, b2048 | 25min | **2.59** | **3.21** |

**DSIVI wins on every axis**: better NLL (2.59 vs 3.35), better RMSE (3.20 vs 4.74), comparable or faster wall-clock time.

### Key Findings

1. **DSIVI dominates UIVI on BNN**: 23% better NLL, 33% better RMSE
2. **rev2 is the sweet spot**: 2 reverse epochs achieves best BNN, fastest per epoch
3. **Batch=2048 is faster than UIVI per epoch**: 0.146-0.148s vs 0.178s (18% faster!)
4. **No annealing is viable**: Fastest convergence to good BNN (18 min), same final quality
5. **Annealing helps slightly for best RMSE**: anneal1000 gets 3.20 vs 3.21 (marginal)
6. **rev5 is strictly worse**: Slower (0.348s), marginally better NLL (2.62 vs 2.60 for batch4096)

### Recommended DSIVI Config for BNN

```yaml
train:
  epochs: 15000  # BNN converges by 7-13K
  batch_size: 2048  # Faster per epoch than UIVI
  annealing:
    enabled: false  # or steps: 1000 (marginal difference)
  reverse:
    epochs: 2  # Minimal overhead, best BNN
    batch_size: 4096
```

---

## Phase 1 Summary (ELBO-focused, archived)

Best ELBO: -586.5 (DSIVI veryslow-30K with anneal=500, step=6000, rev10, 30K epochs).
ELBO-optimal config is very different from BNN-optimal config.
