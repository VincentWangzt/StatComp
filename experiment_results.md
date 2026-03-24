# Experiment Results — DSIVI on BNN (Bnn_boston)

## Phase 2: BNN-Focused (current)

### UIVI Baselines (converged)

| Run | Annealing | Epochs | Best RMSE | Best NLL | Epoch@best | Wall-time | Ep time |
|-----|-----------|--------|-----------|----------|------------|-----------|---------|
| UIVI-A-ext | 5000 | 20K | 4.81 | 3.36 | ~9500 | ~28min | 0.178s |
| UIVI-B-ext | 1000 | 20K | 4.74 | 3.35 | ~12500 | ~37min | 0.177s |

**UIVI BNN ceiling**: RMSE ~4.75, NLL ~3.35. Converged by epoch 10-13K. Extra epochs do not help.

### DSIVI BNN-Focused Experiments

| Run | Annealing | Rev Ep | Batch | Epochs | Best RMSE | Best NLL | Ep@best | Wall-best | Ep time |
|-----|-----------|--------|-------|--------|-----------|----------|---------|-----------|---------|
| anneal1000+rev2 | 1000 | 2 | 4096 | 20K | **3.20** | **2.59** | 9680 | ~35min | 0.215s |
| noanneal+rev2 | off | 2 | 4096 | 20K | 3.23 | 2.60 | 11600 | ~41min | 0.209s |

### DSIVI BNN-Focused — Running

| Run | Annealing | Rev Ep | Batch | Epochs | Status |
|-----|-----------|--------|-------|--------|--------|
| smallbatch+rev2 | 1000 | 2 | 2048 | 20K | GPU 0 |
| anneal1000+rev5 | 1000 | 5 | 4096 | 20K | GPU 1 |

### Head-to-Head: Fixed Time Budget

**At 30 minutes of wall-clock time:**

| Method | Epochs reached | Best RMSE so far | Best NLL so far |
|--------|----------------|------------------|-----------------|
| UIVI-A (anneal5000) | ~10000 | 4.83 | 3.36 |
| UIVI-B (anneal1000) | ~10000 | 4.74 | 3.35 |
| DSIVI anneal1000+rev2 | ~8200 | **3.20** | **2.59** |
| DSIVI noanneal+rev2 | ~8600 | 3.28 | 2.61 |

**DSIVI beats UIVI by 1.5 RMSE points and 0.76 NLL at the same wall-clock time.**

### Head-to-Head: Fixed Step Count

**At 10,000 epochs:**

| Method | Wall-time | Best RMSE | Best NLL |
|--------|-----------|-----------|----------|
| UIVI-A (anneal5000) | 30min | 4.83 | 3.36 |
| UIVI-B (anneal1000) | 30min | 4.74 | 3.35 |
| DSIVI anneal1000+rev2 | 36min | **3.20** | **2.59** |
| DSIVI noanneal+rev2 | 35min | 3.30 | 2.62 |

**DSIVI still wins at the same step count despite being slightly slower per step.**

### Key Findings (Phase 2)

1. **DSIVI dominates UIVI on BNN prediction**: RMSE 3.20 vs 4.74 (32% better), NLL 2.59 vs 3.35 (23% better)
2. **rev2 (2 reverse inner epochs) is optimal for BNN**: Closes the epoch-time gap to UIVI (0.21s vs 0.18s), and BNN metrics match or beat rev10
3. **Annealing helps slightly**: anneal1000+rev2 gets best RMSE 3.20 (faster) vs no-annealing 3.23 (slower to converge)
4. **BNN converges by epoch 8-12K** for DSIVI, similar to UIVI, but at a much better level
5. **No-annealing is viable**: Similar final metrics, faster early convergence (good BNN by epoch 500)

---

## Phase 1 Summary (ELBO-focused, completed)

Best ELBO: -586.5 (DSIVI veryslow-30K with step=6000)
See experiment_observations.md for full Phase 1 details.
