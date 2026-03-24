# Experiment Results — DSIVI on BNN (Bnn_boston)

Structured results table. Only includes runs from the current experiment round (2026-03-24+).

## UIVI Baselines

| Run | Annealing | Epochs | Best ELBO | KSD | Fisher | BNN RMSE | BNN NLL |
|-----|-----------|--------|-----------|-----|--------|----------|---------|
| UIVI-A | linear/5000 | 10K | **-909.81** @ 10K | 3.53 | 4.1M | 4.86 | 3.42 |
| UIVI-B | linear/1000 | 5K | -1072.66 @ 5K | -0.50 | 1.2M | 4.85 | 3.39 |

## DSIVI Experiments — Completed

| Run | Anneal | Rev Ep | LR Step | Epochs | Best ELBO | Epoch | KSD | Fisher | BNN RMSE | BNN NLL | Stable? |
|-----|--------|--------|---------|--------|-----------|-------|-----|--------|----------|---------|---------|
| baseline | 1000 | 10 | 2000 | 10K | -1383 | 90 | 102 | 6.5M | **3.33** | **2.62** | ❌ nan@590 |
| anneal500 | 500 | 10 | 2000 | 10K | -1095 | 9840 | 1.4 | 2.8M | 4.46 | 3.10 | ✅ |
| anneal5000 | 5000 | 10 | 2000 | 10K | -1367 | 760 | 0.2 | 485K | 4.26 | 2.96 | ❌ nan@800 |
| rev2 | 500 | 2 | 2000 | 10K | -1308 | 120 | 119 | 7.3M | **3.25** | 2.66 | ❌ nan@600 |
| anneal500-20K | 500 | 10 | 2000 | 20K | -999 | 14950 | 0.2 | 5.4M | 4.46 | 3.10 | ✅ |
| **slowlr-20K** | **500** | **10** | **4000** | **20K** | **-940** | **19440** | 0.18 | 2.8M | 4.97 | 3.28 | ✅ |

## DSIVI Experiments — Running

| Run | Anneal | Rev Ep | LR Step | Epochs | Best ELBO | Epoch | Status |
|-----|--------|--------|---------|--------|-----------|-------|--------|
| slowlr-30K | 500 | 10 | 5000 | 30K | -1059 | 11520 | GPU 0, ~12K |
| veryslow-30K | 500 | 10 | 6000 | 30K | — | — | GPU 1, just launched |

## Key Findings

1. **DSIVI reaches -940 ELBO** (slowlr-20K) — within 30 units of UIVI-A (-910). This demonstrates DSIVI is competitive with UIVI on BNN.
2. **Three critical hyperparameters identified**:
   - **Annealing = 500 steps** (shorter is better for DSIVI stability)
   - **Reverse inner epochs = 10** (needed for ELBO stability)
   - **VI LR schedule: StepLR(step=4000+, gamma=0.7)** (slower decay crucial for long runs)
3. **ELBO trajectory scales with epochs**: -1095 (10K) → -999 (20K standard) → -940 (20K slow LR)
4. **LR decay is the bottleneck**: Standard StepLR(2000, 0.7) peaks at epoch ~15K. Slower decay (step=4000) allows continued improvement to epoch 19K+.
5. **Still improving**: With step=5000 and 30K epochs, expect ELBO below -940.
