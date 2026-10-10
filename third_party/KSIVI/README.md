# Kernel Semi-Implicit Variational Inference

This is the code repository for our paper [Kernel Semi-Implicit Variational Inference](https://arxiv.org/abs/2405.18997) (ICML 2024). 

For example, you can run the training scripts for different experiments in file `./run.sh`.

## Local X-shaped reproduction

This directory was imported from upstream commit
`4d995745265b06f37143f156760fa5ebe7ecbbec`. `UPSTREAM.json` records the
original file hashes; parent-workspace commit `896ca1b` contains the untouched
snapshot. The only omitted upstream file is macOS metadata `datasets/.DS_Store`.

The local reproduction uses the authors' `SIMINet`, X-shaped target, Gaussian
kernel/median rule, Adam betas `(0.9, 0.99)`, and StepLR. It preserves
`configs/x_shaped.yml`: two width-50 ReLU hidden layers, latent dimension 3,
batch size 500 in each of two independent batches, 50,000 updates, initial
variance 1, exponential variance parameters, learning rate 0.001, decay 0.9
every 1,000 updates, and **no annealing**. These defaults differ from the
earlier StatComp investigation; this is an independent reproduction under the
authors' defaults, not an exact rerun of that campaign.

The four treatments, each run with seeds 42, 43, and 44, are:

1. Conditional variance and the original conditional-score loss.
2. The canonical shared global variance and original loss.
3. Conditional variance and the conventional Gaussian Stein-kernel loss.
4. Conditional variance and original loss with minimum variance 0.2.

Conditional variance shares the original mean network's hidden layers and adds
a linear log-variance head. Its output weights start at zero and biases at the
canonical log variance, giving identical initial means and variances across
methods. Creating the extra head preserves the training RNG state. The first
three treatments retain the upstream variance floor `exp(-20)`; the fourth
uses `max(exp(-20), 0.2)`. The Stein treatment holds each fitted median width
fixed during backpropagation, as in the earlier investigation; the other three
retain the upstream derivative through the fitted width. Their finite-batch
updates should therefore not be interpreted as two unbiased estimators of one
fixed-bandwidth objective.

The Gaussian kernel code is unchanged from upstream. A matrix-product distance
optimization was screened, then discarded: float32 rounding can change the
selected median pair and its gradient even when loss values agree closely.
The interrupted pilot is excluded from the reported results. The `check`
command compares values and full parameter gradients against the pristine
upstream expression. Diagnostics run outside the training RNG stream and are
saved every 5,000 updates, avoiding
the upstream per-epoch plotting and snapshot overhead. Checkpoints include
Adam, scheduler, and RNG state for reproducible resumption.

Run from the parent StatComp worktree using its existing virtual environment:

```powershell
.\.venv\Scripts\python.exe third_party/KSIVI/reproduce_x_shaped.py check
.\.venv\Scripts\python.exe third_party/KSIVI/reproduce_x_shaped.py benchmark
.\.venv\Scripts\python.exe third_party/KSIVI/reproduce_x_shaped.py run --jobs 6 --threads 1
```

Raw checkpoints, progress, training trajectories, and final samples stay under
`results/ksivi_upstream_local_20261010/`. Its `report/` subdirectory contains the
3-by-4 comparison, 12 separate panel images, per-run metrics, and a manifest
with complete configurations, source hashes, software versions, and results.
Sliced-Wasserstein evaluation uses 10,000 samples and 128 directions. Black
curves show the exact target density; every panel uses the same viewing window
and count color scale. Samples outside the plot remain included in metrics.

Regenerate the report without retraining:

```powershell
.\.venv\Scripts\python.exe third_party/KSIVI/reproduce_x_shaped.py report
```

Audit the final checkpoints, regenerate their samples exactly, recompute every
saved metric, and compare parameter gradients with independent loss expressions:

```powershell
.\.venv\Scripts\python.exe third_party/KSIVI/verify_reproduction.py
```

## Completed local results

All 12 runs completed 50,000 updates locally on the Intel Core Ultra 9 185H
CPU. The final sliced-Wasserstein distances are below; smaller is better.
The last column reports the mean and sample standard deviation over three seeds.

| Treatment | Seed 42 | Seed 43 | Seed 44 | Mean (SD) |
| --- | ---: | ---: | ---: | ---: |
| Conditional, original loss | 0.4804 | 0.0791 | 0.5466 | 0.3687 (0.2530) |
| Global, original loss | 0.0397 | 0.0403 | 0.0355 | 0.0385 (0.0026) |
| Conditional, Stein loss | 0.0378 | 0.0448 | 0.0772 | 0.0532 (0.0210) |
| Conditional, minimum variance 0.2 | 0.0381 | 0.0439 | 0.0461 | 0.0427 (0.0041) |

The conditional-versus-global gap reproduces under the authors' defaults.
The largest conditional errors occur in seeds 42 and 44, where the fitted
distributions put excess mass on one side of the target. Both interventions
improve every paired conditional run in this experiment. Independent samples
from the exact target score 0.0303--0.0381 against the common reference sample,
which indicates the scale of finite-sample evaluation noise. These are three
paired seeds, not a universal ranking of estimators. The canonical defaults
differ jointly from the earlier investigation, so this reproduction does not
isolate which changed setting affects the severity of the gap. The Stein
treatment also includes the bandwidth-detachment policy described above.

For a diagonal Gaussian component, its conditional score has coordinate second
moment `E[score_j^2] = E[1/v_j]`. A minimum variance of 0.2 bounds this moment by
5 per coordinate. This controls one source of score noise, rather than bounding
every parameter-gradient variance. The Stein loss removes the explicit
component-score terms: seed 42 fits well despite mean inverse component
variance about 20 million. Small component variance therefore does not itself
imply a poor marginal distribution. These observations are consistent with the
earlier finite-batch noise explanation; the images alone do not prove causality
or a universal variance advantage for the Stein gradient.

The final audit passed for all 12 runs: saved samples regenerate exactly from
the checkpoints, every metric recomputes, all nine original-loss gradients
match the upstream expression exactly, and the three Stein gradients agree
with an independent broadcast expression to relative L2 error below `8.4e-7`.
The initial Stein check also verifies the loss and its gradient through
automatic spatial differentiation of the kernel. The canonical config,
target implementation, and Gaussian kernel retain their original byte hashes.

The committed outputs include 12 individual panels, the combined
`results/ksivi_upstream_local_20261010/report/final_samples.png`, `metrics.csv`,
`manifest.json`, `verification.json`, hardware details, and each run's final
samples, checkpoint, specification, and training trajectory. Interrupted pilot
runs and interim preview images remain excluded from the committed results.
