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
