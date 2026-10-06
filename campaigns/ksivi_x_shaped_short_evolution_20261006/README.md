# KSIVI early evolution: nine repeated experiments

Repeat the previous nine experiments for 5,000 updates, with snapshots at
0, 500, 1,000, ..., 5,000. The three rounds each launch three runs in parallel:

1. Canonical ConditionalGaussian, seeds 42/43/44, learning rate 0.001.
2. Canonical ConditionalGaussian, seed 43, learning rates 5e-4/2e-3/2e-4.
3. ConditionalGaussian with constant initial variance 1.31326162815094,
   seeds 42/43/44, learning rate 0.001. Variance remains trainable.

The other settings match the previous runs: two width-128 SiLU hidden layers,
effective input noise dimension 2, batch size 128, Adam betas (0.9, 0.999),
StepLR every 1,000 updates with gamma 0.9, and linear annealing over 25,000
steps. Because the annealing schedule is unchanged, its factor at step 5,000
is 0.28. Metrics remain temporarily disabled. Each snapshot contains 10,000
samples. Scheduled checkpoints remain every 5,000 steps.

Opt-in `train.plot.initial=true` saves the actual pre-update samples, model
checkpoint, and contour plot. Opt-in `train.sample.preserve_rng=true` isolates
saved samples and plots from training's random-number stream. This prevents
the new initial snapshot and more frequent plots from changing training draws.
Each native plot uses the same samples as its corresponding sample file.

Three 3x11 PNG/PDF grids are generated from the native plots, including the
initial distribution in the first column. Reports verify sample finiteness,
configurations, final checkpoints, optimizer settings, and starting weights.
All learning-rate runs must have identical initial weights and samples.
Constant-variance runs must retain the corresponding seed's hidden layers and
mean head. Optional comparisons record differences against the previous runs'
5,000-step checkpoints.

Run from the repository root on the GPU host inside tmux:

```bash
/root/miniconda3/envs/ruivi/bin/python -u campaigns/ksivi_x_shaped_short_evolution_20261006/run_short_campaign.py \
  --results-root /root/ruivi/results/ksivi_x_shaped_short_evolution_20261006 \
  --tb-root /root/ruivi/tb_logs/ksivi_x_shaped_short_evolution_20261006 \
  --previous-results-root /root/ruivi/results
```

Campaign status is saved under the results root in `runtime/state.json`.
Report artifacts are produced on the server and returned through Git.

Check reporting isolation locally:

```powershell
.\.venv\Scripts\python.exe campaigns\ksivi_x_shaped_short_evolution_20261006\check_reporting.py
```
