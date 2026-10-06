# ConditionalGaussianGlobal seed-43 evolution

Two runs launch in parallel using the canonical KSIVI x-shaped settings with
only the family changed to ConditionalGaussianGlobal:

1. 5,000 steps, native contour plots and saved samples every 500 steps.
2. 50,000 steps, native contour plots and saved samples every 5,000 steps.

Both include the actual initial model, samples, and plot at step zero. Each
snapshot contains 10,000 samples. Reporting preserves training RNG, so the
initial weights/samples and the two models' 5,000-step weights must match.

Seed is 43. The mean network has two width-128 SiLU hidden layers and input
noise dimension 2. Batch size is 128. Mean and global-variance learning rates
are both 0.001; Adam betas are (0.9, 0.999). StepLR decays by 0.9 every 1,000
steps. Annealing remains linear over 25,000 steps, ending at factor 0.28 in
the short run and 1.0 in the long run. Metrics are temporarily disabled.
The global variance starts at softplus(1), approximately 1.3133 per coordinate,
and remains trainable. Scheduled checkpoints are every 5,000 steps.

Reports include individual 1x11 PNG/PDF grids and a combined 2x11 grid. The
combined rows use different time intervals, explicitly labeled in each panel.
Validation checks configurations, finite samples and checkpoints, initial
optimizer state, optimizer groups, scheduler progress, and matching initial
and shared-step weights. The canonical config itself remains ConditionalGaussian.

Run from the repository root on the GPU host inside tmux:

```bash
/root/miniconda3/envs/ruivi/bin/python -u campaigns/ksivi_x_shaped_global_seed43_evolution_20261006/run_global_campaign.py \
  --results-root /root/ruivi/results/ksivi_x_shaped_global_seed43_evolution_20261006 \
  --tb-root /root/ruivi/tb_logs/ksivi_x_shaped_global_seed43_evolution_20261006
```

Runtime status is under the results root in `runtime/state.json`. Figures are
generated on the server, committed and pushed there, then pulled locally.
