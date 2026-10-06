# ConditionalGaussianGlobal seed-44 evolution

Repeat the seed-43 global Gaussian comparison with seed 44. Both runs launch
in parallel and include initial samples, a model checkpoint, and a plot at step
zero:

1. 5,000 updates, 10,000 samples and a contour plot every 500 updates.
2. 50,000 updates, 10,000 samples and a contour plot every 5,000 updates.

All other settings match the seed-43 runs: ConditionalGaussianGlobal, two
width-128 SiLU hidden layers, effective input noise dimension 2, batch size
128, mean/variance learning rates 0.001, Adam betas (0.9, 0.999), and StepLR
gamma 0.9 every 1,000 steps. Annealing remains linear over 25,000 steps, so
the short run ends at factor 0.28 and the long run at factor 1.0. The shared
variance starts at softplus(1), approximately 1.3133 per coordinate, and is
trainable. Metrics are temporarily disabled. Checkpoints remain every 5,000
updates, plus the initial checkpoint.

Reporting preserves training RNG. The two runs must have identical initial
weights and samples, and identical weights at step 5,000. Reports verify all
22 sample/plot snapshots, saved configurations, optimizer groups, scheduler
progress, and checkpoints. Individual 1x11 grids and a combined 2x11 PNG/PDF
grid are generated. Previous seed-43 reports are retained in their own campaign.

The shared experiment script accepts the seed as an argument. Run from the
repository root on the GPU host inside tmux:

```bash
/root/miniconda3/envs/ruivi/bin/python -u campaigns/ksivi_x_shaped_global_seed43_evolution_20261006/run_global_campaign.py \
  --seed 44 \
  --results-root /root/ruivi/results/ksivi_x_shaped_global_seed44_evolution_20261006 \
  --tb-root /root/ruivi/tb_logs/ksivi_x_shaped_global_seed44_evolution_20261006 \
  --report-root campaigns/ksivi_x_shaped_global_seed44_evolution_20261006/generated_reports
```

Status is under the results root in `runtime/state.json`. Generated figures
are committed and pushed on the server, then pulled locally through Git.
