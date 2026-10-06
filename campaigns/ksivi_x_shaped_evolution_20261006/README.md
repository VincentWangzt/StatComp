# Canonical ConditionalGaussian KSIVI evolution

Two sequential rounds, each running three jobs in parallel:

1. Seeds 42, 43, and 44 with the canonical learning rate of 0.001.
2. Initial learning rates 5e-4, 2e-3, and 2e-4, all with seed 43.

The canonical architecture, batch size, annealing, and StepLR decay remain in use.
Toy-target binding makes the effective input noise dimension 2. The VI model has
two width-128 SiLU hidden layers and noise-dependent softplus variance. Metrics
are disabled by run overrides, leaving canonical metric defaults unchanged.
`train.vi.var_lr` is inactive for this model, which has no global `var_raw` parameter.

Each run saves 10,000 samples and a native contour plot every 5,000 updates.
The reports use the exact native plots in a 3x10 grid, with columns 5,000 through
50,000 and rows in the order above. Both PNG and PDF grids are generated.
Every panel's sample tensors and the final model/optimizer checkpoint are checked.

Run on the GPU host inside tmux:

```bash
/root/miniconda3/envs/ruivi/bin/python -u campaigns/ksivi_x_shaped_evolution_20261006/run_campaign.py \
  --results-root /root/ruivi/results/ksivi_x_shaped_evolution_20261006 \
  --tb-root /root/ruivi/tb_logs/ksivi_x_shaped_evolution_20261006
```

Campaign status is saved under the results root in `runtime/state.json`.
Reports are written under this campaign's `generated_reports/` directory.
