# Constant initial variance for ConditionalGaussian

Repeat the canonical seed-42/43/44 comparison with only the variance-head
initialization changed. The last layer's variance-output weights start at zero;
its variance-output biases start at inverse-softplus(1.31326162815094), about 1.
Every epsilon therefore starts with variance 1.31326162815094 in each dimension,
matching the default ConditionalGaussianGlobal initialization. The hidden layers
and mean head keep their seeded random initialization. All parameters remain
trainable, so variance can depend on epsilon after optimization begins.

The canonical model remains ConditionalGaussian with two width-128 SiLU hidden
layers, effective noise dimension 2, batch size 128, learning rate 0.001 with
StepLR(1000, 0.9), annealing enabled for 25,000 steps, and 50,000 total steps.
Metric evaluation is temporarily disabled. Three runs launch in parallel and
save 10,000 samples and a contour plot every 5,000 steps. The 3x10 report uses
those native plots, with columns 5,000 through 50,000.

The new `vi_model.variance_init` option is opt-in. Its canonical value is null,
which preserves the existing random variance initialization for softplus models
and existing log_var_init behavior for logvar models.

Run from the repository root on the GPU host inside tmux:

```bash
/root/miniconda3/envs/ruivi/bin/python -u campaigns/ksivi_x_shaped_evolution_20261006/run_campaign.py \
  --rounds seeds --variance-init 1.31326162815094 \
  --results-root /root/ruivi/results/ksivi_x_shaped_constant_variance_init_20261006 \
  --tb-root /root/ruivi/tb_logs/ksivi_x_shaped_constant_variance_init_20261006 \
  --report-root campaigns/ksivi_x_shaped_constant_variance_init_20261006/generated_reports
```

Reports validate saved configurations, sample tensors, and final checkpoints.
The manifest also records initial variance and final conditional variance
statistics across the saved epsilon samples.

Initialization and trainability checks can be rerun locally:

```powershell
.\.venv\Scripts\python.exe campaigns\ksivi_x_shaped_constant_variance_init_20261006\check_initialization.py
```
