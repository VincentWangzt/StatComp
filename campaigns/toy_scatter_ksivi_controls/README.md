# KSIVI Student-t annealing and regularization comparison

This comparison began with two seed 43 control runs against the original run in
`campaigns/toy_scatter_ksivi_additional/manifest.json` and was expanded to five
matched seeds per control. Each uses 50,000 iterations, batch size 128, learning
rate 0.001, the Riesz kernel, and the existing variational family and
bandwidth-gradient behavior.

| Condition | Annealing | Log-density regularization |
| --- | --- | --- |
| Original setup (existing run) | Disabled | Inactive with `warmup_only` |
| Annealing + warmup regularization | Enabled, linear over 25,000 iterations | Coefficient `0.05 * annealing_factor` while the factor is below 1 |
| Always-on regularization | Disabled | Coefficient 0.05 throughout training |

The annealing setup uses `train.annealing.enabled=true` and
`train.ksivi.log_p_reg_mode=warmup_only`. Enabling annealing also activates the
existing warmup regularization. To test annealing alone, use `log_p_reg=0`.

The always-on setup uses `train.annealing.enabled=false` and
`train.ksivi.log_p_reg_mode=always`. The regularized loss is
`KSD_loss - 0.05 * mean(logp(z))` at every iteration. A cutoff independent of
annealing would require an additional runner setting.

Run `bash scripts/run_ksivi_student_controls.sh` under the `ruivi` environment
on the experiment server. The two setups have separate manifests under
`campaigns/toy_scatter_ksivi_annealing/` and `campaigns/toy_scatter_ksivi_logp/`.
Training outputs and TensorBoard logs stay under the corresponding existing
`results/` and `tb_logs/` folders. Periodic metrics and contour sampling are disabled
as in the earlier qualitative runs.

The launcher now selects seeds 42, 43, 44, 45, and 46 for both setups. Resume
reuses the completed seed 43 runs and adds exactly eight new runs. One job per
setup runs concurrently on the GPU, with separate controller logs under each
campaign's `runtime/` folder.

After all ten condition-seed pairs complete,
`scripts/compare_ksivi_student_control_seeds.py` produces a two-row comparison
across the five seeds, a view of each full plotted cloud, and a comparison of
the minimum empirical sliced-W1 seed within each setup. The per-seed CSV retains
every run. The aggregate CSV reports mean and standard error across five seeds,
using sample standard deviation divided by the square root of five.

`scripts/compare_ksivi_student_controls.py` checks the effective saved configs
before comparing final 10,000-sample files. It reuses the seed comparison's target
reference, 256 projection directions for empirical sliced-W1, and deterministic
2,000-point plot subsets. The standard view uses the paper's [-5, 5] axes, and
the second view shows the full extent of each plotted subset.

Generate the reports remotely from pushed code, commit and push the completed
manifests and reports there, then pull them into the local checkout.

The initial seed 43 control runs completed with finite final VI parameters and
samples. Their comparison at 50,000 iterations is:

| Condition | Samples within [-5, 5]^2 | Empirical sliced-W1 |
| --- | ---: | ---: |
| Original setup | 1.04% | 28.691 |
| Annealing + warmup regularization | 41.32% | 4.920 |
| Always-on regularization, annealing disabled | 67.43% | 3.062 |

Both controls reduce drift in this seed. Persistent regularization gives the
smaller empirical distance. The final sample clouds remain displaced from the
target; the exact baseline has 99.34% of its samples within the same plot bounds.

The expanded comparison completed exactly eight new runs: seeds 42, 44, 45, and
46 for each setup. The existing seed 43 runs were reused. All ten final
checkpoints and 10,000-sample files are finite, and their saved configs pass the
matched-settings checks in `scripts/compare_ksivi_student_control_seeds.py`.

| Seed | Annealing + warmup reg.: sliced-W1 | Always-on reg.: sliced-W1 |
| --- | ---: | ---: |
| 42 | 4.818 | 3.443 |
| 43 | 4.920 | 3.062 |
| 44 | 4.500 | 3.556 |
| 45 | 3.312 | 3.382 |
| 46 | 5.066 | 3.358 |
| Mean ± SE | 4.523 ± 0.317 | 3.360 ± 0.082 |

Mean coverage within [-5, 5]^2 is 47.45% ± 4.77 percentage points for annealing
plus warmup regularization and 63.32% ± 1.19 percentage points for always-on
regularization. These uncertainties are standard errors across five seeds.
Coverage and sliced-W1 use all 10,000 final samples; sliced-W1 uses the same
10,000-sample exact-target reference and 256 projection directions for every run.

Seed 45 has the smallest empirical sliced-W1 within the annealing setup, and seed
43 has the smallest within the always-on setup. Always-on regularization has the
lower five-seed mean distance and standard error. The sample clouds show
displacement and excess spread relative to the target.

The new reports under `generated_reports/finalization/` are:

- `seed_control_summary.csv`: all ten condition-seed records.
- `control_seed_aggregate.csv`: five-seed mean and SE, with the minimum-distance
  seed for each setup.
- `figures/ksivi_student_control_seeds.png`: both setups across all five seeds,
  using the paper's [-5, 5] axes and deterministic 2,000-sample plot subsets.
- `figures/ksivi_student_control_seeds_full_range.png`: the full extent of each
  plotted subset.
- `figures/ksivi_student_control_best_seeds.png`: the target and the
  minimum-distance seed within each setup.
