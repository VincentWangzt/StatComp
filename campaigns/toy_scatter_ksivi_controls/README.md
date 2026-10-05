# KSIVI Student-t annealing and regularization comparison

These two additional runs compare config controls against the completed seed 43
run in `campaigns/toy_scatter_ksivi_additional/manifest.json`. Each uses 50,000
iterations, batch size 128, learning rate 0.001, the Riesz kernel, and the existing
variational family and bandwidth-gradient behavior.

| Condition | Annealing | Log-density regularization |
| --- | --- | --- |
| Original setup (existing run) | Disabled | Inactive with `warmup_only` |
| Annealing + warmup regularization | Enabled, linear over 25,000 iterations | Coefficient `0.05 * annealing_factor` while the factor is below 1 |
| Always-on regularization | Disabled | Coefficient 0.05 throughout training |

The first new run uses `train.annealing.enabled=true` and
`train.ksivi.log_p_reg_mode=warmup_only`. Enabling annealing also activates the
existing warmup regularization. To test annealing alone, use `log_p_reg=0`.

The second new run uses `train.annealing.enabled=false` and
`train.ksivi.log_p_reg_mode=always`. The regularized loss is
`KSD_loss - 0.05 * mean(logp(z))` at every iteration. A cutoff independent of
annealing would require an additional runner setting.

Run `bash scripts/run_ksivi_student_controls.sh` under the `ruivi` environment
on the experiment server. The two runs have separate manifests under
`campaigns/toy_scatter_ksivi_annealing/` and `campaigns/toy_scatter_ksivi_logp/`.
Training outputs and TensorBoard logs stay under the corresponding existing
`results/` and `tb_logs/` folders. Periodic metrics and contour sampling are disabled
as in the earlier qualitative runs.

`scripts/compare_ksivi_student_controls.py` checks the effective saved configs
before comparing final 10,000-sample files. It reuses the seed comparison's target
reference, 256 projection directions for empirical sliced-W1, and deterministic
2,000-point plot subsets. The standard view uses the paper's [-5, 5] axes, and
the second view shows the full extent of each plotted subset.

Generate the reports remotely from pushed code, commit and push the completed
manifests and reports there, then pull them into the local checkout.

Both additional runs completed with finite final VI parameters and samples. The
single-seed comparison at 50,000 iterations is:

| Condition | Samples within [-5, 5]^2 | Empirical sliced-W1 |
| --- | ---: | ---: |
| Original setup | 1.04% | 28.691 |
| Annealing + warmup regularization | 41.32% | 4.920 |
| Always-on regularization, annealing disabled | 67.43% | 3.062 |

Both controls reduce drift in this seed. Persistent regularization gives the
smaller empirical distance. The final sample clouds remain displaced from the
target; the exact baseline has 99.34% of its samples within the same plot bounds.
