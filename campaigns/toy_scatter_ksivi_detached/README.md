# KSIVI Student-t: detached median bandwidth

This comparison uses seeds 42, 43, and 44 for three setups, giving exactly nine
new runs. Every run uses `train.ksivi.detach_bandwidth=true` and
`train.ksivi.detach_kernel=false`: gradients pass through the Riesz kernel's
sample inputs, while the fitted median bandwidth is held constant during
backpropagation. The bandwidth is refitted at every iteration.

| Setup | Annealing | Log-density regularization |
| --- | --- | --- |
| Annealing + warmup reg. | Linear, 25,000 iterations | `0.05 * annealing_factor` while the factor is below 1 |
| Always-on reg. | Disabled | Coefficient 0.05 for all iterations |
| No reg., no warmup | Disabled | Coefficient 0.0 |

Other settings match the preceding controls: 50,000 iterations, batch size 128,
learning rate 0.001, Riesz kernel, affine inverse-covariance weighting, and the
existing conditional Gaussian variational family. Periodic metrics and contour
sampling are disabled; final samples contain 10,000 points.

Run `bash scripts/run_ksivi_student_detached_bandwidth.sh` under the `ruivi`
environment on the experiment server. One run from each setup executes
concurrently. The setups have separate manifests under
`campaigns/toy_scatter_ksivi_detached_{annealing,logp,plain}/`; experiment outputs
stay under `results/` and TensorBoard logs under `tb_logs/`.

`scripts/compare_ksivi_student_detached_bandwidth.py` checks the saved configs,
final checkpoints, and sample files, then produces three-row scatter grids with
the target and all three seeds. A second grid shows the full extent of each
plotted subset. Each panel uses a deterministic 2,000-sample subset.

The CSV reports retain all nine runs and report mean ± SE across three seeds.
Empirical sliced-W1 uses the same exact-target reference and 256 projection
directions as the preceding comparisons. A paired CSV compares the new runs
with existing runs whose median bandwidth remains differentiable. Prior
unregularized runs are available for seeds 42 and 43; the seed 44 baseline is
marked unavailable. Its absence does not trigger an additional training run.

Create and test source changes locally, commit and push, then sync the remote
repository before starting the launcher. Generate and commit the reports on the
remote server, push them, and pull them into the local checkout.

The source was validated before launch with a controlled Riesz scaling check:
attached and detached medians give identical forward kernel matrices, while
detachment retains a nonzero gradient through the sample inputs. That gradient
agrees with an independently evaluated fixed-bandwidth formula. All three
setups also completed three-iteration CPU smoke runs with finite saved weights
and samples.

The upstream comparison was checked against public revision
`4d995745265b06f37143f156760fa5ebe7ecbbec` of
[longinYu/KSIVI](https://github.com/longinYu/KSIVI).

| Upstream experiment | Kernel | Median bandwidth detached | Score annealing | Log-density regularization |
| --- | --- | --- | --- | --- |
| [Langevin](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/configs/kernel_sivi_langevin_post.yml) | Gaussian | No | Disabled | None |
| [Banana](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/configs/banana.yml) | Gaussian | No | Disabled | None |
| [X-shaped](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/configs/x_shaped.yml) | Gaussian | No | Disabled | None |
| [Multimodal](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/configs/multimodal.yml) | IMQ | Yes | Enabled | None |
| [Student-t](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/configs/student_uc.yml) | Riesz | Yes | Enabled | `0.05 * alpha(t)` for `t < 20000` |
| [Boston BNN](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/configs/kernel_sivi_boston.yml) | Gaussian | No | Disabled | Coefficient 1.0 throughout |

The upstream [kernel implementation](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/utils/kernels.py)
always detaches the IMQ and Riesz medians. For the Gaussian kernel, its
`detach=False` keeps the median differentiable. Sample-input gradients are
enabled in all these configurations. Our existing Gaussian settings have the
same bandwidth-gradient behavior; the new flag provides the upstream IMQ/Riesz
behavior while retaining sample-input gradients.

All four upstream toy configs use 50,000 iterations, batch size 500, learning
rate 0.001, and a learning-rate multiplier of 0.9 every 1,000 iterations. The
Langevin config uses 100,000 iterations, batch size 128, learning rate 0.0002
for both the mean network and global variance, and a multiplier of 0.9 every
10,000 iterations. Our toy batch size is 128. Our Langevin config enables
50,000-step score annealing and uses learning rate 0.001. Our Banana and
X-shaped configs also enable score annealing. The upstream Boston config uses
20,000 iterations and batch size 100; our BNN configs use 100,000 iterations
and batch size 128. Both specify 100 pretraining steps, no score annealing,
coefficient-1 persistent log-density regularization, and EMA decay 0.999.

The upstream [annealing function](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/utils/annealing.py)
uses `min(1, 0.1 + t / 25000)` for the annealed toy configs, reaching 1 at
22,500 iterations. Its [Student-t trainer](https://github.com/longinYu/KSIVI/blob/4d995745265b06f37143f156760fa5ebe7ecbbec/sivistein_t_student.py)
stops regularization at iteration 20,000. Our linear schedule reaches 1 at
25,000 iterations, and `warmup_only` regularization stops at that point.
The current trials isolate bandwidth detachment within our existing setup.

All nine requested runs completed at 50,000 iterations with finite final VI
parameters and 10,000-sample files. There were no skipped updates due to
nonfinite losses. Saved configs and matched prior configs pass the report
script's checks.

| Setup | Seed 42: sliced-W1 | Seed 43: sliced-W1 | Seed 44: sliced-W1 | Mean ± SE (three seeds) |
| --- | ---: | ---: | ---: | ---: |
| Annealing + warmup reg. | 0.047 | 8.949 | 0.056 | 3.017 ± 2.966 |
| Always-on reg. | 0.507 | 0.706 | 3.547 | 1.587 ± 0.982 |
| No reg., no warmup | 3.836 | 7.638 | 6.054 | 5.843 ± 1.103 |

Mean coverage within [-5, 5]^2 is 78.04% ± 21.78 percentage points for annealing
plus warmup regularization, 82.40% ± 11.61 percentage points for always-on
regularization, and 42.46% ± 9.81 percentage points for the unregularized setup.
The uncertainties are standard errors across three seeds. Coverage and distance
use all 10,000 final samples.

The annealing setup gives close visual matches to the target for seeds 42 and
44. Seed 42 has the smallest empirical sliced-W1 among the nine runs. Seed 43
in the same setup remains displaced and overdispersed. Always-on regularization
has the smallest three-seed mean distance; its seed 44 run remains displaced.
All three unregularized runs show displacement or excess spread.

The paired comparison shows annealing seed 42 improving from 4.818 to 0.047
and seed 44 from 4.500 to 0.056 with bandwidth detachment. Annealing seed 43
changes from 4.920 to 8.949. Always-on seed 42 changes from 3.443 to 0.507,
seed 43 from 3.062 to 0.706, and seed 44 from 3.556 to 3.547. Unregularized
seeds 42 and 43 change from 31.995 and 28.691 to 3.836 and 7.638, respectively.
The unregularized seed 44 has no prior matched run. The two available prior
unregularized configs specify coefficient 0.05 with annealing disabled and
`warmup_only`, which makes their regularization term inactive; the new runs
explicitly set the coefficient to zero.

The generated reports under `generated_reports/finalization/` are:

- `seed_summary.csv`: all nine completed runs and their settings.
- `aggregate_summary.csv`: three-seed mean and SE for each setup.
- `bandwidth_paired_comparison.csv`: the corresponding prior and new results,
  with the missing unregularized seed 44 baseline marked unavailable.
- `figures/ksivi_student_detached_bandwidth.png`: all three seeds and setups.
- `figures/ksivi_student_detached_bandwidth_full_range.png`: the full extent of
  each plotted subset.
- `figures/ksivi_student_detached_bandwidth_best_seeds.png`: the target and the
  minimum-distance seed within each setup, all seed 42 in this comparison.
