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
