# Additional KSIVI Student-t seeds

This campaign tests seeds 43, 45, and 46 for the Student-t panel of the toy scatter
grid. It uses the existing `configs/ksivi_student_uc.yaml`: Riesz kernel, 50,000
training iterations, batch size 128, learning rate 0.001, and disabled annealing.
Periodic metric evaluation and contour plots are disabled, as in the earlier
qualitative trials. Each run saves 10,000 final samples.

Run `bash scripts/run_ksivi_student_additional_seeds.sh` under the `ruivi`
environment on the experiment server. Results and TensorBoard logs stay under
`results/toy_scatter_ksivi_additional/` and `tb_logs/toy_scatter_ksivi_additional/`.
The separate manifest preserves the original 15-run campaign and earlier trials.

`scripts/compare_ksivi_student_seeds.py` produces comparisons against the current
seed 1 panel and the exact target baseline. Both views use the same deterministic
2,000-point subsets. The standard view keeps the paper's [-5, 5] axes; the second
view shows the full extent of each plotted subset. No samples are filtered to
increase the number of visible points.

The CSV records bounds coverage, medians, means, standard deviations, the 90th
percentile of distance from the origin, and checkpoint finiteness. It also records
an empirical sliced-W1 distance against a deterministic 10,000-point subset of
the exact baseline, using the same 256 projections for every seed. This comparison
is separate from the manuscript's quantitative tables.

Generate the report artifacts remotely from pushed code, commit and push the
completed manifests, figures, and CSV there, then pull them into the local checkout.
