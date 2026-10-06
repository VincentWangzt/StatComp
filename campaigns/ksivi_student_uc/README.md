# Canonical KSIVI Student-t update

The canonical `configs/ksivi_student_uc.yaml` now enables 25,000-step linear
annealing, coefficient-0.05 log-density regularization during warmup, and
detachment of the fitted median bandwidth. Sample-input gradients remain
enabled. The experiment retains 50,000 iterations, batch size 128, learning rate
0.001, the affine-invariant Riesz kernel, and the existing variational family.

`bash scripts/run_ksivi_student_canonical.sh` completes seeds 42 through 46 in
`campaigns/toy_scatter_ksivi_detached_annealing/`. Seeds 42, 43, and 44 reuse the
matching completed controls; their effective configuration hashes are unchanged.
Seeds 45 and 46 complete the five-seed set. The canonical config disables
periodic metrics and plot sampling, matching all five runs and preserving their
training random stream. Metrics are evaluated from final checkpoints.
Saved samples and checkpoints retain their original paths under `results/`.

On the server, run `python scripts/finalize_ksivi_student_uc.py` after training.
It validates the five saved configs, checkpoints, and sample files, evaluates
each final checkpoint through the existing paper evaluation workflow, and
records raw and aggregate results under `generated_reports/finalization/`.
The report uses negative ELBO as the KL-style metric, and truncated sliced W2
with coordinate threshold 8. Both uncertainties are standard errors across all
five seeds. Seed 43 is included in these quantitative results.

The evaluator replaces the KSIVI / `student_uc` group in the paper's cached
evaluation outputs and regenerates the toy metric tables. Other method-target
groups retain their existing results. The toy method-grid caption states the
five-seed exception to the existing ten-seed protocol. End-to-end timing for
this group is taken from its controller summary.

Generate the scatter grid with
`python scripts/run_finalization.py --config configs/finalization/toy_scatter_grid.yaml --only scatter_grid`.
Its method-target manifest override selects the canonical KSIVI Student-t seed 42. The
original 15-run manifest retains its historical training records. The plot
generator exports the PDF beside `paper/experiments/experiments.tex`.
The default finalization config uses the same override when selecting runs for
reevaluation. Cached rows retain their source manifest so subsequent table
generation uses the corresponding training times.

Generate and commit report artifacts on the remote server from pushed source,
then push and pull them locally through git.
