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

All five seeds completed with finite final weights and 10,000 saved samples at
iteration 50,000. There were no nonfinite-update warnings. The full paper
evaluation produced finite metrics without constrained-sampling fallbacks.

| Metric | Mean ± SE across five seeds | Seed 42 |
| --- | ---: | ---: |
| KL-style negative ELBO | 4.511571 ± 1.786559 | 2.729987 |
| Truncated W2, coordinate threshold 8 | 0.748537 ± 0.628046 | 0.082846 |

Seeds 42, 44, 45, and 46 have KL-style values between 2.711 and 2.735 and
truncated W2 between 0.083 and 0.149. Seed 43 has values 11.658 and 3.260,
respectively, and contributes to the larger standard errors. All five seeds
are included in the reported averages.

Independent checks reproduced the means and sample-standard-deviation divided
by square root of five. The other method-target groups are identical in the
cached per-run, raw, and aggregate evaluations. The updated scatter PNG changes
only the KSIVI Student-t panel; the other 17 panels are pixel-identical. The
campaign PDF and the PDF exported beside the paper source are byte-identical.

The section-only manuscript compiled locally to 13 pages using
`latexmk -pdf -outdir=build main.tex`. Rendered pages 2, 5, 7, and 8 were visually
checked, including the updated scatter grid, five-seed protocol, metric table,
and training description. The wrapper retains its three existing unresolved
theory labels: `assump:bounded_score`, `assump:bounded_reparam`, and
`app:linear-growth`. The user's manual edits to `experiments.tex` are preserved.
