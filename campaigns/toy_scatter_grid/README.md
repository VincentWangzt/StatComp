# Toy scatter grid

The original camera-ready campaign contains exactly 15 targeted training runs: SIVI,
KSIVI, AISIVI, UIVI, and DIVI on `x_shaped`, `student_uc`, and `8_gaussians`.
Each original method-target pair uses one seed:
seed 42 for SIVI, KSIVI, UIVI, and DIVI; seed 44 for AISIVI.

Run `bash scripts/run_toy_scatter_grid.sh` under the `ruivi` environment in tmux.
Training uses the checked-in method configs, including 50,000 iterations for
KSIVI and 10,000 for the other methods. Intermediate metric evaluation and contour
plots are disabled for this figure-only campaign. Training outputs and TensorBoard
logs stay under `results/toy_scatter_grid/` and `tb_logs/toy_scatter_grid/`.
The manifest records the effective configuration hashes and final artifact paths.

The figure has three rows and six columns: ground truth, SIVI, KSIVI, AISIVI,
UIVI, and DIVI. Each panel contains 2,000 samples with the existing target contours.
Ground-truth samples come from the existing exact baselines and require no training
run. Plot subsampling is deterministic for each method-target panel and remains
unchanged when other columns are added or reordered. The generator exports the
PDF beside `paper/experiments/experiments.tex`, where it is included by filename.
Quantitative tables use the existing ten-seed results, the five-seed canonical
KSIVI Student-t update, and the supplementary measurements supplied in the rebuttal.

Additional qualitative seed trials are run with
`bash scripts/run_toy_scatter_seed_trials.sh`. The default seeds are 0, 1, and 2,
on only AISIVI / `8_gaussians` and KSIVI / `student_uc`. Their separate manifests
preserve every trial independently of the original 15-run campaign. Selected
panels use per-target seed overrides: seed 43 for AISIVI / `8_gaussians` and seed 42
from the annealed, regularized, detached-bandwidth KSIVI / `student_uc` campaign.
All three initial AISIVI trials have samples within the plotting bounds.
The previously selected KSIVI seed 1 has the largest in-bounds fraction among its three trials
(2.22% of the 10,000 saved samples), with 46 of the 2,000 plotted points visible;
the run still exhibits support drift. `scripts/summarize_toy_scatter_seed_trials.py`
records all completed trials and the original two panels in `seed_trials.csv`.

Four further AISIVI / `8_gaussians` runs with seeds 45, 43, 42, and 46 are recorded
in `campaigns/toy_scatter_aisivi_additional/manifest.json`. The selected seed 43 has
finite final model parameters and 99.95% of saved samples within the plot bounds.

Three further KSIVI / `student_uc` runs with seeds 43, 45, and 46 are recorded in
`campaigns/toy_scatter_ksivi_additional/manifest.json`. They use the same Riesz
kernel and training setup as the earlier Student-t trials. All have finite final
VI parameters and samples, with in-bounds fractions of 1.04%, 7.64%, and 2.53%,
respectively. Each run exhibits substantial drift from the target. The seed audit
includes these trials. The selected Student-t panel now uses the canonical
annealing + warmup regularization setup with detached median bandwidth, seed 42.
Its five-seed campaign is recorded in
`campaigns/toy_scatter_ksivi_detached_annealing/manifest.json`, and its quantitative
evaluation is documented in `campaigns/ksivi_student_uc/README.md`.

After the original campaign and both sets of additional trials complete, generate the
selected figure and audit report with:

```bash
python scripts/summarize_toy_scatter_seed_trials.py
python scripts/run_finalization.py --config configs/finalization/toy_scatter_grid.yaml --only scatter_grid
```

The older figure campaign performed periodic metric and plot sampling. These
calls consume the same PyTorch random stream as training, so disabling them
changes the stochastic training trajectory even with the same initial seed.

Generate report artifacts on the experiment server from pushed code, commit and
push them there, then pull them locally, as required by the repository workflow.
