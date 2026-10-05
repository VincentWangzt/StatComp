# Toy scatter grid

The original camera-ready campaign contains exactly 15 targeted training runs: SIVI,
KSIVI, AISIVI, UIVI, and DIVI on `x_shaped`, `student_uc`, and `8_gaussians`.
Each method-target pair uses one seed. The existing figure selection is retained:
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
Quantitative tables retain the existing
ten-seed results and the supplementary measurements supplied in the rebuttal.

Additional qualitative seed trials are run with
`bash scripts/run_toy_scatter_seed_trials.sh`. The default seeds are 0, 1, and 2,
on only AISIVI / `8_gaussians` and KSIVI / `student_uc`. Their separate manifests
preserve every trial independently of the original 15-run campaign. Selected
panels use per-target seed overrides. `scripts/summarize_toy_scatter_seed_trials.py`
records all completed trials and the original two panels in `seed_trials.csv`.

The older figure campaign performed periodic metric and plot sampling. These
calls consume the same PyTorch random stream as training, so disabling them
changes the stochastic training trajectory even with the same initial seed.

Generate report artifacts on the experiment server from pushed code, commit and
push them there, then pull them locally, as required by the repository workflow.
