# Toy scatter grid

The camera-ready scatter figure uses exactly 15 targeted training runs: SIVI,
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
run. Plot subsampling is deterministic. Quantitative tables retain the existing
ten-seed results and the supplementary measurements supplied in the rebuttal.

Generate report artifacts on the experiment server from pushed code, commit and
push them there, then pull them locally, as required by the repository workflow.
