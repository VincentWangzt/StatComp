# Additional AISIVI eight-Gaussian seeds

This campaign runs only AISIVI on `8_gaussians` with seeds 45, 43, 42, and 46.
It uses the same configuration as the earlier qualitative trials: 10,000 main
iterations with intermediate metrics disabled and periodic plots disabled.
Training outputs and TensorBoard logs remain under `results/` and `tb_logs/`.

Run `bash scripts/run_aisivi_8_gaussians_additional_seeds.sh` under the `ruivi`
environment in tmux. The manifest retains all four runs. The comparison places
seeds in the requested order and uses the same fixed target axes, target contours,
and deterministic 2,000-point subsets as the paper grid, with no in-range labels.
`seed_summary.csv` records statistics over every saved sample. An optional
`--full-range` plot shows the full plotted cloud for runs that drift outside the
target axes.

The four final VI models and their saved samples are finite. Seeds 45 and 46 have
non-finite final reverse-flow parameters and exhibit support drift. The summary
records the finiteness of both models for every seed.

Generate and commit the reports on the experiment server from pushed code, then
pull the artifacts locally through Git.
