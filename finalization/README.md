# Finalization Module

Processes experiment results from the `default_config_grid` campaign into
publication-ready figures and LaTeX tables.

## Usage

Run the full pipeline:

```bash
python scripts/run_finalization.py
```

Run selected modules with `--only` (may be repeated):

```bash
python scripts/run_finalization.py --only scatter_grid
python scripts/run_finalization.py --only evaluate --set evaluation.overwrite=true
python scripts/run_finalization.py --only kl_iteration_grid --only kl_time_grid
```

Override configuration values with `--set`:

```bash
python scripts/run_finalization.py --set selection.seeds=[42] --set evaluation.device=cpu
```

The default configuration is at `configs/finalization/default_config_grid.yaml`.
A custom config can be passed with `--config <path>`.

Available `--only` modules:

| Module | Description |
|--------|-------------|
| `evaluate` | Re-evaluate checkpoints and write per-run metrics |
| `scatter_grid` | Toy target sample scatter plots |
| `scatter_hist_grid` | Scatter plots with marginal histograms |
| `langevin_trace_grid` | Langevin target trace plots |
| `kl_iteration_grid` | KL divergence vs. iteration curves |
| `kl_time_grid` | KL divergence vs. wall-clock time curves |
| `grad_norm_iteration_grid` | Gradient norm vs. iteration curves |
| `weight_norm_iteration_grid` | Weight norm vs. iteration curves |
| `m_eps_iteration_grid` | Mixing samples (m_eps) vs. iteration curves |
| `score_4th_moment_iteration_grid` | Score fourth moments vs. iteration curves |
| `score_diff_l2_fourth_iteration_grid` | E[\\|\\|score_p - score_q\\|\\|^4] vs. iteration curves |
| `toy_tables` | Summary metrics table for toy targets |
| `toy_method_grid` | Per-method breakdown table for toy targets |
| `langevin_table` | Metrics table for the Langevin target |
| `student_edge_table` | Edge-length W2 table for Student-UC target |
| `bnn_table` | RMSE/NLL table for BNN targets |

## Outputs

All outputs are written to:

```
campaigns/default_config_grid/generated_reports/finalization/
├── figures/          # PNG and PDF figures
├── tables/           # LaTeX .tex table files
├── reevaluation_summary.csv
└── finalization_report.md
```
