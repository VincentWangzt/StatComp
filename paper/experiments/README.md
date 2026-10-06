# Experiment Section Scaffold

This directory is a standalone LaTeX workspace for drafting the paper's
experiment section and appendix before they are merged into the remote full
paper source.

## Files

- `main.tex`: local wrapper for isolated compilation.
- `experiments.tex`: the main experiment-section body (Section 5 in the paper).
- `experiment_appendix.tex`: the experiment appendix with detailed configurations, metrics, and per-target settings.
- `neurips_2026.sty`: local paper style used by the standalone wrapper.
- `ref.bib`: local bibliography database for experiment-section drafting.
- `AGENTS.md`: workflow guidance for future agent edits in this directory.
- `CLAUDE.md`: guidance for Claude Code sessions in this directory.

No fake macro, figure, or table files are included here. Keep `neurips_2026.sty`
and `ref.bib` aligned with the remote paper source when those assets change.

## Compile Locally

From this directory:

```powershell
latexmk -pdf -outdir=build main.tex
```

The expected PDF is:

```text
build/main.pdf
```

Treat files under `build/` as verification artifacts, not manuscript source.

## Future Paper Integration

When the remote full paper source is available:

1. Replace or reconcile the standalone wrapper preamble in `main.tex` with the
   remote paper class, packages, style files, and macros.
2. Keep bibliography wiring pointed at the real remote `.bib` and bibliography
   style files.
3. Keep `experiments.tex` focused on the main experiment section and
   `experiment_appendix.tex` focused on the appendix details, so each can be
   merged into the full paper with minimal editing.

Generated figures and tables should be referenced from campaign/finalization
outputs rather than copied into this directory. Expected default locations are:

```text
../../campaigns/default_config_grid/generated_reports/finalization/figures/
../../campaigns/default_config_grid/generated_reports/finalization/tables/
```

For example, after those artifacts exist:

```latex
\includegraphics{\finalizationfigdir/toy_scatter_grid.pdf}
\input{\finalizationtabledir/toy_metrics.tex}
```

## Camera-ready results and analysis

The camera-ready source and its committed toy scatter PDF are retained from
`plot-camera-ready`. The appendix includes the five-method toy table, fixed-DIVI
score comparison, target-score discrepancy, reverse-update ablations, and
four-layer RealNVP comparison. Table captions use generic mean/SE wording;
actual seed-count exceptions are recorded in the shared training protocol.

The score-error rows use shared frozen DIVI checkpoints. The latency row
preserves a separate benchmark from `rebuttal-0726` using each method's own
10,000-step checkpoint, seed 42, RTX 3090, and 100 timed calls. Its uncertainties
are sample standard deviations across those calls, recovered from the original
`score_estimator_timing_x_shaped` report at `fbd0b2e`; they are not five-seed
standard errors. The measured means are unchanged.

The release entrypoints `scripts/run_score_approximation.py` and
`scripts/run_score_estimator_timing.py` both use shared DIVI checkpoints and
AISIVI proposals refitted against them. Their outputs require a new experiment
run before replacing manuscript numbers. Preserved own-trajectory diagnostics
and reverse-update ablation results are separate experiments. See
[`finalization/README.md`](../../finalization/README.md) for inputs and commands.

The local wrapper does not contain the full paper's assumption definitions or
linear-growth appendix. Its references to those labels remain unresolved in
isolated compilation and should resolve when these sections are included in
the complete paper. The local PDF is only a layout-verification artifact.
