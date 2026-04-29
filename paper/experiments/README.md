# Experiment Section Scaffold

This directory is a standalone LaTeX workspace for drafting only the paper's
experiment section before it is merged into the remote full paper source.

## Files

- `main.tex`: local wrapper for isolated compilation.
- `experiment.tex`: the experiment-section content to merge later.
- `neurips_2026.sty`: local paper style used by the standalone wrapper.
- `ref.bib`: local bibliography database for experiment-section drafting.
- `AGENTS.md`: workflow guidance for future agent edits in this directory.

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
3. Keep `experiment.tex` focused on the experiment section so it can be merged
   into the full paper with minimal editing.

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
