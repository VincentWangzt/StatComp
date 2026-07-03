# Experiment Reference Papers

This folder keeps local copies of the papers used as style and setup references for the experiment section draft.

## What Is Here

- `2506.05088-kpg-sivi.pdf` and `2506.05088-kpg-sivi.md`: KPG-SIVI.
- `2506.03839v1-revisiting-uivi-aisivi.pdf` and `2506.03839v1-revisiting-uivi-aisivi.md`: revisiting UIVI and AISIVI.
- `2405.18997-ksivi.pdf` and `2405.18997-ksivi.md`: KSIVI.
- `page_checks/`: rendered PNGs of representative pages that were manually compared against the Markdown conversions.
- `conversion_check.md`: notes on conversion, page-count checks, visual spot checks, and known limitations.

## How To Use The Markdown Files

Use the `.md` files for fast reading and search while drafting or reviewing `../experiment.tex`.  They include explicit `## Page N` markers, so search hits can be traced back to the original PDF page.

Useful searches:

```powershell
Select-String -Path .\*.md -Pattern "Experiments|Experimental details|Implementation Details|BNN|Langevin|UCI"
Select-String -Path .\*.md -Pattern "computational environment|VRAM|PyTorch|batch size|iterations"
Select-String -Path .\*.md -Pattern "Bayesian Neural Network|Conditioned Diffusion|Toy Experiments"
```

For careful style comparison, start with these checked pages:

- KPG-SIVI: pages 7 and 14.
- AISIVI/UIVI revisiting: pages 6 and 13.
- KSIVI: pages 6, 18, and 19.

## When To Use The PDFs

The PDFs remain the source of truth for exact equations, tables, figure layout, and citation context.  Use the Markdown for navigation and prose-level comparison, then open the corresponding PDF page when copying notation, checking a table, or matching the visual organization of an experiment section.

The Markdown extraction is intentionally readable rather than layout-faithful.  Complex equations and tables are linearized, and figures are represented only by extracted captions and nearby prose.
