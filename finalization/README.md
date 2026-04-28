# Finalization Module

Run the default one-pass workflow on the remote repo after syncing code:

```bash
python scripts/run_finalization.py
```

Useful overrides:

```bash
python scripts/run_finalization.py --only scatter_grid
python scripts/run_finalization.py --only evaluate --set evaluation.overwrite=true
python scripts/run_finalization.py --set selection.seeds=[42] --set evaluation.device=cpu
```

Default outputs are written under:

```text
campaigns/default_config_grid/generated_reports/finalization/
```
