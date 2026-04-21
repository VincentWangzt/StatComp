# DSIVI Toy Backbone Grid 2026-04-18

- Official runs: 40
- Smoke runs: 8
- Generated configs: `configs/generated/dsivi_backbone_toys_20260418`
- Manifest: `campaigns/dsivi_backbone_toys_20260418/manifest.json`
- Smoke manifest: `campaigns/dsivi_backbone_toys_20260418/smoke_manifest.json`
- Queue: `campaigns/dsivi_backbone_toys_20260418/queue_gpu0.txt`

## Local Commands

```powershell
.\.venv\Scripts\python.exe scripts\generate_dsivi_backbone_toy_grid.py
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase smoke --manifest campaigns\dsivi_backbone_toys_20260418\smoke_manifest.json --campaign-dir campaigns\dsivi_backbone_toys_20260418
.\.venv\Scripts\python.exe scripts\summarize_grid_benchmark.py --phase official --manifest campaigns\dsivi_backbone_toys_20260418\manifest.json --campaign-dir campaigns\dsivi_backbone_toys_20260418
.\.venv\Scripts\python.exe scripts\fetch_grid_benchmark_artifacts.py --remote-repo ~/ruivi-dsivi-backbone-toys --campaign-slug dsivi_backbone_toys_20260418
```

## Remote Queue Commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ruivi
python scripts/generate_dsivi_backbone_toy_grid.py
python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0 --manifest campaigns/dsivi_backbone_toys_20260418/smoke_manifest.json --campaign-dir campaigns/dsivi_backbone_toys_20260418
python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0 --manifest campaigns/dsivi_backbone_toys_20260418/manifest.json --campaign-dir campaigns/dsivi_backbone_toys_20260418
```
