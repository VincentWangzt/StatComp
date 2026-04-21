# DSIVI Backbone Seed Repeats 2026-04-19

- Official runs: 60
- Smoke runs: 4
- New seeds: `43, 44, 45`
- Reused seed: `42` from `dsivi_backbone_toys_20260418`
- Generated configs: `configs/generated/dsivi_backbone_seed_repeats_20260419`
- Manifest: `campaigns/dsivi_backbone_seed_repeats_20260419/manifest.json`
- Smoke manifest: `campaigns/dsivi_backbone_seed_repeats_20260419/smoke_manifest.json`
- Queue: `campaigns/dsivi_backbone_seed_repeats_20260419/queue_gpu0.txt`

## Local Commands

```powershell
.\.venv\Scripts\python.exe scripts\generate_dsivi_backbone_seed_repeats.py
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase official --manifest campaigns\dsivi_backbone_seed_repeats_20260419\manifest.json --campaign-dir campaigns\dsivi_backbone_seed_repeats_20260419
.\.venv\Scripts\python.exe scripts\fetch_grid_benchmark_artifacts.py --remote-repo ~/ruivi-dsivi-backbone-seed-repeats --campaign-slug dsivi_backbone_seed_repeats_20260419
.\.venv\Scripts\python.exe scripts\summarize_grid_benchmark.py --phase official --manifest campaigns\dsivi_backbone_seed_repeats_20260419\manifest.json --campaign-dir campaigns\dsivi_backbone_seed_repeats_20260419
.\.venv\Scripts\python.exe scripts\summarize_dsivi_seed_repeats.py
```

## Remote Queue Commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ruivi
python scripts/generate_dsivi_backbone_seed_repeats.py
python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0 --manifest campaigns/dsivi_backbone_seed_repeats_20260419/smoke_manifest.json --campaign-dir campaigns/dsivi_backbone_seed_repeats_20260419
python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0 --manifest campaigns/dsivi_backbone_seed_repeats_20260419/manifest.json --campaign-dir campaigns/dsivi_backbone_seed_repeats_20260419
```
