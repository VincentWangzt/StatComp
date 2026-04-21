# Toy Method Defaults Grid 2026-04-20

- Official runs: 25
- Smoke runs: 5
- Artifact root: `/root/autodl-tmp`
- Generated configs: `configs/generated/toy_method_defaults_20260420`
- Manifest: `campaigns/toy_method_defaults_20260420/manifest.json`
- Smoke manifest: `campaigns/toy_method_defaults_20260420/smoke_manifest.json`
- Queue: `campaigns/toy_method_defaults_20260420/queue_gpu0.txt`

## Local Commands

```powershell
.\.venv\Scripts\python.exe scripts\generate_toy_method_defaults_grid.py
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase smoke --manifest campaigns\toy_method_defaults_20260420\smoke_manifest.json --campaign-dir campaigns\toy_method_defaults_20260420
.\.venv\Scripts\python.exe scripts\summarize_grid_benchmark.py --phase official --manifest campaigns\toy_method_defaults_20260420\manifest.json --campaign-dir campaigns\toy_method_defaults_20260420
.\.venv\Scripts\python.exe scripts\summarize_toy_method_defaults_grid.py
.\.venv\Scripts\python.exe scripts\fetch_grid_benchmark_artifacts.py --remote-repo ~/ruivi-toy-method-defaults --campaign-slug toy_method_defaults_20260420 --remote-artifact-root /root/autodl-tmp
```

## Remote Queue Commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ruivi
mkdir -p /root/autodl-tmp/results/toy_method_defaults_20260420 /root/autodl-tmp/tb_logs/toy_method_defaults_20260420
python scripts/generate_toy_method_defaults_grid.py
python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0 --manifest campaigns/toy_method_defaults_20260420/smoke_manifest.json --campaign-dir campaigns/toy_method_defaults_20260420
python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0 --manifest campaigns/toy_method_defaults_20260420/manifest.json --campaign-dir campaigns/toy_method_defaults_20260420
```
