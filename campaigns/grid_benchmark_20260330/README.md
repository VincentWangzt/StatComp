# Grid Benchmark 2026-03-30

- Official runs: 216
- Smoke runs: 5
- Generated configs: `configs/generated/grid_benchmark_20260330`
- Manifest: `campaigns/grid_benchmark_20260330/manifest.json`
- Smoke manifest: `campaigns/grid_benchmark_20260330/smoke_manifest.json`
- Markdown log: `grid_benchmark_2026-03-30.md`

## Local Commands

```powershell
.\.venv\Scripts\python.exe scripts\generate_grid_benchmark.py
.\.venv\Scripts\python.exe scripts\fetch_grid_benchmark_artifacts.py
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase official
.\.venv\Scripts\python.exe scripts\summarize_grid_benchmark.py --phase official
.\.venv\Scripts\python.exe scripts\manual_check_grid_benchmark.py
```

## Remote Queue Commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ruivi
python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0
python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0
python scripts/run_grid_queue.py --phase official --queue gpu1 --gpu 1
```
