# Langevin/BNN Knob Grid 2026-04-22

- Official runs: 18
- Smoke runs: 4
- Artifact root: `/root/autodl-tmp`
- Generated configs: `configs/generated/langevin_bnn_knob_grid_20260422`
- Manifest: `campaigns/langevin_bnn_knob_grid_20260422/manifest.json`
- Smoke manifest: `campaigns/langevin_bnn_knob_grid_20260422/smoke_manifest.json`
- Queue: `campaigns/langevin_bnn_knob_grid_20260422/queue_gpu0.txt`

## Local Commands

```powershell
.\.venv\Scripts\python.exe scripts\generate_langevin_bnn_knob_grid.py
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase smoke --manifest campaigns\langevin_bnn_knob_grid_20260422\smoke_manifest.json --campaign-dir campaigns\langevin_bnn_knob_grid_20260422
.\.venv\Scripts\python.exe scripts\show_grid_status.py --phase official --manifest campaigns\langevin_bnn_knob_grid_20260422\manifest.json --campaign-dir campaigns\langevin_bnn_knob_grid_20260422
.\.venv\Scripts\python.exe scripts\summarize_grid_benchmark.py --phase official --manifest campaigns\langevin_bnn_knob_grid_20260422\manifest.json --campaign-dir campaigns\langevin_bnn_knob_grid_20260422
.\.venv\Scripts\python.exe scripts\render_langevin_bnn_knob_grid_report.py
.\.venv\Scripts\python.exe scripts\fetch_grid_benchmark_artifacts.py --remote-repo ~/ruivi --campaign-slug langevin_bnn_knob_grid_20260422 --remote-artifact-root /root/autodl-tmp
```

## Remote Queue Commands

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate ruivi
mkdir -p /root/autodl-tmp/results/langevin_bnn_knob_grid_20260422 /root/autodl-tmp/tb_logs/langevin_bnn_knob_grid_20260422
python scripts/run_grid_queue.py --phase smoke --queue gpu0 --gpu 0 --manifest campaigns/langevin_bnn_knob_grid_20260422/smoke_manifest.json --campaign-dir campaigns/langevin_bnn_knob_grid_20260422
python scripts/run_grid_queue.py --phase official --queue gpu0 --gpu 0 --manifest campaigns/langevin_bnn_knob_grid_20260422/manifest.json --campaign-dir campaigns/langevin_bnn_knob_grid_20260422
```
