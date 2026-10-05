#!/usr/bin/env bash
# The four AISIVI seeds requested for an additional eight-Gaussian comparison.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
campaign=toy_scatter_aisivi_additional
mkdir -p "results/$campaign" "tb_logs/$campaign"
python scripts/run_default_config_grid_sweep.py \
  --campaign-slug "$campaign" \
  --results-dir "results/$campaign" --tb-dir "tb_logs/$campaign" \
  --seeds 45 43 42 46 --methods aisivi --targets 8_gaussians --gpus 0 \
  --extra-override train.log.metric_log_freq=0 \
  --extra-override train.plot.freq=1000000000 --retry-failed
python scripts/compare_aisivi_8_gaussians_seeds.py
