#!/usr/bin/env bash
# Three further qualitative KSIVI Student-t trials, with the existing Riesz setup.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
campaign=toy_scatter_ksivi_additional
mkdir -p "results/$campaign" "tb_logs/$campaign"
python scripts/run_default_config_grid_sweep.py \
  --campaign-slug "$campaign" \
  --results-dir "results/$campaign" --tb-dir "tb_logs/$campaign" \
  --seeds 43 45 46 --methods ksivi --targets student_uc --gpus 0 \
  --extra-override train.log.metric_log_freq=0 \
  --extra-override train.plot.freq=1000000000 --retry-failed
python scripts/compare_ksivi_student_seeds.py
