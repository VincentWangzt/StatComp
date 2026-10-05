#!/usr/bin/env bash
# Additional qualitative trials on only the two sparse method-target pairs.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
if [ "$#" -eq 0 ]; then
  seeds=(0 1 2)
else
  seeds=("$@")
fi

for method in aisivi ksivi; do
  target=8_gaussians
  if [ "$method" = ksivi ]; then
    target=student_uc
  fi
  campaign="toy_scatter_seed_${method}"
  mkdir -p "results/$campaign" "tb_logs/$campaign"
  python scripts/run_default_config_grid_sweep.py \
    --campaign-slug "$campaign" \
    --results-dir "results/$campaign" --tb-dir "tb_logs/$campaign" \
    --seeds "${seeds[@]}" --methods "$method" --targets "$target" --gpus 0 \
    --extra-override train.log.metric_log_freq=0 \
    --extra-override train.plot.freq=1000000000 --retry-failed
done
