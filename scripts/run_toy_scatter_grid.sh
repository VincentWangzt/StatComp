#!/usr/bin/env bash
# Exactly 15 training runs: one selected seed for each of five methods on three targets.
# Run under the ruivi environment, inside tmux on the experiment server.
set -euo pipefail
cd "$(dirname "$0")/.."
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
mkdir -p results/toy_scatter_grid tb_logs/toy_scatter_grid

python scripts/run_default_config_grid_sweep.py \
  --campaign-slug toy_scatter_grid \
  --results-dir results/toy_scatter_grid \
  --tb-dir tb_logs/toy_scatter_grid \
  --seeds 42 --seed-override aisivi=44 \
  --methods sivi ksivi aisivi uivi dsivi \
  --targets x_shaped student_uc 8_gaussians --gpus 0 \
  --extra-override train.log.metric_log_freq=0 \
  --extra-override train.plot.freq=1000000000 \
  "$@"

# Refuse to publish an incomplete or expanded campaign.
python - <<'PY'
from finalization.artifacts import completed_runs, load_manifest
from finalization.config import load_config
cfg = load_config("configs/finalization/toy_scatter_grid.yaml")
manifest = load_manifest(cfg.campaign.manifest_path)
records = completed_runs(manifest)
expected = {(44 if method == "AISIVI" else 42, method, target)
            for method in cfg.selection.methods for target in cfg.selection.scatter_targets}
actual = {(r.seed, r.method, r.target) for r in records}
assert len(manifest) == len(records) == 15 and actual == expected, (len(manifest), len(records), actual)
PY
# Render the original 15 runs independently of the additional seed trials.
python scripts/run_finalization.py --config configs/finalization/toy_scatter_grid.yaml --only scatter_grid \
  --set 'campaign.additional_manifest_paths=[]' \
  --set selection.target_seed_overrides.AISIVI.8_gaussians=44 \
  --set selection.target_seed_overrides.KSIVI.student_uc=42
