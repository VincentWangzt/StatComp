#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p baselines/exact test_standard_gaussian/logs


python - <<'PY'
import torch
from pathlib import Path

out_path = Path("baselines/exact/standard_gaussian_exact_100k.pt")
if not out_path.exists():
    torch.manual_seed(42)
    samples = torch.randn(100000, 2)
    torch.save(
        {
            "samples": samples,
            "target": "standard_gaussian",
            "source": "exact_sampler",
            "num_samples": 100000,
            "seed": 42,
        },
        out_path,
    )
    print(f"[standard_gaussian] {tuple(samples.shape)} -> {out_path}")
else:
    print(f"[standard_gaussian] using existing {out_path}")
PY
declare -A CONFIGS=(
  [DIVI]="test_standard_gaussian/configs/divi_standard_gaussian.yaml"
  [UIVI]="test_standard_gaussian/configs/uivi_standard_gaussian.yaml"
  [KDVI]="test_standard_gaussian/configs/kdvi_standard_gaussian.yaml"
)

for method in DIVI UIVI KDVI; do
  config="${CONFIGS[$method]}"
  log_path="test_standard_gaussian/logs/${method}.log"
  echo "[$(date -Is)] Running ${method}: ${config}" | tee "${log_path}"
  python src.py --config "${config}" 2>&1 | tee -a "${log_path}"
done

python test_standard_gaussian/plot_curves.py --results-root results/test_standard_gaussian --output-dir test_standard_gaussian
