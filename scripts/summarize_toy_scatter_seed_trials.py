"""Record all qualitative seed trials without replacing the quantitative metrics."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from finalization.artifacts import completed_runs, find_final_samples, load_manifest, load_sample_z
from finalization.plots import _target_bbox


def main() -> None:
    pairs = {("AISIVI", "8_gaussians"), ("KSIVI", "student_uc")}
    campaigns = ["toy_scatter_grid", "toy_scatter_seed_aisivi", "toy_scatter_seed_ksivi",
                 "toy_scatter_aisivi_additional", "toy_scatter_ksivi_additional"]
    rows = []
    for campaign in campaigns:
        for record in completed_runs(load_manifest(f"campaigns/{campaign}/manifest.json")):
            if (record.method, record.target) not in pairs:
                continue
            sample_path, epoch = find_final_samples(record.result_path)
            samples = load_sample_z(sample_path)
            assert torch.isfinite(samples).all(), record.run_id
            bbox = _target_bbox(record.target)
            assert bbox is not None
            inside = ((samples[:, 0] >= bbox[0]) & (samples[:, 0] <= bbox[1])
                      & (samples[:, 1] >= bbox[2]) & (samples[:, 1] <= bbox[3]))
            rows.append({
                "campaign": campaign, "method": record.method, "target": record.target,
                "seed": record.seed, "final_epoch": epoch, "sample_count": len(samples),
                "fraction_in_plot_bounds": float(inside.float().mean()),
                "mean_x": float(samples[:, 0].mean()), "mean_y": float(samples[:, 1].mean()),
                "std_x": float(samples[:, 0].std()), "std_y": float(samples[:, 1].std()),
                "result_path": record.entry["result_path"],
            })
    rows.sort(key=lambda row: (row["method"], row["seed"]))
    out_dir = REPO_ROOT / "campaigns/toy_scatter_grid/generated_reports/finalization"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "seed_trials.csv"
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row["method"], row["target"], "seed", row["seed"],
              "fraction in bounds", f'{row["fraction_in_plot_bounds"]:.4f}',
              "std", f'{row["std_x"]:.3f}', f'{row["std_y"]:.3f}')
    print(f"Wrote {out_path.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
