"""
Script to compute Sliced Wasserstein (W2) distance between generated samples and HMC ground truth.
"""

import torch
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add the root directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR / "scripts"))

from utils.metrics import compute_sliced_wasserstein

# Define paths
DATA_DIR = ROOT_DIR / "data"
BASELINES_DIR = ROOT_DIR.parent / "baselines" / "hmc"
PLOTS_DIR = ROOT_DIR / "plots" / "W2_distance"

METHODS = ["AISIVI", "RSIVI", "SIVI", "UIVI"]
TARGETS = ["banana", "multimodal", "x_shaped"]


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_hmc_samples(target):
    """Load HMC ground truth samples."""
    file_path = BASELINES_DIR / f"{target}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    return data["samples"]


def main():
    print("=" * 50)
    print("Computing W2 (Sliced Wasserstein) Distance for all samples")
    print("=" * 50)

    ensure_dir(PLOTS_DIR)

    # Load HMC ground truth for each target
    hmc_samples = {}
    for target in TARGETS:
        hmc_samples[target] = load_hmc_samples(target)
        print(
            f"Loaded HMC samples for {target}: shape {hmc_samples[target].shape}"
        )

    # Results storage
    results = []

    # Process each method and target
    for method in METHODS:
        print(f"\nProcessing {method}...")
        for target in TARGETS:
            method_target_dir = DATA_DIR / method / target
            if not method_target_dir.exists():
                print(f"  Warning: {method_target_dir} not found")
                continue

            # Check for run directories
            run_dirs = sorted([
                d for d in method_target_dir.iterdir()
                if d.is_dir() and "run_" in d.name
            ],
                              key=lambda x: x.name)

            if not run_dirs:
                print(f"  No run directories found in {method_target_dir}")
                continue

            for run_dir in run_dirs:
                try:
                    run_id = int(run_dir.name.split("_")[1])
                except (IndexError, ValueError):
                    run_id = run_dir.name

                epoch_dirs = sorted(
                    [d for d in run_dir.iterdir() if d.is_dir()],
                    key=lambda x: int(x.name))

                print(f"  {target} ({run_dir.name}): {len(epoch_dirs)} epochs")

                for epoch_dir in epoch_dirs:
                    epoch_file = epoch_dir / f"epoch_{epoch_dir.name}.pt"
                    if not epoch_file.exists():
                        continue

                    data = torch.load(epoch_file,
                                      map_location="cpu",
                                      weights_only=False)
                    samples = data["samples"]
                    time_val = data["time"]
                    epoch = data["epoch"]

                    # Subsample to 10,000 samples if needed
                    if samples.shape[0] > 10000:
                        indices = torch.randperm(samples.shape[0])[:10000]
                        samples = samples[indices]

                    hmc_sub = hmc_samples[target]
                    if hmc_sub.shape[0] > 10000:
                        indices = torch.randperm(hmc_sub.shape[0])[:10000]
                        hmc_sub = hmc_sub[indices]

                    # Compute W2 distance
                    w2 = compute_sliced_wasserstein(samples,
                                                    hmc_sub,
                                                    num_projections=1000,
                                                    device=torch.device("cpu"),
                                                    p=2)

                    results.append({
                        "method": method,
                        "target": target,
                        "run_id": run_id,
                        "epoch": epoch,
                        "time": time_val,
                        "W2_distance": w2
                    })

    # Create DataFrame and save
    df = pd.DataFrame(results)
    csv_path = PLOTS_DIR / "W2_distance.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nW2 distance results saved to: {csv_path}")

    # Print summary statistics
    print("\n" + "=" * 50)
    print("Summary Statistics (Final Epoch W2 Distance)")
    print("=" * 50)

    for target in TARGETS:
        print(f"\n{target}:")
        target_df = df[df["target"] == target]
        for method in METHODS:
            method_df = target_df[target_df["method"] == method]
            if not method_df.empty:
                final_w2 = method_df[method_df["epoch"] == method_df["epoch"].
                                     max()]["W2_distance"].values[0]
                print(f"  {method}: {final_w2:.4f}")


if __name__ == "__main__":
    main()
