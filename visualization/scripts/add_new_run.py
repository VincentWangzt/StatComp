"""
Script to add new run data from results/ to the existing data/ folder.
"""

import torch
import os
from pathlib import Path

# Define paths
ROOT_DIR = Path(__file__).parent.parent
RESULTS_DIR = ROOT_DIR / "results"
DATA_DIR = ROOT_DIR / "data"

TARGETS = ["banana", "multimodal", "x_shaped"]


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_next_run_id(method, target):
    """Get the next available run_id for a method/target combination."""
    method_target_dir = DATA_DIR / method / target
    if not method_target_dir.exists():
        return 1

    existing_runs = [
        d for d in method_target_dir.iterdir()
        if d.is_dir() and "run_" in d.name
    ]
    if not existing_runs:
        return 1

    max_run = max(int(d.name.split("_")[1]) for d in existing_runs)
    return max_run + 1


def save_organized_sample(method, target, run_id, epoch, samples, time_val):
    """Save organized sample to standardized location with run_id."""
    save_dir = DATA_DIR / method / target / f"run_{run_id}" / str(epoch)
    ensure_dir(save_dir)
    save_path = save_dir / f"epoch_{epoch}.pt"

    data = {
        "samples":
        samples.cpu()
        if isinstance(samples, torch.Tensor) else torch.tensor(samples),
        "time":
        float(time_val),
        "method":
        method,
        "target":
        target,
        "epoch":
        int(epoch),
        "run_id":
        run_id
    }
    torch.save(data, save_path)
    return save_path


def process_aisivi():
    """Process new AISIVI data from results/aisivi_bsivi/."""
    print("Processing AISIVI...")

    for target in TARGETS:
        file_path = RESULTS_DIR / "aisivi_bsivi" / f"trajectory_aisivi_{target}.pt"
        if not file_path.exists():
            print(f"  Warning: {file_path} not found")
            continue

        run_id = get_next_run_id("AISIVI", target)
        print(f"  {target}: Adding as run_{run_id}")

        data = torch.load(file_path, map_location="cpu", weights_only=False)
        epochs = data["epochs"]
        times = data["time"]
        samples = data["samples"]  # Shape: (N, 10000, 2)

        for i in range(len(epochs)):
            epoch = int(epochs[i].item())
            time_val = float(times[i].item())
            sample = samples[i]  # Shape: (10000, 2)
            save_organized_sample("AISIVI", target, run_id, epoch, sample,
                                  time_val)

        print(f"  {target}: processed {len(epochs)} epochs")


def process_reverse_uivi():
    """Process new Reverse UIVI data (saving as RSIVI)."""
    target_dirs = {
        "banana": "20251215_020122",
        "multimodal": "20251216_015856",
        "x_shaped": "20251216_214413"
    }

    print("Processing Reverse UIVI (saving as RSIVI)...")

    for target, run_dir in target_dirs.items():
        samples_dir = RESULTS_DIR / "reverse_uivi" / target / run_dir / "samples"
        if not samples_dir.exists():
            print(f"  Warning: {samples_dir} not found")
            continue

        run_id = get_next_run_id("RSIVI", target)
        print(f"  {target}: Adding as run_{run_id}")

        sample_files = list(samples_dir.glob("samples_epoch_*.pt"))
        count = 0

        for sample_file in sample_files:
            data = torch.load(sample_file,
                              map_location="cpu",
                              weights_only=False)
            epoch = data["epoch"]
            time_val = data["train_time"]
            samples = data["z"]  # Shape: (10000, 2)
            save_organized_sample("RSIVI", target, run_id, epoch, samples,
                                  time_val)
            count += 1

        print(f"  {target}: processed {count} epochs")


def main():
    print("=" * 50)
    print("Adding new run data from results/")
    print("=" * 50)

    # Process each method
    process_aisivi()
    process_reverse_uivi()

    print("=" * 50)
    print("New run data added successfully!")
    print("=" * 50)


if __name__ == "__main__":
    main()
