"""
Script to clean and organize data from different methods into a unified structure.
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


def save_organized_sample(method, target, epoch, samples, time_val):
    """Save organized sample to standardized location."""
    save_dir = DATA_DIR / method / target / str(epoch)
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
        int(epoch)
    }
    torch.save(data, save_path)
    return save_path


def process_aisivi_bsivi():
    """Process AISIVI and BSIVI data."""
    methods = ["aisivi", "bsivi"]

    for method in methods:
        print(f"Processing {method}...")
        for target in TARGETS:
            file_path = RESULTS_DIR / "aisivi_bsivi" / f"trajectory_{method}_{target}.pt"
            if not file_path.exists():
                print(f"  Warning: {file_path} not found")
                continue

            data = torch.load(file_path,
                              map_location="cpu",
                              weights_only=False)
            epochs = data["epochs"]
            times = data["time"]
            samples = data["samples"]  # Shape: (200, 10000, 2)

            for i in range(len(epochs)):
                epoch = int(epochs[i].item())
                time_val = float(times[i].item())
                sample = samples[i]  # Shape: (10000, 2)
                save_organized_sample(method, target, epoch, sample, time_val)

            print(f"  {target}: processed {len(epochs)} epochs")


def process_ksivi_sivism():
    """Process KSIVI and SIVISM data."""
    method_files = {
        "ksivi": {
            "banana": "ksivi_banana.pt",
            "multimodal": "ksivi_multimodal.pt",
            "x_shaped": "ksivi_xshaped.pt"
        },
        "sivism": {
            "banana": "sivism_banana.pt",
            "multimodal": "sivism_multimodal.pt",
            "x_shaped": "sivism_xshaped.pt"
        }
    }

    for method, target_files in method_files.items():
        print(f"Processing {method}...")
        for target, filename in target_files.items():
            file_path = RESULTS_DIR / "ksivi&sivism" / filename
            if not file_path.exists():
                print(f"  Warning: {file_path} not found")
                continue

            data_list = torch.load(file_path,
                                   map_location="cpu",
                                   weights_only=False)

            for item in data_list:
                epoch = item["epoch"] * 50  # Original epochs are scaled
                time_val = item["time"]
                samples = item["samples"]  # Shape: (10000, 2)
                save_organized_sample(method, target, epoch, samples, time_val)

            print(f"  {target}: processed {len(data_list)} epochs")


def process_sivi():
    """Process SIVI data."""
    target_files = {
        "banana": "sivi_2d_banana.pt",
        "multimodal": "sivi_2d_gmm2d.pt",  # Note: different naming
        "x_shaped": "sivi_2d_Xshape.pt"  # Note: different naming
    }

    print("Processing sivi...")
    for target, filename in target_files.items():
        file_path = RESULTS_DIR / "sivi" / filename
        if not file_path.exists():
            print(f"  Warning: {file_path} not found")
            continue

        data_list = torch.load(file_path,
                               map_location="cpu",
                               weights_only=False)

        for item in data_list:
            epoch = item["epoch"]
            time_val = item["time"]
            samples = item["samples"]  # Shape: (10000, 2)
            save_organized_sample("sivi", target, epoch, samples, time_val)

        print(f"  {target}: processed {len(data_list)} epochs")


def process_uivi():
    """Process UIVI data."""
    target_dirs = {
        "banana": "20251215_014024",
        "multimodal": "20251215_022343",
        "x_shaped": "20251215_030701"
    }

    print("Processing uivi...")
    for target, run_dir in target_dirs.items():
        samples_dir = RESULTS_DIR / "uivi" / target / run_dir / "samples"
        if not samples_dir.exists():
            print(f"  Warning: {samples_dir} not found")
            continue

        sample_files = list(samples_dir.glob("samples_epoch_*.pt"))
        count = 0

        for sample_file in sample_files:
            data = torch.load(sample_file,
                              map_location="cpu",
                              weights_only=False)
            epoch = data["epoch"]
            time_val = data["train_time"]
            samples = data["z"]  # Shape: (10000, 2)
            save_organized_sample("uivi", target, epoch, samples, time_val)
            count += 1

        print(f"  {target}: processed {count} epochs")


def process_reverse_uivi():
    """Process Reverse UIVI data (renamed to reverse_sivi)."""
    target_dirs = {
        "banana": "20251215_020122",
        "multimodal": "20251216_015856",
        "x_shaped": "20251216_214413"
    }

    print("Processing reverse_uivi (saving as reverse_sivi)...")
    for target, run_dir in target_dirs.items():
        samples_dir = RESULTS_DIR / "reverse_uivi" / target / run_dir / "samples"
        if not samples_dir.exists():
            print(f"  Warning: {samples_dir} not found")
            continue

        sample_files = list(samples_dir.glob("samples_epoch_*.pt"))
        count = 0

        for sample_file in sample_files:
            data = torch.load(sample_file,
                              map_location="cpu",
                              weights_only=False)
            epoch = data["epoch"]
            time_val = data["train_time"]
            samples = data["z"]  # Shape: (10000, 2)
            save_organized_sample("reverse_sivi", target, epoch, samples,
                                  time_val)
            count += 1

        print(f"  {target}: processed {count} epochs")


def main():
    print("=" * 50)
    print("Organizing data from different methods")
    print("=" * 50)

    # Clean data directory if exists
    if DATA_DIR.exists():
        import shutil
        shutil.rmtree(DATA_DIR)

    ensure_dir(DATA_DIR)

    # Process each method
    process_aisivi_bsivi()
    process_ksivi_sivism()
    process_sivi()
    process_uivi()
    process_reverse_uivi()

    print("=" * 50)
    print("Data organization complete!")
    print(f"Organized data saved to: {DATA_DIR}")
    print("=" * 50)


if __name__ == "__main__":
    main()
