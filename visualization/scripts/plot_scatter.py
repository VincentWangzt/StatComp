"""
Script to create scatter plots for final epoch samples against target distribution contours.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Add the root directory to path
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.target_models import Banana_shape, X_shaped, Multimodal, DEFAULT_BBOX

# Define paths
DATA_DIR = ROOT_DIR / "data"
PLOTS_DIR = ROOT_DIR / "plots" / "scatter_plots"

METHODS = ["AISIVI", "RSIVI", "SIVI", "UIVI"]
METHOD_NAMES = {
    "AISIVI": "AISIVI",
    "RSIVI": "RSIVI",
    "SIVI": "SIVI",
    "UIVI": "UIVI"
}
TARGETS = ["banana", "multimodal", "x_shaped"]
TARGET_NAMES = {
    "banana": "Banana",
    "multimodal": "Multimodal",
    "x_shaped": "X-shaped"
}

# Target model classes
TARGET_MODELS = {
    "banana": Banana_shape,
    "multimodal": Multimodal,
    "x_shaped": X_shaped
}


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def get_best_run_id(method, target):
    """Get the run_id with the minimum KL divergence."""
    csv_path = ROOT_DIR / "plots" / "kl_divergence" / "kl_divergence.csv"
    if not csv_path.exists():
        return None

    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None

    subset = df[(df["method"] == method) & (df["target"] == target)]
    if subset.empty:
        return None

    if "run_id" not in subset.columns:
        return None

    # Get the row with minimum KL divergence
    best_row = subset.loc[subset["kl_divergence"].idxmin()]
    return best_row["run_id"]


def get_final_epoch_data(method, target):
    """Get the final epoch data for a method and target."""
    method_target_dir = DATA_DIR / method / target
    if not method_target_dir.exists():
        return None

    # Determine which run directory to use
    run_dir = None
    best_run_id = get_best_run_id(method, target)

    if best_run_id is not None:
        # Try to find the directory for best run
        candidates = [f"run_{best_run_id}", str(best_run_id)]
        for cand in candidates:
            if (method_target_dir / cand).exists():
                run_dir = method_target_dir / cand
                break

    # Fallback: if best run not found or determined, pick the first run directory
    if run_dir is None:
        run_dirs = sorted([
            d for d in method_target_dir.iterdir()
            if d.is_dir() and "run_" in d.name
        ],
                          key=lambda x: x.name)
        if run_dirs:
            run_dir = run_dirs[0]
        else:
            # Fallback to root (old structure)
            run_dir = method_target_dir

    # Find epoch directories
    epoch_dirs = sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.isdigit()],
        key=lambda x: int(x.name))

    if not epoch_dirs:
        return None

    final_epoch_dir = epoch_dirs[-1]
    final_epoch_file = final_epoch_dir / f"epoch_{final_epoch_dir.name}.pt"

    if not final_epoch_file.exists():
        return None

    return torch.load(final_epoch_file, map_location="cpu", weights_only=False)


def plot_scatter_with_contour(samples, target, method, save_path):
    """Plot samples as scatter against target distribution contour."""
    # Create target model
    device = torch.device("cpu")
    target_model = TARGET_MODELS[target](device)

    # Get bounding box
    bbox = DEFAULT_BBOX[target]

    # Subsample if needed (randomly select 10000 points)
    if isinstance(samples, np.ndarray):
        samples = torch.from_numpy(samples)

    num_samples = samples.shape[0]
    if num_samples > 10000:
        indices = torch.randperm(num_samples)[:10000]
        samples = samples[indices]

    # Use the contour_plot method from target_models
    target_model.contour_plot(bbox=bbox,
                              fnet=None,
                              samples=samples,
                              save_to_path=str(save_path),
                              quiver=False,
                              t=None)


def plot_all_scatter_grid():
    """Create a single grid figure with all scatter plots (targets × methods)."""
    n_methods = len(METHODS)
    n_targets = len(TARGETS)

    fig, axes = plt.subplots(n_targets,
                             n_methods,
                             figsize=(4 * n_methods, 4 * n_targets))

    for i, target in enumerate(TARGETS):
        for j, method in enumerate(METHODS):
            ax = axes[i, j]

            # Get data
            data = get_final_epoch_data(method, target)
            if data is None:
                ax.text(0.5,
                        0.5,
                        "No data",
                        ha='center',
                        va='center',
                        transform=ax.transAxes)
                ax.set_title(
                    f"{METHOD_NAMES[method]} - {TARGET_NAMES[target]}")
                continue

            samples = data["samples"].numpy()

            # Subsample if needed (randomly select 10000 points)
            if samples.shape[0] > 10000:
                indices = np.random.choice(samples.shape[0],
                                           10000,
                                           replace=False)
                samples = samples[indices]

            # Create target model and plot
            device = torch.device("cpu")
            target_model = TARGET_MODELS[target](device)
            bbox = DEFAULT_BBOX[target]

            # Plot contour
            xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
            positions = np.vstack([xx.ravel(), yy.ravel()])
            f = -np.log(-np.reshape(
                target_model.logp(torch.Tensor(positions.T).to(
                    device)).cpu().numpy(), xx.shape))

            ax.axis(bbox)
            ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
            ax.contourf(xx, yy, f, cmap='Blues', alpha=0.8, levels=11)
            ax.plot(samples[:, 0],
                    samples[:, 1],
                    '.',
                    markersize=1,
                    color='#ff7f0e',
                    alpha=0.5)

            # Set title for top row only
            if i == 0:
                ax.set_title(METHOD_NAMES[method],
                             fontsize=14,
                             fontweight='bold')

            # Set y-label for left column only
            if j == 0:
                ax.set_ylabel(TARGET_NAMES[target],
                              fontsize=12,
                              fontweight='bold')

            ax.tick_params(labelsize=8)

    plt.tight_layout()
    save_path = PLOTS_DIR / "all_scatter_plots_grid.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved grid plot to: {save_path}")


def main():
    print("=" * 50)
    print("Creating Scatter Plots")
    print("=" * 50)

    for target in TARGETS:
        print(f"\nProcessing {target}...")
        target_dir = PLOTS_DIR / target
        ensure_dir(target_dir)

        for method in METHODS:
            data = get_final_epoch_data(method, target)
            if data is None:
                print(f"  {method}: No data found")
                continue

            samples = data["samples"].numpy()
            epoch = data["epoch"]

            save_path = target_dir / f"{method}_scatter_plot.png"
            plot_scatter_with_contour(samples, target, method, save_path)
            print(
                f"  {method}: Saved scatter plot (epoch {epoch}) to {save_path}"
            )

    # Create combined grid plot
    print("\nCreating combined grid plot...")
    plot_all_scatter_grid()

    print("\n" + "=" * 50)
    print("Scatter plots complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
