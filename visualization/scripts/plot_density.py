"""
Script to create density plots comparing generated samples with HMC ground truth.
"""
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.lines import Line2D

# Set style
plt.style.use('seaborn-v0_8-whitegrid')

# Define paths
ROOT_DIR = Path(__file__).parent.parent
DATA_DIR = ROOT_DIR / "data"
BASELINES_DIR = ROOT_DIR.parent / "baselines" / "hmc"
PLOTS_DIR = ROOT_DIR / "plots" / "density_plots"

METHODS = ["AISIVI", "RSIVI", "SIVI", "UIVI"]
METHOD_NAMES = {
    "AISIVI": "AISIVI",
    "RSIVI": "RSIVI",
    "SIVI": "SIVI",
    "UIVI": "UIVI",
    "DSIVI": "DSIVI",
}
TARGETS = ["banana", "multimodal", "x_shaped"]
TARGET_NAMES = {
    "banana": "Banana",
    "multimodal": "Multimodal",
    "x_shaped": "X-shaped"
}

DEFAULT_BBOX = {
    "multimodal": [-5, 5, -5, 5],
    "banana": [-3.5, 3.5, -6, 1],
    "x_shaped": [-5, 5, -5, 5],
}


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def load_hmc_samples(target):
    """Load HMC ground truth samples."""
    file_path = BASELINES_DIR / f"{target}.pt"
    data = torch.load(file_path, map_location="cpu", weights_only=False)
    return data["samples"].numpy()


def get_best_run_id(method, target):
    """Get the run_id with the minimum KL divergence."""
    csv_path = PLOTS_DIR.parent / "kl_divergence" / "kl_divergence.csv"
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


def plot_density_comparison(samples, hmc_samples, target, method, save_path):
    """Create side-by-side density plots comparing generated samples with HMC."""

    # Subsample if needed
    if samples.shape[0] > 10000:
        indices = np.random.choice(samples.shape[0], 10000, replace=False)
        samples = samples[indices]

    if hmc_samples.shape[0] > 10000:
        indices = np.random.choice(hmc_samples.shape[0], 10000, replace=False)
        hmc_samples = hmc_samples[indices]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    bbox = DEFAULT_BBOX[target]

    # HMC density plot
    ax1 = axes[0]
    sns.kdeplot(x=hmc_samples[:, 0],
                y=hmc_samples[:, 1],
                ax=ax1,
                fill=True,
                cmap="Blues",
                levels=15,
                thresh=0.05)
    ax1.scatter(hmc_samples[:, 0], hmc_samples[:, 1], alpha=0.1, s=1, c='blue')
    ax1.set_xlim(bbox[0], bbox[1])
    ax1.set_ylim(bbox[2], bbox[3])
    ax1.set_xlabel("$x_1$", fontsize=12)
    ax1.set_ylabel("$x_2$", fontsize=12)
    ax1.set_title("HMC Ground Truth", fontsize=14)
    ax1.set_aspect('equal')

    # Generated samples density plot
    ax2 = axes[1]
    sns.kdeplot(x=samples[:, 0],
                y=samples[:, 1],
                ax=ax2,
                fill=True,
                cmap="Oranges",
                levels=15,
                thresh=0.05)
    ax2.scatter(samples[:, 0], samples[:, 1], alpha=0.1, s=1, c='orange')
    ax2.set_xlim(bbox[0], bbox[1])
    ax2.set_ylim(bbox[2], bbox[3])
    ax2.set_xlabel("$x_1$", fontsize=12)
    ax2.set_ylabel("$x_2$", fontsize=12)
    ax2.set_title(f"{METHOD_NAMES[method]}", fontsize=14)
    ax2.set_aspect('equal')

    fig.suptitle(
        f"{TARGET_NAMES[target]} Distribution: HMC vs {METHOD_NAMES[method]}",
        fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overlay_density(samples, hmc_samples, target, method, save_path):
    """Create overlay density plot comparing generated samples with HMC."""

    # Subsample if needed
    if samples.shape[0] > 10000:
        indices = np.random.choice(samples.shape[0], 10000, replace=False)
        samples = samples[indices]

    if hmc_samples.shape[0] > 10000:
        indices = np.random.choice(hmc_samples.shape[0], 10000, replace=False)
        hmc_samples = hmc_samples[indices]

    fig, ax = plt.subplots(figsize=(8, 8))

    bbox = DEFAULT_BBOX[target]

    # HMC density - blue
    sns.kdeplot(
        x=hmc_samples[:, 0],
        y=hmc_samples[:, 1],
        ax=ax,
        fill=False,
        color='blue',
        levels=10,
        linewidths=2,
        linestyles='-',
        thresh=0.01,
    )

    # Generated samples density - orange
    sns.kdeplot(
        x=samples[:, 0],
        y=samples[:, 1],
        ax=ax,
        fill=False,
        color='orange',
        levels=10,
        linewidths=2,
        linestyles='--',
    )

    ax.set_xlim(bbox[0], bbox[1])
    ax.set_ylim(bbox[2], bbox[3])
    ax.set_xlabel("$x_1$", fontsize=12)
    ax.set_ylabel("$x_2$", fontsize=12)
    ax.set_title(
        f"{TARGET_NAMES[target]}: HMC vs {METHOD_NAMES[method]} (Contour Overlay)",
        fontsize=14)
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='HMC'),
        Line2D([0], [0],
               color='orange',
               lw=2,
               linestyle='--',
               label=METHOD_NAMES[method])
    ]

    ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

    ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_overlay_grid(hmc_samples):
    """Create a 3×7 grid of density overlay plots (targets × methods)."""
    n_methods = len(METHODS)
    n_targets = len(TARGETS)

    fig, axes = plt.subplots(n_targets,
                             n_methods,
                             figsize=(3.5 * n_methods, 3.5 * n_targets))

    for i, target in enumerate(TARGETS):
        bbox = DEFAULT_BBOX[target]
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
                continue

            samples = data["samples"].numpy()

            # Subsample if needed
            if samples.shape[0] > 10000:
                indices = np.random.choice(samples.shape[0],
                                           10000,
                                           replace=False)
                samples = samples[indices]

            hmc_sub = hmc_samples[target]
            if hmc_sub.shape[0] > 10000:
                indices = np.random.choice(hmc_sub.shape[0],
                                           10000,
                                           replace=False)
                hmc_sub = hmc_sub[indices]

            # HMC density - blue
            sns.kdeplot(
                x=hmc_sub[:, 0],
                y=hmc_sub[:, 1],
                ax=ax,
                fill=False,
                color='blue',
                levels=8,
                linewidths=1.5,
                linestyles='-',
                thresh=0.01,
            )

            # Generated samples density - orange
            sns.kdeplot(
                x=samples[:, 0],
                y=samples[:, 1],
                ax=ax,
                fill=False,
                color='orange',
                levels=8,
                linewidths=1.5,
                linestyles='--',
                thresh=0.01,
            )

            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
            ax.set_aspect('equal')
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Set title for top row only
            if i == 0:
                ax.set_title(METHOD_NAMES[method],
                             fontsize=12,
                             fontweight='bold')

            # Set y-label for left column only
            if j == 0:
                ax.set_ylabel(TARGET_NAMES[target],
                              fontsize=11,
                              fontweight='bold')

            ax.tick_params(labelsize=7)

    # Add legend
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, linestyle='-', label='HMC'),
        Line2D([0], [0], color='orange', lw=2, linestyle='--', label='Method')
    ]
    fig.legend(handles=legend_elements,
               loc='upper center',
               ncol=2,
               fontsize=11,
               bbox_to_anchor=(0.5, 1.02))

    plt.tight_layout(rect=(0, 0, 1, 0.98))
    save_path = PLOTS_DIR / "all_density_overlay_grid.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved overlay grid to: {save_path}")


def plot_individual_density_grid(hmc_samples):
    """Create a 3×8 grid of individual density plots (targets × (HMC + methods))."""
    n_methods = len(METHODS)
    n_targets = len(TARGETS)

    # 8 columns: HMC + 7 methods
    fig, axes = plt.subplots(n_targets,
                             n_methods + 1,
                             figsize=(3.5 * (n_methods + 1), 3.5 * n_targets))

    all_columns = ["hmc"] + METHODS
    column_names = {"hmc": "HMC"}
    column_names.update(METHOD_NAMES)

    for i, target in enumerate(TARGETS):
        bbox = DEFAULT_BBOX[target]
        for j, col in enumerate(all_columns):
            ax = axes[i, j]

            if col == "hmc":
                samples = hmc_samples[target].copy()
                cmap = "Blues"
                scatter_color = 'blue'
            else:
                data = get_final_epoch_data(col, target)
                if data is None:
                    ax.text(0.5,
                            0.5,
                            "No data",
                            ha='center',
                            va='center',
                            transform=ax.transAxes)
                    continue
                samples = data["samples"].numpy()
                cmap = "Oranges"
                scatter_color = 'orange'

            # Subsample if needed
            if samples.shape[0] > 10000:
                indices = np.random.choice(samples.shape[0],
                                           10000,
                                           replace=False)
                samples = samples[indices]

            # Plot density
            sns.kdeplot(
                x=samples[:, 0],
                y=samples[:, 1],
                ax=ax,
                fill=True,
                cmap=cmap,
                levels=12,
                thresh=0.05,
            )
            ax.scatter(samples[:, 0],
                       samples[:, 1],
                       alpha=0.1,
                       s=0.5,
                       c=scatter_color)

            ax.set_xlim(bbox[0], bbox[1])
            ax.set_ylim(bbox[2], bbox[3])
            ax.set_aspect('equal')
            ax.set_xlabel("")
            ax.set_ylabel("")

            # Set title for top row only
            if i == 0:
                ax.set_title(column_names[col], fontsize=12, fontweight='bold')

            # Set y-label for left column only
            if j == 0:
                ax.set_ylabel(TARGET_NAMES[target],
                              fontsize=11,
                              fontweight='bold')

            ax.tick_params(labelsize=7)

    plt.tight_layout()
    save_path = PLOTS_DIR / "all_density_individual_grid.png"
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved individual density grid to: {save_path}")


def main():
    print("=" * 50)
    print("Creating Density Plots")
    print("=" * 50)

    # Load HMC samples for each target
    hmc_samples = {}
    for target in TARGETS:
        hmc_samples[target] = load_hmc_samples(target)
        print(
            f"Loaded HMC samples for {target}: shape {hmc_samples[target].shape}"
        )

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

            # Side-by-side density plot
            save_path = target_dir / f"{method}_density_plot.png"
            plot_density_comparison(samples, hmc_samples[target], target,
                                    method, save_path)
            print(
                f"  {method}: Saved density plot (epoch {epoch}) to {save_path}"
            )

            # Overlay density plot
            save_path_overlay = target_dir / f"{method}_density_overlay.png"
            plot_overlay_density(samples, hmc_samples[target], target, method,
                                 save_path_overlay)

    # Create grid plots
    print("\nCreating grid plots...")
    plot_overlay_grid(hmc_samples)
    plot_individual_density_grid(hmc_samples)

    print("\n" + "=" * 50)
    print("Density plots complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
