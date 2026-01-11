"""
Script to plot KL divergence over epochs and training time.
"""

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# Define paths
ROOT_DIR = Path(__file__).parent.parent
PLOTS_DIR = ROOT_DIR / "plots" / "kl_divergence"

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

# Color palette for methods
COLORS = {
    "AISIVI": "#e74c3c",
    "RSIVI": "#e91e63",
    "SIVI": "#f39c12",
    "UIVI": "#1abc9c"
}


def ensure_dir(path):
    """Ensure directory exists."""
    Path(path).mkdir(parents=True, exist_ok=True)


def apply_smoothing(series, window=None):
    """Apply exponential moving average smoothing.
    
    Args:
        series: pandas Series to smooth
        window: smoothing window size. If None, uses 1/10 of series length.
    """
    if window is None:
        window = max(1, len(series) // 10)
    return series.ewm(span=window, adjust=False).mean()


def remove_outliers(df, column, threshold=3):
    """Remove outliers using z-score method."""
    z_scores = np.abs((df[column] - df[column].mean()) / df[column].std())
    return df[z_scores < threshold]


def plot_individual_method(df, method, target, save_dir):
    """Plot KL divergence for a single method and target."""
    method_df = df[(df["method"] == method) & (df["target"] == target)].copy()
    if method_df.empty:
        return

    # Calculate average time for each epoch across runs
    if "run_id" in method_df.columns:
        avg_time = method_df.groupby("epoch")["time"].mean().reset_index()
        # Merge avg_time back to method_df
        method_df = method_df.drop(columns=["time"]).merge(avg_time,
                                                           on="epoch")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot over epochs
    ax1 = axes[0]
    # Plot raw traces (if multiple runs)
    if "run_id" in method_df.columns:
        sns.lineplot(data=method_df,
                     x="epoch",
                     y="kl_divergence",
                     units="run_id",
                     ax=ax1,
                     color=COLORS[method],
                     alpha=0.2,
                     estimator=None,
                     linewidth=1)

    # Plot mean and CI
    sns.lineplot(data=method_df,
                 x="epoch",
                 y="kl_divergence",
                 ax=ax1,
                 color=COLORS[method],
                 label=METHOD_NAMES[method],
                 linewidth=2)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("KL Divergence", fontsize=12)
    ax1.set_title(
        f"{METHOD_NAMES[method]} - {TARGET_NAMES[target]} (vs Epoch)",
        fontsize=14)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Plot over time
    ax2 = axes[1]

    # Plot mean and CI using relevant time average
    sns.lineplot(data=method_df,
                 x="time",
                 y="kl_divergence",
                 ax=ax2,
                 color=COLORS[method],
                 label=METHOD_NAMES[method],
                 linewidth=2)

    ax2.set_xlabel("Training Time (s)", fontsize=12)
    ax2.set_ylabel("KL Divergence", fontsize=12)
    ax2.set_title(f"{METHOD_NAMES[method]} - {TARGET_NAMES[target]} (vs Time)",
                  fontsize=14)
    ax2.legend()
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = save_dir / f"{method}_kl_divergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_methods(df, target, save_dir):
    """Plot KL divergence for all methods on a single target."""
    target_df = df[df["target"] == target].copy()

    # Calculate average time for each method and epoch
    if "run_id" in target_df.columns:
        # We need to average time within each method-epoch group
        avg_time = target_df.groupby(["method",
                                      "epoch"])["time"].mean().reset_index()
        # Remove original time column and merge the averaged one
        target_df = target_df.drop(columns=["time"]).merge(
            avg_time, on=["method", "epoch"])

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Plot over epochs
    ax1 = axes[0]

    sns.lineplot(data=target_df,
                 x="epoch",
                 y="kl_divergence",
                 hue="method",
                 palette=COLORS,
                 linewidth=2,
                 ax=ax1,
                 marker='o',
                 markevery=1)

    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("KL Divergence", fontsize=12)
    ax1.set_title(f"All Methods - {TARGET_NAMES[target]} (vs Epoch)",
                  fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(left=0)
    ax1.set_ylim(bottom=0)

    # Plot over time
    ax2 = axes[1]

    sns.lineplot(data=target_df,
                 x="time",
                 y="kl_divergence",
                 hue="method",
                 palette=COLORS,
                 linewidth=2,
                 ax=ax2,
                 marker='o',
                 markevery=1)

    ax2.set_xlabel("Training Time (s)", fontsize=12)
    ax2.set_ylabel("KL Divergence", fontsize=12)
    ax2.set_title(f"All Methods - {TARGET_NAMES[target]} (vs Time)",
                  fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(left=0)
    ax2.set_ylim(bottom=0)

    plt.tight_layout()
    save_path = save_dir / "all_methods_kl_divergence.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_methods_log_scale(df, target, save_dir):
    """Plot KL divergence for all methods on a single target with log scale."""
    target_df = df[df["target"] == target].copy()

    # Calculate average time for each method and epoch
    if "run_id" in target_df.columns:
        # We need to average time within each method-epoch group
        avg_time = target_df.groupby(["method",
                                      "epoch"])["time"].mean().reset_index()
        # Remove original time column and merge the averaged one
        target_df = target_df.drop(columns=["time"]).merge(
            avg_time, on=["method", "epoch"])

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Plot over epochs (log scale)
    ax1 = axes[0]

    sns.lineplot(data=target_df,
                 x="epoch",
                 y="kl_divergence",
                 hue="method",
                 palette=COLORS,
                 linewidth=2,
                 ax=ax1,
                 marker='o',
                 markevery=1)

    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("KL Divergence (log scale)", fontsize=12)
    ax1.set_title(
        f"All Methods - {TARGET_NAMES[target]} (vs Epoch, Log Scale)",
        fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.set_xlim(left=0)

    # Plot over time (log scale)
    ax2 = axes[1]

    sns.lineplot(data=target_df,
                 x="time",
                 y="kl_divergence",
                 hue="method",
                 palette=COLORS,
                 linewidth=2,
                 ax=ax2,
                 marker='o',
                 markevery=1)

    ax2.set_yscale("log")
    ax2.set_xlabel("Training Time (s)", fontsize=12)
    ax2.set_ylabel("KL Divergence (log scale)", fontsize=12)
    ax2.set_title(f"All Methods - {TARGET_NAMES[target]} (vs Time, Log Scale)",
                  fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.set_xlim(left=0)

    plt.tight_layout()
    save_path = save_dir / "all_methods_kl_divergence_log.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_all_methods_loglog_scale(df, target, save_dir):
    """Plot KL divergence for all methods with log-log scale (both axes)."""
    target_df = df[df["target"] == target].copy()

    # Calculate average time for each method and epoch
    if "run_id" in target_df.columns:
        # We need to average time within each method-epoch group
        avg_time = target_df.groupby(["method",
                                      "epoch"])["time"].mean().reset_index()
        # Remove original time column and merge the averaged one
        target_df = target_df.drop(columns=["time"]).merge(
            avg_time, on=["method", "epoch"])

    fig, axes = plt.subplots(2, 1, figsize=(8, 12))

    # Plot over epochs (log-log scale)
    ax1 = axes[0]

    sns.lineplot(data=target_df,
                 x="epoch",
                 y="kl_divergence",
                 hue="method",
                 palette=COLORS,
                 linewidth=2,
                 ax=ax1,
                 marker='o',
                 markevery=1)

    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Epoch (log scale)", fontsize=12)
    ax1.set_ylabel("KL Divergence (log scale)", fontsize=12)
    ax1.set_title(f"All Methods - {TARGET_NAMES[target]} (Log-Log, vs Epoch)",
                  fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)

    # Plot over time (log-log scale)
    ax2 = axes[1]

    sns.lineplot(data=target_df,
                 x="time",
                 y="kl_divergence",
                 hue="method",
                 palette=COLORS,
                 linewidth=2,
                 ax=ax2,
                 marker='o',
                 markevery=1)

    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_xlabel("Training Time (s, log scale)", fontsize=12)
    ax2.set_ylabel("KL Divergence (log scale)", fontsize=12)
    ax2.set_title(f"All Methods - {TARGET_NAMES[target]} (Log-Log, vs Time)",
                  fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)

    plt.tight_layout()
    save_path = save_dir / "all_methods_kl_divergence_loglog.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def main():
    print("=" * 50)
    print("Plotting KL Divergence")
    print("=" * 50)

    # Load data
    csv_path = PLOTS_DIR / "kl_divergence.csv"
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} records from {csv_path}")

    # Process each target
    for target in TARGETS:
        print(f"\nProcessing {target}...")
        target_dir = PLOTS_DIR / target
        ensure_dir(target_dir)

        # Plot individual methods
        for method in METHODS:
            plot_individual_method(df, method, target, target_dir)

        # Plot all methods combined
        plot_all_methods(df, target, target_dir)

        # Plot all methods with log scale
        plot_all_methods_log_scale(df, target, target_dir)

        # Plot all methods with log-log scale
        plot_all_methods_loglog_scale(df, target, target_dir)

    print("\n" + "=" * 50)
    print("KL Divergence plotting complete!")
    print("=" * 50)


if __name__ == "__main__":
    main()
