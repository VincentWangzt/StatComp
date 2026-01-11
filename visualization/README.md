# SIVI Visualization Benchmark

This repository offers a comprehensive pipeline for processing, visualizing, and benchmarking various Semi-Implicit Variational Inference (SIVI) methods against HMC ground truth on 2D toy distributions.

## Functionality

The codebase performs the following key functions:
1.  **Data Standardization**: Unifies diverse output formats from different SIVI implementations into a standard structure.
2.  **Metric Analysis**: Computes Kullback-Leibler (KL) divergence and Wasserstein-2 (W2) distance between generated samples and Hamiltonian Monte Carlo (HMC) baselines.
3.  **Visualization**: Generates high-quality plots to inspect model performance:
    - **KL Divergence Curves**: Tracking performance over training epochs and wall-clock time.
    - **Scatter Plots**: Visualizing sample distributions against target contours.
    - **Density Estimation**: Comparing estimated densities with ground truth.

## Input Data Format

The pipeline expects raw experiment results in a `results/` directory. While the raw structure can vary (handled by `scripts/organize_data.py`), the processed data is standardized into `data/` with the following structure:

```
data/{method}/{target}/run_{run_id}/{epoch}/epoch_{epoch}.pt
```

Each `.pt` file must contain a dictionary with:
- `samples`: `torch.Tensor` of shape `(N, 2)` (generated samples).
- `time`: `float` (cumulative training time).
- `method`: `str` (method identifier).
- `target`: `str` (target distribution name).
- `epoch`: `int` (training epoch).
- `run_id`: `int` (unique run identifier).


### Target Distributions
- `banana`
- `multimodal`
- `x_shaped`

## Usage

### 1. Environment Setup
Activate the conda environment:
```bash
conda activate ai_basis
```

### 2. Data Processing Pipeline
Run the scripts in the following order to reproduce the analysis:

```bash
# 1. Organize raw results into standard format
python scripts/organize_data.py

# 2. Compute KL divergence metrics
python scripts/compute_kl_divergence.py

# 3. Generate Visualizations
python scripts/plot_kl_divergence.py  # KL curves (linear & log scale)
python scripts/plot_scatter.py        # Scatter grids with contours (transposed)
python scripts/plot_density.py        # Density plots (individual & overlay grids)
```

## Output

- **`data/`**: Standardized PyTorch files.
- **`plots/`**: Generated figures.
    - `kl_divergence/`: KL curves and `kl_divergence.csv`.
    - `scatter_plots/`: Scatter grids (`all_scatter_plots_grid.png`) and individual plots.
    - `density_plots/`: Density grids (`all_density_overlay_grid.png`, etc.) and comparison plots.

