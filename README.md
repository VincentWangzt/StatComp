## Environment Setup

### Install uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it already.

### Create environment and install dependencies

```bash
uv sync
```

This single command will:
1. Create a `.venv/` virtual environment (with the correct Python version)
2. Install all dependencies, including PyTorch

PyTorch wheels from PyPI are platform-aware — CUDA-enabled on Linux/Windows, MPS/CPU on macOS — so no manual backend selection is needed.

### Activate the environment

```bash
source .venv/bin/activate   # Linux / macOS
.venv\Scripts\activate      # Windows
```

Alternatively, you can prefix commands with `uv run` to skip activation (e.g. `uv run python src.py ...`).

### Data preparation

For data-dependent targets (BNN, logistic regression), run once before first use:

```bash
python prepare_data.py
```

## Running Experiments

To run an experiment, use the `src.py` script with a configuration file:

```bash
python src.py --config configs/sivi_banana.yaml
```

The example configurations are located in the `configs/` directory. Results and checkpoints will be saved in `results/`, and TensorBoard logs in `tb_logs/`.

Additional command-line arguments can be added via `key=value` pairs:

```bash
python src.py --config configs/sivi_banana.yaml train.epochs=20000 train.vi.lr=0.001
```

Results will be saved in the `results/` directory.
