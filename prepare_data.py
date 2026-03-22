"""One-time data download and preprocessing script.

Prepares ``.pt`` files under ``data/`` for data-dependent targets:

- **Waveform**: UCI Waveform (21 features, 3 classes → binarised to 2).
  Generated synthetically following the original UCI specification.
- **Boston Housing**: 13 features regression (loaded from bundled text file
  or the KSIVI ``datasets/`` directory).

Usage::

    python prepare_data.py

Output files::

    data/waveform/train.pt   – dict(X=..., y=...)
    data/waveform/test.pt    – dict(X=..., y=...)
    data/boston/train.pt      – dict(X=..., y=..., mean_y=..., std_y=...)
    data/boston/test.pt       – dict(X=..., y=...)
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Waveform (UCI) — synthetic generation following the standard specification
# ---------------------------------------------------------------------------

# The UCI waveform generator produces 21 continuous attributes.
# Three classes are defined by shifted/combined triangle waves with N(0,1) noise.

_WAVE_H1 = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0], dtype=np.float64)
_WAVE_H2 = np.array([0, 0, 0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0], dtype=np.float64)
_WAVE_H3 = np.array([0, 0, 1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)


def _generate_waveform(n_samples: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Generate the UCI Waveform dataset (3 classes, 21 features).

    Each sample belongs to one of three classes defined by:
        class 0: u * h1 + (1-u) * h2 + noise
        class 1: u * h1 + (1-u) * h3 + noise
        class 2: u * h2 + (1-u) * h3 + noise
    where u ~ Uniform(0,1) and noise ~ N(0,1).
    """
    rng = np.random.RandomState(seed)
    n_per_class = n_samples // 3
    remainder = n_samples - 3 * n_per_class

    Xs, ys = [], []
    templates = [(_WAVE_H1, _WAVE_H2), (_WAVE_H1, _WAVE_H3), (_WAVE_H2, _WAVE_H3)]
    for cls_idx, (ha, hb) in enumerate(templates):
        n_cls = n_per_class + (1 if cls_idx < remainder else 0)
        u = rng.uniform(0, 1, size=(n_cls, 1))
        noise = rng.randn(n_cls, 21)
        X_cls = u * ha + (1 - u) * hb + noise
        Xs.append(X_cls)
        ys.append(np.full(n_cls, cls_idx, dtype=np.int64))

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def prepare_waveform(out_dir: str = "data/waveform", seed: int = 42) -> None:
    """Generate waveform data, binarise, add bias, split, standardise, and save."""
    os.makedirs(out_dir, exist_ok=True)

    X, y = _generate_waveform(n_samples=5000, seed=seed)

    # Binarise: class 0 vs rest
    y_bin = (y == 0).astype(np.float64)

    # Train/test split (80/20)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.8 * len(X))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y_bin[idx[:n_train]], y_bin[idx[n_train:]]

    # Standardise features using training stats
    mu = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu) / std
    X_test = (X_test - mu) / std

    # Append bias column
    X_train = np.hstack([X_train, np.ones((len(X_train), 1))])
    X_test = np.hstack([X_test, np.ones((len(X_test), 1))])

    torch.save(
        {"X": torch.tensor(X_train, dtype=torch.float32),
         "y": torch.tensor(y_train, dtype=torch.float32)},
        os.path.join(out_dir, "train.pt"),
    )
    torch.save(
        {"X": torch.tensor(X_test, dtype=torch.float32),
         "y": torch.tensor(y_test, dtype=torch.float32)},
        os.path.join(out_dir, "test.pt"),
    )
    print(f"Waveform saved to {out_dir}/  "
          f"(train: {X_train.shape}, test: {X_test.shape})")


# ---------------------------------------------------------------------------
# Boston Housing
# ---------------------------------------------------------------------------

_BOSTON_SEARCH_PATHS = [
    "data/boston/boston_housing.txt",
    "datasets/boston_housing.txt",
    "../KSIVI/datasets/boston_housing.txt",
    "/data/workspace/KSIVI/datasets/boston_housing.txt",
]


def _find_boston_file() -> str:
    for p in _BOSTON_SEARCH_PATHS:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"Cannot find boston_housing.txt.  Searched: {_BOSTON_SEARCH_PATHS}"
    )


def prepare_boston(out_dir: str = "data/boston", seed: int = 42) -> None:
    """Load Boston Housing, split, standardise, and save."""
    os.makedirs(out_dir, exist_ok=True)

    src = _find_boston_file()
    data = np.loadtxt(src)
    # Last column is the target
    X = data[:, :-1]
    y = data[:, -1]

    # Train/test split (90/10, matching KSIVI)
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(X))
    n_train = int(0.9 * len(X))
    X_train, X_test = X[idx[:n_train]], X[idx[n_train:]]
    y_train, y_test = y[idx[:n_train]], y[idx[n_train:]]

    # Standardise X using training stats
    mu_x = X_train.mean(axis=0)
    std_x = X_train.std(axis=0) + 1e-8
    X_train = (X_train - mu_x) / std_x
    X_test = (X_test - mu_x) / std_x

    # Standardise y using training stats
    mean_y = float(y_train.mean())
    std_y = float(y_train.std()) + 1e-8
    y_train_norm = (y_train - mean_y) / std_y
    # y_test stays in original scale for evaluation

    torch.save(
        {
            "X": torch.tensor(X_train, dtype=torch.float32),
            "y": torch.tensor(y_train_norm, dtype=torch.float32).unsqueeze(-1),
            "mean_y": torch.tensor(mean_y, dtype=torch.float32),
            "std_y": torch.tensor(std_y, dtype=torch.float32),
        },
        os.path.join(out_dir, "train.pt"),
    )
    torch.save(
        {
            "X": torch.tensor(X_test, dtype=torch.float32),
            "y": torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1),
        },
        os.path.join(out_dir, "test.pt"),
    )
    print(f"Boston saved to {out_dir}/  "
          f"(train: {X_train.shape}, test: {X_test.shape}, "
          f"mean_y={mean_y:.4f}, std_y={std_y:.4f})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    prepare_waveform()
    prepare_boston()
    print("Data preparation complete.")
