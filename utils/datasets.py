"""Data loading utilities for data-dependent target distributions.

Loads pre-processed ``.pt`` files created by ``prepare_data.py``.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
import scipy.io
import numpy as np
from sklearn.model_selection import train_test_split

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def load_waveform(
    data_dir: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-processed Waveform dataset.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``train.pt`` and ``test.pt``.
        Defaults to ``<project_root>/data/waveform/``.
    device : torch.device
        Target device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        ``(X_train, y_train, X_test, y_test)`` on *device*.
        ``X_train`` has shape ``(N_train, 22)`` (21 features + bias).
        ``y_train`` has shape ``(N_train,)`` (binary labels).
    """
    d = Path(data_dir) if data_dir else _DATA_DIR / "waveform"
    train = torch.load(os.path.join(d, "train.pt"), map_location=device, weights_only=True)
    test = torch.load(os.path.join(d, "test.pt"), map_location=device, weights_only=True)
    return train["X"], train["y"], test["X"], test["y"]


def load_waveform_mat(
    mat_path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the original waveform `.mat` split used by the official KSIVI repo."""
    data = scipy.io.loadmat(mat_path)
    return (
        torch.from_numpy(data["X_train"]).to(device).float(),
        torch.from_numpy(data["y_train"]).to(device).reshape(-1).float(),
        torch.from_numpy(data["X_test"]).to(device).float(),
        torch.from_numpy(data["y_test"]).to(device).reshape(-1).float(),
    )


def load_boston(
    data_dir: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load pre-processed Boston Housing dataset.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing ``train.pt`` and ``test.pt``.
        Defaults to ``<project_root>/data/boston/``.
    device : torch.device
        Target device for returned tensors.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test, mean_y_train, std_y_train)``
        on *device*.  ``y_train`` is standardised; ``y_test`` is in original
        scale.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR / "boston"
    train = torch.load(os.path.join(d, "train.pt"), map_location=device, weights_only=True)
    test = torch.load(os.path.join(d, "test.pt"), map_location=device, weights_only=True)
    return (
        train["X"],
        train["y"],
        test["X"],
        test["y"],
        train["mean_y"],
        train["std_y"],
    )


def load_boston_official_split(
    txt_path: str | Path,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load the raw official Boston train/test split before dev splitting."""
    data = np.loadtxt(txt_path)
    X_input = torch.from_numpy(data[:, :-1]).to(device).float()
    y_input = torch.from_numpy(data[:, -1]).to(device).float()

    X_train, X_test, y_train, y_test = train_test_split(
        X_input,
        y_input,
        test_size=0.1,
        random_state=42,
    )
    return X_train, y_train, X_test, y_test


def load_bnn_regression(
    name: str,
    data_dir: str | Path | None = None,
    device: torch.device = torch.device("cpu"),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Load a pre-processed UCI BNN regression dataset.

    Parameters
    ----------
    name : str
        Dataset name, e.g. ``"concrete"``, ``"power"``, ``"protein"``,
        ``"winered"``, ``"yacht"``.  Expects ``data/<name>/train.pt`` and
        ``data/<name>/test.pt`` prepared by ``prepare_data.py``.
    data_dir : str or Path, optional
        Override the directory.  Defaults to ``<project_root>/data/<name>/``.
    device : torch.device
        Target device for returned tensors.

    Returns
    -------
    tuple
        ``(X_train, y_train, X_test, y_test, mean_y_train, std_y_train)``
        on *device*.  ``y_train`` is standardised; ``y_test`` is in original
        scale.
    """
    d = Path(data_dir) if data_dir else _DATA_DIR / name
    train = torch.load(os.path.join(d, "train.pt"), map_location=device, weights_only=True)
    test = torch.load(os.path.join(d, "test.pt"), map_location=device, weights_only=True)
    return (
        train["X"],
        train["y"],
        test["X"],
        test["y"],
        train["mean_y"],
        train["std_y"],
    )
