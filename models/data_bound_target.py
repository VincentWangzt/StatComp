"""Wrapper that binds a dataset to data-dependent target distributions.

Data-dependent targets (``LRwaveform``, ``Bnn``) have ``logp``/``score``
signatures that require ``(Z, batchdataset, batchlabel, scale_sto)`` instead of
the standard ``(X,)`` expected by the runner.  :class:`DataBoundTarget` presents
the standard single-argument interface by storing the dataset internally and
forwarding calls with the bound data.

A factory function :func:`build_data_bound_target` handles data loading and
construction for each data-dependent target type.
"""

from __future__ import annotations

import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from models.target_models import Bnn, LRwaveform


class DataBoundTarget:
    """Wrap a data-dependent target to present the standard ``logp(X)``/``score(X)`` interface.

    Parameters
    ----------
    inner : LRwaveform | Bnn
        The underlying data-dependent target model.
    dataset : torch.Tensor
        Training features, shape ``(N, d)`` or ``(N, d+1)`` with bias col.
    labels : torch.Tensor
        Training labels, shape ``(N,)`` or ``(N, 1)``.
    batch_size : int or None
        If *None* or ``>= N``: use full data each call (``scale_sto=1``).
        Otherwise: random minibatch of this size, with ``scale_sto = N / batch_size``.
    z_dim : int
        Dimensionality of the parameter space.
    device : torch.device
        Computation device.
    test_data : tuple or None
        Optional ``(X_test, y_test, mean_y_train, std_y_train)`` for BNN evaluation.
    """

    def __init__(
        self,
        inner: LRwaveform | Bnn,
        dataset: torch.Tensor,
        labels: torch.Tensor,
        batch_size: int | None = None,
        z_dim: int = 2,
        device: torch.device = torch.device("cpu"),
        test_data: tuple[torch.Tensor, ...] | None = None,
    ) -> None:
        self.inner = inner
        self.dataset = dataset.to(device)
        self.labels = labels.to(device)
        self.z_dim = z_dim
        self.device = device
        self.name = inner.name
        self.test_data = test_data

        N = self.dataset.shape[0]
        if batch_size is None or batch_size >= N:
            self.batch_size: int | None = None
            self.scale_sto: float = 1.0
        else:
            self.batch_size = batch_size
            self.scale_sto = float(N) / float(batch_size)

    def _get_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return full data or a random minibatch."""
        if self.batch_size is None:
            return self.dataset, self.labels
        idx = torch.randint(0, self.dataset.shape[0], (self.batch_size,))
        return self.dataset[idx], self.labels[idx]

    def logp(self, X: torch.Tensor) -> torch.Tensor:
        """Compute log-density with bound dataset."""
        data, labels = self._get_batch()
        return self.inner.logp(X, data, labels, self.scale_sto)

    def score(self, X: torch.Tensor) -> torch.Tensor:
        """Compute score with bound dataset."""
        data, labels = self._get_batch()
        return self.inner.score(X, data, labels, self.scale_sto)

    # ------------------------------------------------------------------
    # Visualization (runner falls through contour_plot → trace_plot)
    # ------------------------------------------------------------------

    def contour_plot(self, *args: Any, **kwargs: Any) -> None:
        """Not applicable for high-dimensional data-dependent targets."""
        raise NotImplementedError

    def trace_plot(
        self,
        u: torch.Tensor,
        figpath: str | None = None,
        figname: str | None = None,
        figtitle: str = "",
    ) -> None:
        """Plot marginal histograms for the first few parameter dimensions."""
        u_np = u.detach().cpu().numpy()
        n_show = min(6, u_np.shape[1])
        rows = 2 if n_show > 3 else 1
        cols = min(3, n_show)
        fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows))
        axes_flat = np.asarray(axes).flat
        for i, ax in enumerate(axes_flat):
            if i < n_show:
                ax.hist(u_np[:, i], bins=50, density=True, alpha=0.7)
                ax.set_title(f"dim {i}")
            else:
                ax.set_visible(False)
        plt.suptitle(figtitle)
        plt.tight_layout()
        if figpath is not None and figname is not None:
            plt.savefig(os.path.join(figpath, figname), dpi=150)
        plt.close()

    # ------------------------------------------------------------------
    # BNN-specific evaluation helpers
    # ------------------------------------------------------------------

    def rmse_llk(self, Z: torch.Tensor) -> tuple[float, float]:
        """Compute test RMSE and log-likelihood (BNN only)."""
        if self.test_data is None or not isinstance(self.inner, Bnn):
            raise RuntimeError("rmse_llk requires BNN inner model with test_data")
        X_test, y_test, mean_y, std_y = self.test_data
        return self.inner.rmse_llk(Z, X_test, y_test, mean_y, std_y)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def build_data_bound_target(
    target_type: str,
    target_cfg: dict[str, Any] | None,
    device: torch.device,
) -> DataBoundTarget:
    """Build a :class:`DataBoundTarget` for a data-dependent target type.

    Parameters
    ----------
    target_type : str
        One of ``"LRwaveform"`` or ``"Bnn_boston"``.
    target_cfg : dict or None
        Target config section (may contain ``data.batch_size``, ``data.path``, etc.).
    device : torch.device
        Computation device.

    Returns
    -------
    DataBoundTarget
        Ready-to-use target with standard ``logp(X)``/``score(X)`` interface.
    """
    from utils.datasets import load_boston, load_waveform

    target_cfg = target_cfg or {}
    data_cfg = target_cfg.get("data", {}) or {}
    data_batch_size = data_cfg.get("batch_size", None)

    if target_type == "LRwaveform":
        X_train, y_train, X_test, y_test = load_waveform(device=device)
        inner = LRwaveform(device=device)
        z_dim = X_train.shape[1]  # features + bias column
        return DataBoundTarget(
            inner=inner,
            dataset=X_train,
            labels=y_train,
            batch_size=data_batch_size,
            z_dim=z_dim,
            device=device,
        )

    elif target_type == "Bnn_boston":
        X_train, y_train, X_test, y_test, mean_y, std_y = load_boston(
            device=device,
        )
        d = X_train.shape[1]
        n_hidden = int(data_cfg.get("n_hidden", 50))
        loglambda = float(data_cfg.get("loglambda", -1.003869799168037))
        loggamma = float(data_cfg.get("loggamma", -2.555990767319021))
        inner = Bnn(
            device=device,
            d=d,
            n_hidden=n_hidden,
            loglambda=loglambda,
            loggamma=loggamma,
        )
        z_dim = inner.dim_wb  # (d+1)*n_hidden + (n_hidden+1)
        return DataBoundTarget(
            inner=inner,
            dataset=X_train,
            labels=y_train,
            batch_size=data_batch_size,
            z_dim=z_dim,
            device=device,
            test_data=(X_test, y_test, mean_y, std_y),
        )

    else:
        raise ValueError(f"Unknown data-dependent target type: {target_type}")
