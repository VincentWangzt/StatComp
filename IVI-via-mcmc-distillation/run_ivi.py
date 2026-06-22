"""
Runnable conversion of `implicit-variational-inference-via-mcmc-self-distillation.ipynb`.

Keeps only the FIRST `model.learn` call from the notebook:
    model.learn(0.001, 0.01, 100000, batch_size=128,
                warm_up_interval=50000, anneal_freq=5000,
                anneal_rate=0.75, method='mala')

After training, computes the KL ITE evaluation metric:
    KL( q_phi(x) || p_target(x) )
estimated with the kNN-based estimator `ite.cost.BDKL_KnnK`,
using model samples vs ground-truth GMM samples.
"""

import os
import sys
import math
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal

import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt

# Make `ite` importable when running from this folder
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from models import GMM  # local file in this folder
import ite  # noqa: E402  (project-root ite package)

# Load the project's `models.target_models` by file path — the local
# `models.py` in this folder shadows the project-level `models/` package
# when both are on sys.path, so a regular import would resolve to the
# wrong module. The project `models/` is a namespace package (no
# __init__.py), so we just need to load target_models.py directly.
import importlib.util as _importlib_util
_pkg_target_models = os.path.join(_ROOT, "models", "target_models.py")
if os.path.exists(_pkg_target_models):
    _spec_tm = _importlib_util.spec_from_file_location(
        "_project_target_models", _pkg_target_models,
    )
    _project_target_models = _importlib_util.module_from_spec(_spec_tm)
    sys.modules["_project_target_models"] = _project_target_models
    try:
        _spec_tm.loader.exec_module(_project_target_models)
        _target_registry = _project_target_models.target_distribution
    except Exception as _e:
        print(f"[run_ivi] failed to import project target_models: {_e}",
              file=sys.stderr)
        _target_registry = {}
else:
    _target_registry = {}


# ---------------------------------------------------------------------------
# Target adapters
# ---------------------------------------------------------------------------
class _ProjectGMMAdapter:
    """Wrap a project-target instance (e.g. EightGaussians, EightGaussiansSmall)
    to match the API expected by ImVIDrift: `.name`, `.n_dim`, `.logp`,
    `.score`, `.sample`."""

    def __init__(self, inner, display_name: str):
        self._inner = inner
        # Keep both aliases so external code can reach the underlying project
        # target either way.
        self.inner = inner
        self.name = display_name
        self.n_dim = int(inner.z_dim)
        self.z_dim = int(inner.z_dim)

    def logp(self, x):
        return self._inner.logp(x)

    def score(self, x):
        return self._inner.score(x)

    def sample(self, n):
        return self._inner.sample(n)

    def contour_plot(self, *args, **kwargs):
        """Delegate to the project target's contour_plot.

        This is the exact KDVI plotting path
        (``models.target_models.Toy_2D.contour_plot`` as inherited by
        ``EightGaussians`` / ``EightGaussiansSmall``).
        """
        return self._inner.contour_plot(*args, **kwargs)


def build_target(name: str):
    """Build a target distribution by name.

    Names:
      - 'gmm'                : notebook geometry, GMM(8 clusters, sigma=0.1, r=1)
      - '8_gaussians'        : project EightGaussians  (radius=4, sigma=0.5)
      - '8_gaussians_small'  : project EightGaussiansSmall (radius=1, sigma=0.1)
                               — equivalent geometry to 'gmm' but using the
                               project's class & registry (used for sanity
                               checks).
    """
    if name == "gmm":
        return GMM(n_clusters=8, sigma=0.1, r=1)

    if name in _target_registry:
        device = torch.device("cpu")
        inner = _target_registry[name](device=device)
        return _ProjectGMMAdapter(inner, display_name=name)

    raise ValueError(f"Unknown target name: {name!r}")


# ---------------------------------------------------------------------------
# Network and target distributions (verbatim from the notebook)
# ---------------------------------------------------------------------------
class Transform(nn.Module):
    def __init__(self, latent_dim, out_dim, hidden_units=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ELU(),
            nn.Linear(hidden_units, out_dim * 2),
        )

    def forward(self, x):
        return torch.chunk(self.model(x), 2, dim=-1)


class Banana(nn.Module):
    name = "banana"

    def __init__(self, cov):
        super().__init__()
        self.dist = MultivariateNormal(torch.zeros(2), cov)
        self.cov_inv = torch.inverse(cov)
        self.n_dim = cov.size()[0]

    def logp(self, x):
        v = torch.stack([x[:, 0], x[:, 1] + 1 + x[:, 0] ** 2], dim=1)
        return self.dist.log_prob(v)

    def score(self, x):
        v = torch.stack([x[:, 0], x[:, 1] + 1 + x[:, 0] ** 2], dim=1)
        grad = -v @ self.cov_inv.t()
        return torch.stack(
            [grad[:, 0] + 2 * x[:, 0] * grad[:, 1], grad[:, 1]], dim=1
        )

    def sample(self, sample_size):
        v = self.dist.sample((sample_size,))
        x1 = v[:, 0]
        x2 = v[:, 1] - x1 ** 2 - 1
        return torch.stack([x1, x2], dim=1)


# ---------------------------------------------------------------------------
# Distance / kernel utilities
# ---------------------------------------------------------------------------
def energy_distance_batch(samples, y):
    term1 = torch.cdist(samples, y, p=2).mean()
    term2 = torch.cdist(y, y, p=2).mean()
    term3 = torch.cdist(samples, samples, p=2).mean()
    return term1 - 0.5 * term2 - 0.5 * term3


def rbf_kernel(X, Y, h=-1):
    pairwise_dists = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)
    if h <= 0:
        h = torch.median(pairwise_dists)
        h = h / torch.log(torch.tensor(X.shape[0], dtype=X.dtype) + 1.0)
    return torch.exp(-pairwise_dists / (2 * h))


def exp_kernel(X, Y, h=-1):
    pairwise_dists = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)
    if h <= 0:
        h = torch.median(pairwise_dists)
        h = h / torch.log(torch.tensor(X.shape[0], dtype=X.dtype) + 1.0)
    return torch.exp(-(pairwise_dists ** 0.5) / (2 * h))


def imq_kernel(X, Y, c=1.0, beta=-0.5):
    pairwise_dists = (X.unsqueeze(1) - Y.unsqueeze(0)).pow(2).sum(-1)
    return (c ** 2 + pairwise_dists) ** beta


def riesz_kernel(X, Y, alpha=1.0, epsilon=1e-3):
    diff = X.unsqueeze(1) - Y.unsqueeze(0)
    dist2 = (diff ** 2).sum(-1) + epsilon ** 2
    return dist2 ** (-alpha / 2)


def maximum_mean_discrepancy(samples, y, test=False):
    term1 = torch.cdist(samples, y, p=2)
    term2 = torch.cdist(y, y, p=2)

    h = torch.cat([term1, term2], dim=0).median().detach()

    term1 = (-term1 / (2 * h)).exp().mean()
    term2 = (-term2 / (2 * h)).exp().mean()

    if test:
        term3 = torch.cdist(samples, samples, p=2)
        term3 = (-term3 / (2 * h)).exp().mean()
        return 0.5 * term2 - term1 + 0.5 * term3
    return 0.5 * term2 - term1


# ---------------------------------------------------------------------------
# Implicit VI drift model (verbatim from the notebook)
# ---------------------------------------------------------------------------
class ImVIDrift(nn.Module):
    def __init__(self, target_model, latent_dim=32, hidden_units=256):
        super().__init__()
        self.target = target_model
        self.transform = Transform(latent_dim, target_model.n_dim, hidden_units=hidden_units)
        self.latent_dim = latent_dim
        # Default contour bbox; main() may overwrite this depending on --target.
        self._bbox = [-1.5, 1.5, -1.5, 1.5]
        torch.set_num_threads(1)

    def contour_plot(self, bbox, ax, ngrid=100, samples=None, save_to_path=None):
        xx, yy = np.mgrid[bbox[0]:bbox[1]:100j, bbox[2]:bbox[3]:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        f = np.reshape(self.target.logp(torch.Tensor(positions.T)).numpy(), xx.shape)
        if samples is None:
            samples = self.target.sample(2000).numpy()

        cxx, cyy = np.mgrid[bbox[0]:bbox[1]:30j, bbox[2]:bbox[3]:30j]
        cpositions = np.vstack([cxx.ravel(), cyy.ravel()])
        scores = np.reshape(
            self.target.score(torch.Tensor(cpositions.T)).detach().numpy(),
            cpositions.T.shape,
        )

        ax.axis(bbox)
        ax.set_aspect(abs(bbox[1] - bbox[0]) / abs(bbox[3] - bbox[2]))
        percentiles = np.linspace(70, 99.99999, 10)
        levels = np.percentile(f, percentiles)
        ax.contourf(xx, yy, f, cmap="Blues", alpha=0.8, levels=levels)
        ax.plot(samples[:, 0], samples[:, 1], ".", markersize=1, color="#ff7f0e")
        ax.quiver(cxx, cyy, scores[:, 0], scores[:, 1], width=0.002)

        if save_to_path is not None:
            plt.savefig(save_to_path, bbox_inches="tight")

    def sample(self, batch_size, log_std_min=-3):
        samp_z = torch.randn(batch_size, self.latent_dim)
        mean, log_std = self.transform(samp_z)
        log_std = log_std.clamp(log_std_min)
        samp_x_raw = torch.randn_like(mean)
        return samp_x_raw * log_std.exp() + mean

    def mala(self, samp_x, stepsz, anneal_coef=1.0):
        random_noise = torch.randn_like(samp_x)
        prop_x = (
            samp_x
            + stepsz * anneal_coef * self.target.score(samp_x)
            + math.sqrt(2 * stepsz) * random_noise
        )

        logq_prop_x = torch.sum(
            -0.5 * math.log(2 * math.pi)
            - 0.5 * math.log(2 * stepsz)
            - 0.5 * random_noise ** 2,
            dim=-1,
        )
        backward_mean = prop_x + stepsz * anneal_coef * self.target.score(prop_x)
        logq_prop_x_backward = torch.sum(
            -0.5 * math.log(2 * math.pi)
            - 0.5 * math.log(2 * stepsz)
            - 0.5 * (samp_x - backward_mean) ** 2 / (2 * stepsz),
            dim=-1,
        )

        curr_logp_x = anneal_coef * self.target.logp(samp_x)
        prop_logp_x = anneal_coef * self.target.logp(prop_x)
        log_accept_ratio = (
            prop_logp_x + logq_prop_x_backward - curr_logp_x - logq_prop_x
        )
        accept = torch.log(torch.rand_like(log_accept_ratio)) < log_accept_ratio

        if samp_x.dim() == 1:
            return prop_x if accept else samp_x
        return torch.where(accept.unsqueeze(-1), prop_x, samp_x), accept.float().mean()

    def sgld(self, samp_x, stepsz, anneal_coef=1.0):
        return (
            samp_x
            + stepsz * anneal_coef * self.target.score(samp_x)
            + math.sqrt(2 * stepsz) * torch.randn_like(samp_x)
        )

    def drift_loss(self, stepsz, anneal_coef, batch_size=32, method="sgld"):
        samp_x = self.sample(batch_size)
        if method == "sgld":
            next_x = self.sgld(samp_x, stepsz, anneal_coef)
            # SGLD has no accept/reject: every proposal is accepted (mirrors
            # KDVI's sgld_transition accept_rate == 1.0).
            accept_rate = 1.0
        elif method == "mala":
            next_x, accept_rate = self.mala(samp_x, stepsz, anneal_coef)
            accept_rate = float(accept_rate)
        else:
            raise NotImplementedError
        loss = maximum_mean_discrepancy(next_x.detach(), samp_x)
        return loss, accept_rate

    def learn(
        self,
        stepsz,
        drift_stepsz=0.1,
        max_iter=10000,
        batch_size=32,
        test_freq=1000,
        anneal_freq=10000,
        anneal_rate=0.75,
        warm_up_interval=5000,
        method="sgld",
        kl_eval_freq=None,
        eval_callback=None,
        accept_log_path=None,
        rng_isolation=False,
    ):
        optimizer = torch.optim.Adam(self.parameters(), lr=stepsz)

        # Silent per-step acceptance-rate tracker. Streams to its own CSV
        # (step,accept_rate) without any console output, mirroring KDVI's
        # per-step "kdvi/accept_rate" scalar. accept_rate is the mean MALA
        # accept fraction over the batch (1.0 for SGLD).
        accept_fh = None
        if accept_log_path is not None:
            os.makedirs(os.path.dirname(accept_log_path), exist_ok=True)
            accept_fh = open(accept_log_path, "w")
            accept_fh.write("step,accept_rate\n")

        try:
            for i in range(1, max_iter + 1):
                anneal_coef = min(1.0, 0.1 + i * 1.0 / warm_up_interval)
                loss, accept_rate = self.drift_loss(
                    drift_stepsz / anneal_coef,
                    anneal_coef,
                    batch_size=batch_size,
                    method=method,
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if accept_fh is not None:
                    accept_fh.write(f"{i},{accept_rate:.6f}\n")

                # Parity RNG isolation: snapshot the RNG after the training
                # step so the diagnostic draws below (KSD print, eval_callback)
                # do not advance the training RNG stream. This makes the run
                # byte-identical to KDVI (which isolates eval RNG the same way)
                # regardless of test_freq / kl_eval_freq cadence.
                _rng_state = (
                    torch.get_rng_state() if rng_isolation else None)

                if i % test_freq == 0:
                    with torch.no_grad():
                        samp_x = self.sample(2000)
                        target_x = self.target.sample(2000)
                        ksd_est = maximum_mean_discrepancy(target_x, samp_x, test=True)
                    print(
                        "[Iter {:d}/{:d}] [KSD loss: {:.4f}]".format(
                            i, max_iter, ksd_est
                        ),
                        flush=True,
                    )
                    # Periodically flush the accept-rate buffer to disk so the
                    # tracker survives long runs / interruptions.
                    if accept_fh is not None:
                        accept_fh.flush()

                # KDVI-aligned evaluation + contour plotting cadence. Fires
                # every ``kl_eval_freq`` iterations and always on the final
                # iteration so the schedule matches KDVI's
                # ``train.log.metric_log_freq``.
                if (
                    eval_callback is not None
                    and kl_eval_freq is not None
                    and (i % kl_eval_freq == 0 or i == max_iter)
                ):
                    eval_callback(i)

                if i % anneal_freq == 0:
                    for g in optimizer.param_groups:
                        g["lr"] *= anneal_rate

                # Restore the pre-diagnostic RNG so eval/plot draws leave the
                # training RNG stream untouched.
                if _rng_state is not None:
                    torch.set_rng_state(_rng_state)
        finally:
            if accept_fh is not None:
                accept_fh.close()


# ---------------------------------------------------------------------------
# KL ITE evaluation (fixed ground-truth reference)
# ---------------------------------------------------------------------------
def kl_ite_estimate(q_samples_np: np.ndarray, ref_samples_np: np.ndarray,
                    cost_obj=None) -> float:
    """Estimate KL( q_phi(x) || p_target(x) ) with ``ite.cost.BDKL_KnnK``.

    This mirrors ``runner/base_runner.py::evaluate_vi_to_baseline_kl`` exactly:
    the first argument is the model (``q_phi``) sample matrix and the second is
    the ground-truth reference sample matrix, so the estimator returns
    KL(q || ref). Both inputs must be ``float64`` ``(N, D)`` arrays.

    Args:
        q_samples_np: Model samples, shape ``(N, D)``.
        ref_samples_np: Fixed ground-truth reference samples, shape ``(M, D)``.
        cost_obj: Optional reused ``ite.cost.BDKL_KnnK`` instance (the estimator
            is stateless, so reusing one instance is equivalent to creating a
            fresh one per call as KDVI does).

    Returns:
        Estimated KL divergence as a Python float.
    """
    if cost_obj is None:
        cost_obj = ite.cost.BDKL_KnnK()
    return float(cost_obj.estimation(q_samples_np, ref_samples_np))


def default_reference_path(target_name: str, num: int, seed: int) -> str:
    """Canonical path for a pregenerated fixed ground-truth reference file.

    Lives under the project ``baselines/exact/`` folder so the same file can be
    consumed by KDVI via a ``target.baseline_path`` override. The naming scheme
    must stay identical to ``scripts/generate_gt_reference.py``.
    """
    return os.path.join(
        _ROOT, "baselines", "exact",
        f"{target_name}_gt{num}_seed{seed}.pt",
    )


def generate_reference_samples(target_model, num: int, seed: int,
                               path: str) -> torch.Tensor:
    """Generate and persist a fixed ground-truth reference set.

    Draws ``num`` exact samples from ``target_model`` under a dedicated,
    run-independent seed and saves them as a plain ``float32`` ``(num, D)``
    tensor. The generation logic is intentionally identical to
    ``scripts/generate_gt_reference.py`` so both entry points produce a
    byte-identical file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.manual_seed(seed)
    with torch.no_grad():
        samples = target_model.sample(num).detach().cpu().to(torch.float32)
    torch.save(samples, path)
    print(f"[run_ivi] generated reference samples -> {path} "
          f"(shape={tuple(samples.shape)}, seed={seed})", flush=True)
    return samples


def load_or_generate_reference(target_model, num: int, seed: int,
                               path: str) -> torch.Tensor:
    """Load the fixed reference file, generating it deterministically if absent.

    Returns a ``float32`` CPU tensor of shape ``(num, D)``.
    """
    if os.path.exists(path):
        data = torch.load(path, map_location="cpu")
        if isinstance(data, dict):
            data = data["samples"]
        samples = torch.as_tensor(data, dtype=torch.float32)
        print(f"[run_ivi] loaded reference samples <- {path} "
              f"(shape={tuple(samples.shape)})", flush=True)
        if samples.shape[0] != num:
            print(f"[run_ivi] WARNING: reference file has {samples.shape[0]} "
                  f"rows but --ref-num={num}; using the file as-is.",
                  file=sys.stderr, flush=True)
        return samples
    return generate_reference_samples(target_model, num, seed, path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iter", type=int, default=100000,
                        help="max_iter for model.learn (default 100000, matches notebook)")
    parser.add_argument("--test-freq", type=int, default=1000,
                        help=(
                            "Frequency (in iterations) of the KSD-print "
                            "diagnostic inside model.learn (default 1000, "
                            "matches notebook). This is the console KSD print "
                            "only; contour plots are driven by --kl-eval-freq."
                        ))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=32,
        help=(
            "Dimension of the standard-normal latent input to the IVI "
            "transform (default 32, matching ImVIDrift's native default)."
        ),
    )
    parser.add_argument(
        "--rng-isolation", action="store_true",
        help=(
            "Reset the training RNG to --seed right before the loop and make "
            "all diagnostic (KSD-print / KL-eval / contour) draws RNG-neutral, "
            "so the training stream is a pure function of (seed, step). Enables "
            "byte-identical parity with KDVI run with "
            "train.parity_rng_isolation=true."
        ),
    )
    parser.add_argument("--kl-num-samples", type=int, default=5000,
                        help="Number of q_phi samples drawn per KL_ITE eval.")
    parser.add_argument(
        "--kl-eval-freq", type=int, default=5000,
        help=(
            "Cadence (in iterations) of the KDVI-aligned KL_ITE evaluation + "
            "contour plot. KL_ITE is also always evaluated on the final "
            "iteration. Set <=0 to disable mid-run evaluation (final-only)."
        ),
    )
    parser.add_argument(
        "--results-dir", type=str, default="results",
        help=(
            "Base directory for run artifacts. Outputs are written to "
            "<results-dir>/IVI/<target>/<timestamp>/, mirroring the KDVI "
            "layout results/<runner>/<target>/<timestamp>/."
        ),
    )
    parser.add_argument(
        "--ref-samples-path", type=str, default=None,
        help=(
            "Path to the fixed pregenerated ground-truth reference file used "
            "for KL_ITE. Defaults to "
            "baselines/exact/<target>_gt<ref-num>_seed<ref-seed>.pt. The file "
            "is generated deterministically if it does not exist. Point KDVI's "
            "target.baseline_path at this same file for a step-identical "
            "comparison."
        ),
    )
    parser.add_argument("--ref-num", type=int, default=5000,
                        help="Number of fixed ground-truth reference samples.")
    parser.add_argument(
        "--ref-seed", type=int, default=0,
        help=(
            "Seed used ONLY to generate the fixed reference set. Kept "
            "independent of --seed so every run/seed compares against the same "
            "ground-truth reference."
        ),
    )
    parser.add_argument("--plot-num", type=int, default=2000,
                        help="Number of q_phi samples overlaid on contour plots.")
    parser.add_argument(
        "--target",
        type=str,
        default="gmm",
        choices=["gmm", "8_gaussians", "8_gaussians_small"],
        help=(
            "Target distribution. 'gmm' = notebook GMM(8, 0.1, 1). "
            "'8_gaussians' = project EightGaussians (r=4, sigma=0.5). "
            "'8_gaussians_small' = project EightGaussiansSmall "
            "(r=1, sigma=0.1)."
        ),
    )
    parser.add_argument(
        "--drift-stepsz",
        type=float,
        default=None,
        help=(
            "Override the second positional arg of model.learn (the MALA "
            "step size). Default None = pick by --target: 0.01 for the "
            "small geometry (notebook), 0.25 for the large 8_gaussians "
            "(scaled by sigma^2 ratio so MALA covers a mode at the same "
            "rate)."
        ),
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        default=None,
        help=(
            "Override contour-plot bbox as 'xmin xmax ymin ymax'. "
            "Defaults to [-1.5, 1.5, -1.5, 1.5] for gmm/small, "
            "[-6, 6, -6, 6] for 8_gaussians."
        ),
    )
    args = parser.parse_args()

    # ---- Build the target (project EightGaussians / EightGaussiansSmall) ----
    target_model = build_target(args.target)

    if args.bbox is not None:
        bbox = list(args.bbox)
    elif args.target == "8_gaussians":
        bbox = [-6.0, 6.0, -6.0, 6.0]
    else:
        bbox = [-1.5, 1.5, -1.5, 1.5]

    # ---- Fixed ground-truth reference for KL_ITE ----
    # Generated/loaded BEFORE the run RNG is seeded so the reference depends
    # only on --ref-seed (run-independent), and the model/training RNG depends
    # only on --seed.
    ref_path = (
        args.ref_samples_path
        if args.ref_samples_path is not None
        else default_reference_path(args.target, args.ref_num, args.ref_seed)
    )
    ref_samples = load_or_generate_reference(
        target_model, args.ref_num, args.ref_seed, ref_path)
    ref_np = ref_samples.cpu().numpy().astype(np.float64)

    # ---- Seed run RNG (model init + training) ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = ImVIDrift(
        target_model,
        hidden_units=256,
        latent_dim=args.latent_dim,
    )
    model._bbox = bbox  # legacy field, kept for compatibility

    if args.drift_stepsz is not None:
        drift_stepsz = float(args.drift_stepsz)
    elif args.target == "8_gaussians":
        # Project EightGaussians has sigma=0.5, vs notebook sigma=0.1, so
        # variance is 25x larger. Scale MALA step size to match.
        drift_stepsz = 0.25
    else:
        drift_stepsz = 0.01

    # ---- Artifact layout: results/IVI/<target>/<timestamp>/ (KDVI style) ----
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(args.results_dir, "IVI", args.target, timestamp)
    plots_dir = os.path.join(save_path, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    print(f"Artifacts will be saved to: {save_path}", flush=True)

    # Run provenance.
    run_config = {
        **vars(args),
        "drift_stepsz_resolved": drift_stepsz,
        "bbox": bbox,
        "save_path": save_path,
        "reference_path": os.path.abspath(ref_path),
        "reference_shape": list(ref_samples.shape),
        "timestamp": timestamp,
    }
    with open(os.path.join(save_path, "run_config.json"), "w") as fh:
        json.dump(run_config, fh, indent=2)

    # KL_ITE trajectory CSV (step,kl), matching KDVI's per-eval logging.
    kl_csv_path = os.path.join(save_path, "kl_ite.csv")
    with open(kl_csv_path, "w") as fh:
        fh.write("step,kl\n")

    # Reused estimator instance (stateless; equivalent to KDVI's fresh-per-call
    # BDKL_KnnK()).
    kl_cost = ite.cost.BDKL_KnnK()
    kl_history: list[tuple[int, float]] = []

    def eval_callback(step: int) -> None:
        """KDVI-aligned KL_ITE evaluation + contour plot at ``step``."""
        with torch.no_grad():
            q_kl = model.sample(args.kl_num_samples).cpu().numpy().astype(
                np.float64)
        kl = kl_ite_estimate(q_kl, ref_np, cost_obj=kl_cost)
        kl_history.append((step, kl))
        print(f"[KL_ITE] step={step} kl={kl:.6f}", flush=True)
        with open(kl_csv_path, "a") as fh:
            fh.write(f"{step},{kl:.6f}\n")

        # Contour plot via the project target's contour_plot (exact KDVI path).
        with torch.no_grad():
            q_plot = model.sample(args.plot_num).cpu().numpy()
        plot_path = os.path.join(plots_dir, f"contour_epoch_{step}.png")
        try:
            target_model.contour_plot(
                bbox,
                fnet=None,
                samples=q_plot,
                save_to_path=plot_path,
                quiver=False,
                t=step,
            )
        except (AttributeError, TypeError):
            # Fallback for non-project targets (e.g. 'gmm') that lack the
            # KDVI-style contour_plot signature.
            fig, ax = plt.subplots(figsize=(5, 5))
            model.contour_plot(bbox, ax, samples=q_plot,
                               save_to_path=plot_path)
            plt.close()

    kl_eval_freq = args.kl_eval_freq if args.kl_eval_freq and args.kl_eval_freq > 0 else None

    # ---- Pre-training visualisation (KDVI-style contour) ----
    with torch.no_grad():
        pre_samples = model.sample(args.plot_num).cpu().numpy()
    pre_path = os.path.join(plots_dir, "contour_pre_training.png")
    try:
        target_model.contour_plot(
            bbox, fnet=None, samples=pre_samples,
            save_to_path=pre_path, quiver=False, t=0,
        )
    except (AttributeError, TypeError):
        fig, ax = plt.subplots(figsize=(5, 5))
        model.contour_plot(bbox, ax, samples=pre_samples, save_to_path=pre_path)
        plt.close()

    # ---- Only the FIRST model.learn call from the notebook is kept ----
    print(f"[run_ivi] target={args.target} latent_dim={args.latent_dim} "
          f"drift_stepsz={drift_stepsz} max_iter={args.max_iter} "
          f"kl_eval_freq={kl_eval_freq}", flush=True)
    # Parity RNG isolation: reset the training RNG stream to the run seed right
    # before training so the pre-training contour-plot draw above does not
    # offset it. KDVI does the same (train.parity_rng_isolation), so both begin
    # training from the identical RNG state.
    if args.rng_isolation:
        torch.manual_seed(args.seed)
    model.learn(
        0.001,
        drift_stepsz,
        args.max_iter,
        batch_size=128,
        test_freq=args.test_freq,
        warm_up_interval=50000,
        anneal_freq=5000,
        anneal_rate=0.75,
        method="mala",
        kl_eval_freq=kl_eval_freq,
        eval_callback=eval_callback,
        accept_log_path=os.path.join(save_path, "accept_rate.csv"),
        rng_isolation=args.rng_isolation,
    )

    # ---- Final KL ITE summary ----
    # When mid-run evaluation is enabled the final iteration was already
    # evaluated inside the loop; reuse that value to avoid a redundant draw.
    if kl_history:
        final_step, kl_div = kl_history[-1]
    else:
        with torch.no_grad():
            q_final = model.sample(args.kl_num_samples).cpu().numpy().astype(
                np.float64)
        kl_div = kl_ite_estimate(q_final, ref_np, cost_obj=kl_cost)
        final_step = args.max_iter

    print("=" * 60)
    print(f"Target             : {args.target}  (name={target_model.name})")
    print(f"latent_dim         : {args.latent_dim}")
    print(f"drift_stepsz       : {drift_stepsz}")
    print(f"final step         : {final_step}")
    print(f"KL ITE (BDKL_KnnK)  KL( q_phi(x) || p_target(x) ) = {kl_div:.6f}")
    print(f"  vi samples       : {args.kl_num_samples}")
    print(f"  ref samples      : {ref_np.shape[0]}  (fixed, {ref_path})")
    print(f"  artifacts        : {save_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
