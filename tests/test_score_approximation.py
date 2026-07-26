from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from finalization.artifacts import RunRecord
from finalization.score_approximation import (
    CellSpec,
    atomic_write_json,
    autograd_mixture_score,
    cell_record_path,
    compute_score_metrics,
    diagonal_gaussian_mixture_block,
    mixture_block_summary,
    native_aisivi_score,
    native_sivi_score,
    pending_cell_specs,
    select_progress_checkpoints,
    streamed_reference_score,
)
from models.vi_model import ConditionalGaussian


def make_model(*, dtype: torch.dtype = torch.float64) -> ConditionalGaussian:
    cfg = OmegaConf.create({
        "z_dim": 2,
        "epsilon_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "device": "cpu",
        "uniform": False,
    })
    return ConditionalGaussian(cfg).to(dtype=dtype)


class FakeReverse:

    def __init__(
        self,
        epsilon: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        self.epsilon = epsilon
        self.log_prob_value = log_prob

    def sample(
        self,
        z: torch.Tensor,
        *,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_samples != self.epsilon.shape[1]:
            raise AssertionError("Unexpected sample count")
        z_aux = z.unsqueeze(1).expand(-1, num_samples, -1)
        return z_aux, self.epsilon, self.log_prob_value


class ScoreApproximationTest(unittest.TestCase):

    def setUp(self) -> None:
        torch.manual_seed(123)

    def test_analytic_mixture_score_matches_autograd(self) -> None:
        model = make_model()
        z = torch.randn(5, 2, dtype=torch.float64)
        epsilon = torch.randn(7, 4, dtype=torch.float64)
        with torch.no_grad():
            mu, raw = model.net(epsilon).chunk(2, dim=-1)
            var, _ = model._variance_from_raw(raw)
            _, analytic = diagonal_gaussian_mixture_block(z, mu, var)
        expected = autograd_mixture_score(model, z, epsilon)
        torch.testing.assert_close(
            analytic,
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_streamed_reference_matches_monolithic_components(self) -> None:
        model = make_model()
        z = torch.randn(4, 2, dtype=torch.float64)
        batches = [
            torch.randn(3, 4, dtype=torch.float64),
            torch.randn(3, 4, dtype=torch.float64),
            torch.randn(3, 4, dtype=torch.float64),
        ]
        with torch.no_grad():
            _, expected = mixture_block_summary(
                model,
                z,
                torch.cat(batches, dim=0),
            )
        with patch.object(
            model,
            "sample_epsilon",
            side_effect=batches,
        ):
            actual = streamed_reference_score(
                model,
                z,
                reverse_batch_size=3,
                num_batches=3,
                accumulator_dtype=torch.float64,
            )
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )

    def test_native_sivi_score_matches_training_mixture_autograd(self) -> None:
        model = make_model()
        z = torch.randn(4, 2, dtype=torch.float64)
        generating_epsilon = torch.randn(4, 4, dtype=torch.float64)
        auxiliary = torch.randn(5, 4, dtype=torch.float64)
        runner = SimpleNamespace(
            vi_model=model,
            training_reverse_sample_num=5,
        )
        with patch.object(model, "sample_epsilon", return_value=auxiliary):
            actual, diagnostics = native_sivi_score(
                runner,
                z,
                generating_epsilon,
            )

        z_grad = z.detach().clone().requires_grad_(True)
        epsilon_aux = auxiliary.unsqueeze(0).expand(z.shape[0], -1, -1)
        epsilon_all = torch.cat(
            [epsilon_aux, generating_epsilon.unsqueeze(1)],
            dim=1,
        )
        z_all = z_grad.unsqueeze(1).expand(-1, epsilon_all.shape[1], -1)
        log_terms = model.logp(z_all, epsilon_all)
        expected = torch.autograd.grad(
            torch.logsumexp(log_terms, dim=1).sum(),
            z_grad,
        )[0]
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        self.assertEqual(diagnostics["native_auxiliary_samples"], 6)

    def test_native_aisivi_score_matches_detached_weight_autograd(self) -> None:
        model = make_model()
        n, k = 3, 4
        z = torch.randn(n, 2, dtype=torch.float64)
        epsilon = torch.randn(n, k, 4, dtype=torch.float64)
        log_q_reverse = torch.randn(n, k, dtype=torch.float64)
        reverse = FakeReverse(epsilon, log_q_reverse)
        runner = SimpleNamespace(
            vi_model=model,
            reverse_model=reverse,
            training_reverse_sample_num=k,
            normalize_reverse_score=False,
        )
        actual, diagnostics = native_aisivi_score(runner, z)

        with torch.no_grad():
            importance = (
                model.log_q_epsilon(epsilon) - log_q_reverse
            ).clamp(max=10.0)
        z_grad = z.detach().clone().requires_grad_(True)
        z_aux = z_grad.unsqueeze(1).expand(-1, k, -1)
        log_terms = model.logp(z_aux, epsilon) + importance
        expected = torch.autograd.grad(
            torch.logsumexp(log_terms, dim=1).sum(),
            z_grad,
        )[0]
        torch.testing.assert_close(
            actual,
            expected,
            rtol=1.0e-10,
            atol=1.0e-10,
        )
        self.assertEqual(diagnostics["native_auxiliary_samples"], k)

    def test_checkpoint_progress_selects_exact_epochs(self) -> None:
        checkpoints = [
            (epoch, Path(f"/tmp/epoch_{epoch}/vi_model.pt"))
            for epoch in range(1000, 10001, 1000)
        ]
        selected = select_progress_checkpoints(
            checkpoints,
            total_epochs=10000,
            progresses=[0.2, 0.4, 0.6, 0.8, 1.0],
        )
        self.assertEqual(
            [epoch for _, epoch, _ in selected],
            [2000, 4000, 6000, 8000, 10000],
        )

    def test_score_metric_definitions(self) -> None:
        method = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        references = torch.tensor([
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ])
        metrics = compute_score_metrics(method, references)
        self.assertAlmostEqual(metrics["method_l2"], 0.0)
        self.assertAlmostEqual(metrics["reference_internal_l2"], 1.0)
        self.assertEqual(
            metrics["reference_repeat_internal_l2"],
            [1.0, 1.0],
        )

    def test_resume_skips_matching_atomic_cell(self) -> None:
        rec = RunRecord(
            run_id="run-1",
            seed=42,
            method="SIVI",
            target="x_shaped",
            runner_type="SIVI",
            config_path=Path("config.yaml"),
            result_path=Path("results"),
            duration_sec=None,
            status="completed",
            entry={},
        )
        spec = CellSpec(
            record=rec,
            progress=0.2,
            epoch=2000,
            checkpoint_dir=Path("checkpoint"),
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            fingerprint = "abc"
            atomic_write_json(
                cell_record_path(root, spec),
                {
                    "analysis_fingerprint": fingerprint,
                    "cell_key": spec.key,
                },
            )
            self.assertEqual(
                pending_cell_specs(
                    [spec],
                    run_root=root,
                    fingerprint=fingerprint,
                    resume=True,
                ),
                [],
            )
            self.assertEqual(
                pending_cell_specs(
                    [spec],
                    run_root=root,
                    fingerprint=fingerprint,
                    resume=False,
                ),
                [spec],
            )


if __name__ == "__main__":
    unittest.main()
