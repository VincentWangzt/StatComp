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
    _summary_rows,
    _write_csv,
    assess_hmc_reference_quality,
    atomic_write_json,
    autograd_mixture_score,
    cell_record_path,
    compute_score_metrics,
    diagonal_gaussian_mixture_block,
    gelman_rubin_rhat,
    mixture_block_summary,
    native_aisivi_score,
    native_sivi_score,
    pending_cell_specs,
    posterior_hmc_reference_scores,
    render_score_approximation_figures,
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


class LinearGaussianVI(torch.nn.Module):
    """One-dimensional model with an analytic epsilon posterior."""

    def __init__(self, conditional_variance: float = 0.5) -> None:
        super().__init__()
        self.conditional_variance = conditional_variance
        self.epsilon_dim = 1

    def log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
            torch.log(
                torch.tensor(
                    2.0 * torch.pi,
                    dtype=epsilon.dtype,
                    device=epsilon.device,
                )
            )
            + epsilon.square().sum(dim=-1)
        )

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        variance = torch.as_tensor(
            self.conditional_variance,
            dtype=z.dtype,
            device=z.device,
        )
        return -0.5 * (
            torch.log(2.0 * torch.pi * variance)
            + ((z - epsilon).square() / variance).sum(dim=-1)
        )

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        return -(z - epsilon) / self.conditional_variance


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

    def test_posterior_hmc_recovers_linear_gaussian_score(self) -> None:
        torch.manual_seed(321)
        model = LinearGaussianVI().to(dtype=torch.float64)
        z = torch.tensor(
            [[-1.5], [-0.5], [0.5], [1.5]],
            dtype=torch.float64,
        )
        posterior_variance = 0.5 / 1.5
        posterior_mean = z / 1.5
        generating_epsilon = (
            posterior_mean
            + posterior_variance**0.5 * torch.randn_like(z)
        )
        chain_scores, diagnostics = posterior_hmc_reference_scores(
            model,
            z,
            generating_epsilon,
            total_samples=800,
            num_chains=4,
            burn_in_steps=100,
            thinning=1,
            step_size=0.1,
            leapfrog_steps=5,
            init_jitter_scale=0.1,
            adapt_step_size=True,
            target_acceptance=0.8,
            adaptation_rate=0.1,
            min_step_size=0.01,
            max_step_size=0.2,
            divergence_threshold=1000.0,
            accumulator_dtype=torch.float64,
        )
        expected = -z / 1.5
        actual = chain_scores.mean(dim=0)
        torch.testing.assert_close(
            actual,
            expected,
            rtol=0.0,
            atol=0.16,
        )
        self.assertEqual(tuple(chain_scores.shape), (4, 4, 1))
        self.assertGreater(
            diagnostics["hmc_post_burn_acceptance_rate"],
            0.6,
        )
        self.assertLess(
            diagnostics["hmc_score_rhat_p95"],
            1.2,
        )
        self.assertEqual(diagnostics["hmc_divergence_fraction"], 0.0)

    def test_posterior_hmc_rejects_nondivisible_sample_budget(self) -> None:
        model = LinearGaussianVI().to(dtype=torch.float64)
        z = torch.zeros(2, 1, dtype=torch.float64)
        with self.assertRaisesRegex(ValueError, "divisible"):
            posterior_hmc_reference_scores(
                model,
                z,
                z.clone(),
                total_samples=101,
                num_chains=4,
                burn_in_steps=1,
                thinning=1,
                step_size=0.05,
                leapfrog_steps=1,
                init_jitter_scale=0.0,
                adapt_step_size=False,
                target_acceptance=0.8,
                adaptation_rate=0.0,
                min_step_size=0.01,
                max_step_size=0.1,
                divergence_threshold=1000.0,
            )

    def test_posterior_hmc_conditional_gaussian_cpu_smoke(self) -> None:
        model = make_model(dtype=torch.float32)
        generating_epsilon, z = model.sampling(num=8)
        chain_scores, diagnostics = posterior_hmc_reference_scores(
            model,
            z,
            generating_epsilon,
            total_samples=40,
            num_chains=4,
            burn_in_steps=5,
            thinning=1,
            step_size=0.02,
            leapfrog_steps=2,
            init_jitter_scale=0.01,
            adapt_step_size=True,
            target_acceptance=0.8,
            adaptation_rate=0.1,
            min_step_size=0.001,
            max_step_size=0.05,
            divergence_threshold=1000.0,
            accumulator_dtype=torch.float64,
        )
        self.assertEqual(tuple(chain_scores.shape), (4, 8, 2))
        self.assertTrue(torch.isfinite(chain_scores).all())
        self.assertTrue(
            0.0 <= diagnostics["hmc_acceptance_rate"] <= 1.0
        )

    def test_rhat_detects_shifted_chain(self) -> None:
        stable = torch.randn(2, 4, 100, 1, dtype=torch.float64)
        shifted = stable.clone()
        shifted[:, 0] += 4.0
        self.assertLess(float(gelman_rubin_rhat(stable).max()), 1.1)
        self.assertGreater(float(gelman_rubin_rhat(shifted).min()), 2.0)

    def test_hmc_quality_checks_warn_without_dropping_metrics(self) -> None:
        quality = OmegaConf.create({
            "max_divergence_fraction": 0.01,
            "max_score_rhat_p95": 1.1,
            "max_epsilon_rhat_p95": 2.0,
            "min_post_burn_acceptance_rate": 0.6,
            "min_worst_chain_acceptance_rate": 0.05,
        })
        diagnostics = {
            "hmc_divergence_fraction": 0.0,
            "hmc_score_rhat_p95": 1.2,
            "hmc_epsilon_rhat_p95": 1.5,
            "hmc_post_burn_acceptance_rate": 0.8,
            "hmc_post_burn_acceptance_min": 0.1,
        }
        status, issues = assess_hmc_reference_quality(
            diagnostics,
            quality,
        )
        self.assertEqual(status, "warning")
        self.assertEqual(len(issues), 1)
        self.assertIn("hmc_score_rhat_p95", issues[0])

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
        target = torch.zeros_like(method)
        metrics = compute_score_metrics(method, references, target)
        self.assertAlmostEqual(metrics["method_l2"], 0.0)
        self.assertAlmostEqual(metrics["method_relative_l2"], 0.0)
        self.assertAlmostEqual(metrics["method_target_l2"], 1.0)
        self.assertAlmostEqual(metrics["reference_target_l2"], 1.0)
        self.assertAlmostEqual(metrics["reference_internal_l2"], 1.0)
        self.assertAlmostEqual(metrics["reference_mean_mcse_l2"], 1.0)
        self.assertEqual(
            metrics["reference_repeat_internal_l2"],
            [1.0, 1.0],
        )

    def test_reference_metrics_survive_unavailable_method_score(self) -> None:
        references = torch.tensor([
            [[0.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ])
        metrics = compute_score_metrics(None, references)
        self.assertIsNone(metrics["method_l2"])
        self.assertAlmostEqual(metrics["reference_internal_l2"], 1.0)

    def test_seed_summary_counts_unavailable_native_scores(self) -> None:
        records = [
            {
                "target": "8_gaussians",
                "method": "AISIVI",
                "progress": 0.6,
                "epoch": 6000,
                "method_l2": None if seed == 42 else float(seed),
                "reference_internal_l2": float(seed) / 10.0,
                "method_runtime_sec": 1.0,
                "reference_runtime_sec": 2.0,
            }
            for seed in range(42, 47)
        ]
        summary = _summary_rows(records)
        self.assertEqual(len(summary), 1)
        self.assertEqual(summary[0]["n_seeds"], 5)
        self.assertEqual(summary[0]["method_n_valid"], 4)
        self.assertEqual(summary[0]["method_n_failed"], 1)

    def test_score_figures_render_from_summary_and_diagnostics(self) -> None:
        cfg = OmegaConf.create({
            "selection": {
                "targets": ["x_shaped"],
                "methods": ["SIVI"],
            },
        })
        summary_rows = [
            {
                "target": "x_shaped",
                "method": "SIVI",
                "progress": progress,
                "epoch": epoch,
                "method_l2_mean": value,
                "method_l2_sd": value / 10,
                "method_target_l2_mean": value / 2,
                "method_target_l2_sd": value / 20,
                "reference_target_l2_mean": value / 3,
                "reference_target_l2_sd": value / 30,
                "reference_internal_l2_mean": value / 20,
                "reference_internal_l2_sd": value / 200,
            }
            for progress, epoch, value in [
                (0.2, 2000, 2.0),
                (1.0, 10000, 1.0),
            ]
        ]
        records = [
            {
                "target": "x_shaped",
                "method": "SIVI",
                "epoch": epoch,
                "diagnostics": {
                    "hmc_post_burn_acceptance_rate": 0.85,
                    "hmc_score_rhat_p95": 1.03,
                    "hmc_final_step_size_median": 0.02,
                },
            }
            for epoch in [2000, 10000]
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = render_score_approximation_figures(
                cfg,
                records,
                summary_rows,
                report_dir=Path(temp_dir),
            )
            self.assertEqual(len(paths), 6)
            self.assertTrue(all(path.stat().st_size > 0 for path in paths))

    def test_csv_reports_use_lf_line_endings(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "report.csv"
            _write_csv(path, [{"value": 1}, {"value": 2}])
            payload = path.read_bytes()
        self.assertNotIn(b"\r\n", payload)
        self.assertEqual(payload.count(b"\n"), 3)

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
