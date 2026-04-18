from __future__ import annotations

import math
import unittest

import torch
import torch.nn as nn

from utils.elm import (
    estimate_log_q_prior,
    estimate_log_q_reverse_is,
    fit_reverse_proposal,
    kde_expected_log_marginal,
    summarize_elm,
)


class ToyVIModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.device = torch.device("cpu")
        self.z_dim = 2
        self.epsilon_dim = 2
        self.uniform = False
        self._epsilon_pool: torch.Tensor | None = None
        self._epsilon_cursor = 0
        self.weight = nn.Parameter(
            torch.tensor(
                [
                    [1.1, -0.3],
                    [0.4, 0.8],
                ],
                dtype=torch.float32,
            )
        )
        self.bias = nn.Parameter(torch.tensor([0.2, -0.1], dtype=torch.float32))
        self.log_var = nn.Parameter(torch.log(torch.tensor([0.35, 0.5], dtype=torch.float32)))

    def set_epsilon_pool(self, epsilon_pool: torch.Tensor | None) -> None:
        self._epsilon_pool = epsilon_pool
        self._epsilon_cursor = 0

    def sample_epsilon(self, num: int = 1000) -> torch.Tensor:
        if self._epsilon_pool is not None:
            start = self._epsilon_cursor
            end = start + num
            if end > int(self._epsilon_pool.shape[0]):
                raise RuntimeError("Deterministic epsilon pool exhausted.")
            self._epsilon_cursor = end
            return self._epsilon_pool[start:end].clone().to(self.device)
        return torch.randn(num, self.epsilon_dim, device=self.device)

    def sampling(self, num: int = 1000) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = self.sample_epsilon(num)
        mu = torch.matmul(epsilon, self.weight.transpose(0, 1)) + self.bias
        std = torch.exp(0.5 * self.log_var)
        z = mu + std * torch.randn_like(mu)
        return epsilon.detach(), z.detach()

    def log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        const = -0.5 * epsilon.shape[-1] * math.log(2.0 * math.pi)
        return const - 0.5 * (epsilon**2).sum(dim=-1)

    def logp(self, z: torch.Tensor, epsilon: torch.Tensor) -> torch.Tensor:
        mu = torch.matmul(epsilon, self.weight.transpose(0, 1)) + self.bias
        var = torch.exp(self.log_var)
        const = -0.5 * z.shape[-1] * math.log(2.0 * math.pi)
        return const - 0.5 * (torch.log(var).sum() + ((z - mu) ** 2 / var).sum(dim=-1))


class ElmApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model = ToyVIModel()
        torch.manual_seed(7)
        _, self.reference_samples = self.model.sampling(num=12)

    def test_prior_estimator_returns_one_value_per_reference(self) -> None:
        estimate = estimate_log_q_prior(
            self.model,
            self.reference_samples,
            num_samples=25,
            epsilon_batch_size=7,
        )
        self.assertEqual(tuple(estimate.log_q_values.shape), (12,))
        self.assertIsNone(estimate.ess_values)
        self.assertEqual(estimate.diagnostics["epsilon_batch_size"], 7)

    def test_prior_estimator_is_batch_size_invariant_for_same_seed(self) -> None:
        generator = torch.Generator(device=self.model.device).manual_seed(123)
        epsilon_pool = torch.randn(24, self.model.epsilon_dim, generator=generator)
        self.model.set_epsilon_pool(epsilon_pool)
        estimate_a = estimate_log_q_prior(
            self.model,
            self.reference_samples,
            num_samples=24,
            epsilon_batch_size=6,
        )
        self.model.set_epsilon_pool(epsilon_pool)
        estimate_b = estimate_log_q_prior(
            self.model,
            self.reference_samples,
            num_samples=24,
            epsilon_batch_size=8,
        )
        self.model.set_epsilon_pool(None)
        self.assertTrue(
            torch.allclose(
                estimate_a.log_q_values,
                estimate_b.log_q_values,
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_summarize_elm_matches_mean_and_stderr(self) -> None:
        estimate = estimate_log_q_prior(
            self.model,
            self.reference_samples,
            num_samples=18,
            epsilon_batch_size=5,
        )
        summary = summarize_elm(estimate)
        expected_mean = float(estimate.log_q_values.mean().item())
        expected_stderr = float(
            estimate.log_q_values.std(unbiased=True).item()
            / math.sqrt(int(estimate.log_q_values.numel()))
        )
        self.assertAlmostEqual(summary.value, expected_mean, places=6)
        self.assertAlmostEqual(summary.stderr, expected_stderr, places=6)
        self.assertTrue(torch.equal(summary.log_q_values, estimate.log_q_values))

    def test_reverse_is_returns_log_q_and_ess_per_reference(self) -> None:
        proposal_fit = fit_reverse_proposal(
            self.model,
            proposal_type="gaussian",
            num_fit_samples=64,
            fit_batch_size=32,
            fit_epochs=2,
        )
        estimate = estimate_log_q_reverse_is(
            self.model,
            proposal_fit.reverse_model,
            self.reference_samples,
            num_is_samples=20,
            is_batch_size=7,
            proposal_cache=proposal_fit.cache,
        )
        self.assertEqual(tuple(estimate.log_q_values.shape), (12,))
        self.assertIsNotNone(estimate.ess_values)
        self.assertEqual(tuple(estimate.ess_values.shape), (12,))
        self.assertEqual(estimate.diagnostics["num_is_samples"], 20)

    def test_fit_reverse_proposal_covers_direct_and_optimizer_modes(self) -> None:
        gaussian_fit = fit_reverse_proposal(
            self.model,
            proposal_type="gaussian",
            num_fit_samples=64,
            fit_batch_size=32,
            fit_epochs=2,
        )
        self.assertEqual(gaussian_fit.fit_mode, "direct_fit")
        self.assertIsNotNone(gaussian_fit.cache)
        self.assertIn("fit_nll", gaussian_fit.diagnostics)

        realnvp_fit = fit_reverse_proposal(
            self.model,
            proposal_type="realnvp",
            num_fit_samples=32,
            fit_batch_size=16,
            fit_epochs=2,
            log_every=0,
        )
        self.assertEqual(realnvp_fit.fit_mode, "optimizer")
        self.assertIsNone(realnvp_fit.cache)
        self.assertIn("fit_loss_initial", realnvp_fit.diagnostics)
        self.assertIn("fit_nll", realnvp_fit.diagnostics)

    def test_kde_expected_log_marginal_matches_scipy_1d_marginals(self) -> None:
        try:
            from scipy.stats import gaussian_kde
        except Exception:  # pragma: no cover - scipy is optional for this unit check
            self.skipTest("scipy is not available")

        model_samples = torch.tensor(
            [
                [-1.0, 0.2],
                [-0.2, 0.5],
                [0.1, 0.9],
                [0.7, 1.2],
                [1.3, 1.8],
            ],
            dtype=torch.float64,
        )
        reference_samples = torch.tensor(
            [
                [-0.5, 0.4],
                [0.3, 1.0],
                [1.1, 1.5],
            ],
            dtype=torch.float64,
        )
        estimate = kde_expected_log_marginal(
            reference_samples,
            model_samples,
            dim_chunk=1,
            ref_chunk=2,
            model_chunk=2,
            dtype=torch.float64,
            device="cpu",
        )

        expected = torch.zeros(reference_samples.shape[0], dtype=torch.float64)
        for dim in range(model_samples.shape[1]):
            kde = gaussian_kde(model_samples[:, dim].numpy())
            density = kde(reference_samples[:, dim].numpy())
            expected += torch.log(torch.from_numpy(density))

        self.assertTrue(
            torch.allclose(
                estimate.per_reference_log_values,
                expected,
                atol=1e-10,
                rtol=1e-10,
            )
        )
        self.assertAlmostEqual(
            estimate.value,
            float(expected.mean().item()),
            places=10,
        )

    def test_kde_expected_log_marginal_is_chunk_invariant(self) -> None:
        generator = torch.Generator().manual_seed(101)
        model_samples = torch.randn(17, 4, generator=generator, dtype=torch.float64)
        reference_samples = torch.randn(9, 4, generator=generator, dtype=torch.float64)

        estimate_a = kde_expected_log_marginal(
            reference_samples,
            model_samples,
            dim_chunk=1,
            ref_chunk=3,
            model_chunk=4,
            dtype=torch.float64,
            device="cpu",
        )
        estimate_b = kde_expected_log_marginal(
            reference_samples,
            model_samples,
            dim_chunk=3,
            ref_chunk=5,
            model_chunk=7,
            dtype=torch.float64,
            device="cpu",
        )

        self.assertTrue(
            torch.allclose(
                estimate_a.per_reference_log_values,
                estimate_b.per_reference_log_values,
                atol=1e-10,
                rtol=1e-10,
            )
        )
        self.assertAlmostEqual(estimate_a.value, estimate_b.value, places=10)

    def test_kde_expected_log_marginal_clamps_constant_dimensions(self) -> None:
        model_samples = torch.ones(6, 2)
        reference_samples = torch.ones(3, 2)

        estimate = kde_expected_log_marginal(
            reference_samples,
            model_samples,
            dim_chunk=2,
            ref_chunk=2,
            model_chunk=3,
            min_bandwidth=1.0e-6,
            dtype=torch.float32,
            device="cpu",
        )

        self.assertEqual(estimate.diagnostics["num_bandwidth_clamped_dims"], 2)
        self.assertTrue(torch.all(estimate.bandwidths == torch.tensor(1.0e-6)))
        self.assertTrue(torch.isfinite(estimate.per_reference_log_values).all())


if __name__ == "__main__":
    unittest.main()
