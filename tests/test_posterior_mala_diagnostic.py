from __future__ import annotations

import unittest

import torch

from finalization.posterior_mala_diagnostic import (
    classical_effective_sample_size,
    posterior_mala_samples,
)


class StandardNormalPosterior(torch.nn.Module):
    def log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        return -0.5 * epsilon.square().sum(dim=-1)

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        del z
        return torch.zeros(
            epsilon.shape[:-1],
            device=epsilon.device,
            dtype=epsilon.dtype,
        )


class PosteriorMalaDiagnosticTests(unittest.TestCase):
    def test_mala_samples_standard_normal(self) -> None:
        torch.manual_seed(7)
        samples, diagnostics, trace = posterior_mala_samples(
            StandardNormalPosterior(),
            torch.zeros(1, 2),
            torch.zeros(1, 4),
            num_chains=8,
            num_steps=800,
            burn_in_steps=200,
            thinning=2,
            step_size=0.05,
            init_jitter_scale=1.0,
            trace_interval=100,
            snapshot_steps=[400],
        )
        self.assertEqual(samples.shape, (8, 300, 4))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertGreater(diagnostics["acceptance_rate"], 0.8)
        self.assertEqual(diagnostics["invalid_proposal_fraction"], 0.0)
        self.assertEqual(len(diagnostics["split_rhat"]), 4)
        self.assertEqual(len(diagnostics["ess"]), 4)
        self.assertEqual(trace[-1]["step"], 800.0)
        self.assertLess(float(samples.mean(dim=(0, 1)).abs().max()), 0.35)

    def test_effective_sample_size_for_independent_draws(self) -> None:
        torch.manual_seed(9)
        samples = torch.randn(6, 500, 4, dtype=torch.float64)
        ess = classical_effective_sample_size(samples)
        self.assertEqual(ess.shape, (4,))
        self.assertTrue(torch.isfinite(ess).all())
        self.assertTrue((ess > 1000).all())
        self.assertTrue((ess <= 3000).all())

    def test_invalid_burn_in_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "burn_in_steps"):
            posterior_mala_samples(
                StandardNormalPosterior(),
                torch.zeros(1, 2),
                torch.zeros(1, 4),
                num_chains=4,
                num_steps=10,
                burn_in_steps=10,
                thinning=1,
                step_size=0.01,
                init_jitter_scale=1.0,
                trace_interval=1,
            )


if __name__ == "__main__":
    unittest.main()
