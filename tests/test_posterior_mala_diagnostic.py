from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from finalization.posterior_mala_diagnostic import (
    _write_report,
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

    def test_two_dimensional_report_uses_raw_epsilon_projection(self) -> None:
        torch.manual_seed(11)
        samples = torch.randn(4, 20, 2, dtype=torch.float64)
        diagnostics = {
            "initial_chain_epsilon": torch.randn(4, 2).tolist(),
            "generating_epsilon": [0.1, -0.2],
            "split_rhat": [1.01, 1.02],
            "split_rhat_max": 1.02,
            "ess_min": 50.0,
            "early_late_standardized_drift_max": 0.2,
            "acceptance_rate": 0.95,
            "post_burn_acceptance_rate": 0.96,
            "invalid_proposal_fraction": 0.0,
            "convergence_pass": False,
        }
        trace = [
            {
                "step": 0.0,
                "acceptance_rate": 0.0,
                "window_acceptance_rate": 0.0,
                "mean_log_posterior": -2.0,
                "epsilon_mean_norm": 0.5,
                "epsilon_sd_mean": 1.0,
            },
            {
                "step": 10.0,
                "acceptance_rate": 0.95,
                "window_acceptance_rate": 0.95,
                "mean_log_posterior": -1.5,
                "epsilon_mean_norm": 0.2,
                "epsilon_sd_mean": 0.9,
            },
        ]
        metadata = {
            "method": "DSIVI",
            "target": "x_shaped",
            "seed": 45,
            "epoch": 10000,
            "checkpoint_dir": "checkpoint",
            "epsilon_dim": 2,
            "z_dim": 2,
            "z": [0.3, -0.4],
            "forward_seed": 1,
            "sampler_seed": 2,
            "gpu_name": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            _write_report(
                report_dir,
                samples,
                diagnostics,
                trace,
                metadata=metadata,
                max_plot_samples=80,
                max_csv_samples=80,
            )
            self.assertEqual(
                diagnostics["projection_kind"],
                "raw_epsilon",
            )
            self.assertTrue(
                (report_dir / "posterior_epsilon_diagnostic.png").is_file()
            )
            self.assertTrue(
                (report_dir / "posterior_mala_metrics.json").is_file()
            )


if __name__ == "__main__":
    unittest.main()
