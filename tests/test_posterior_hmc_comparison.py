from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from finalization.posterior_hmc_comparison import (
    _stratified_indices,
    _write_comparison_report,
    posterior_hmc_samples,
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


class PosteriorHmcComparisonTests(unittest.TestCase):
    def test_hmc_samples_standard_normal(self) -> None:
        torch.manual_seed(17)
        samples, diagnostics, trace = posterior_hmc_samples(
            StandardNormalPosterior(),
            torch.zeros(1, 2),
            torch.zeros(1, 2),
            num_chains=8,
            burn_in_steps=100,
            samples_per_chain=200,
            thinning=1,
            step_size=0.08,
            leapfrog_steps=5,
            init_jitter_scale=1.0,
            adapt_step_size=False,
            target_acceptance=0.9,
            adaptation_rate=0.0,
            min_step_size=0.01,
            max_step_size=0.2,
            divergence_threshold=1000.0,
            trace_interval=50,
        )
        self.assertEqual(samples.shape, (8, 200, 2))
        self.assertTrue(torch.isfinite(samples).all())
        self.assertGreater(diagnostics["acceptance_rate"], 0.9)
        self.assertEqual(diagnostics["divergence_fraction"], 0.0)
        self.assertEqual(len(diagnostics["split_rhat"]), 2)
        self.assertEqual(len(diagnostics["ess"]), 2)
        self.assertEqual(trace[-1]["transition"], 300.0)
        self.assertLess(float(samples.mean(dim=(0, 1)).abs().max()), 0.2)

    def test_stratified_indices_represent_every_chain(self) -> None:
        indices = _stratified_indices(
            chains=5,
            draws=100,
            max_samples=20,
        )
        self.assertEqual(len(indices), 20)
        self.assertEqual({chain for chain, _ in indices}, set(range(5)))
        for chain in range(5):
            draws = [draw for value, draw in indices if value == chain]
            self.assertEqual(draws[0], 0)
            self.assertEqual(draws[-1], 99)

    def test_two_dimensional_comparison_report(self) -> None:
        torch.manual_seed(21)
        mala_samples = torch.randn(4, 20, 2, dtype=torch.float64)
        hmc_samples = torch.randn(3, 30, 2, dtype=torch.float64)
        mala_diagnostics = {
            "acceptance_rate": 0.95,
            "split_rhat_max": 2.0,
            "ess_min": 25.0,
            "convergence_pass": False,
        }
        hmc_diagnostics = {
            "samples_per_chain": 30,
            "acceptance_rate": 0.9,
            "split_rhat": [1.01, 1.02],
            "split_rhat_max": 1.02,
            "ess": [100.0, 110.0],
            "ess_min": 100.0,
            "convergence_pass": False,
            "initial_chain_epsilon": torch.randn(3, 2).tolist(),
        }
        hmc_trace = [
            {
                "transition": 0.0,
                "acceptance_rate": 0.0,
                "window_acceptance_rate": 0.0,
                "mean_log_posterior": -2.0,
                "step_size_median": 0.01,
                "epsilon_mean_norm": 1.0,
                "epsilon_sd_mean": 1.0,
            },
            {
                "transition": 30.0,
                "acceptance_rate": 0.9,
                "window_acceptance_rate": 0.9,
                "mean_log_posterior": -1.0,
                "step_size_median": 0.02,
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
            "forward_seed": 1,
            "hmc_sampler_seed": 2,
            "gpu_name": "test",
        }
        with tempfile.TemporaryDirectory() as directory:
            report_dir = Path(directory)
            _write_comparison_report(
                report_dir,
                mala_samples=mala_samples,
                mala_diagnostics=mala_diagnostics,
                hmc_samples=hmc_samples,
                hmc_diagnostics=hmc_diagnostics,
                hmc_trace=hmc_trace,
                generating_epsilon=torch.tensor(
                    [[0.1, -0.2]],
                    dtype=torch.float64,
                ),
                z=torch.tensor([[0.3, -0.4]], dtype=torch.float64),
                metadata=metadata,
                mala_tail_draws_per_chain=5,
                max_plot_samples=100,
                max_csv_samples=100,
            )
            self.assertTrue(
                (report_dir / "posterior_mala_hmc_comparison.png").is_file()
            )
            self.assertTrue(
                (report_dir / "posterior_mala_hmc_metrics.json").is_file()
            )
            self.assertTrue(
                (report_dir / "posterior_mala_hmc_samples.csv").is_file()
            )


if __name__ == "__main__":
    unittest.main()
