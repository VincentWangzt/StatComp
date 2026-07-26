import unittest
from argparse import Namespace
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from omegaconf import OmegaConf

from models.vi_model import RealNVP
from scripts.run_nfvi_rebuttal import (
    PROJECT_ROOT,
    Variant,
    configure_run,
    write_summary,
)


class RealNVPTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config = OmegaConf.create({
            "z_dim": 2,
            "epsilon_dim": 2,
            "hidden_dim": 16,
            "num_hidden_layers": 2,
            "num_flow_layers": 4,
            "activation": "silu",
            "scale_clip": 3.0,
            "base_trainable": True,
            "device": "cpu",
        })

    def test_sampling_and_exact_density_are_consistent(self) -> None:
        torch.manual_seed(7)
        model = RealNVP(self.config)
        epsilon = model.sample_epsilon(32)
        z, sampled_log_q = model.forward_and_log_prob(epsilon)
        evaluated_log_q = model.logp(z)

        self.assertEqual(z.shape, (32, 2))
        self.assertEqual(sampled_log_q.shape, (32,))
        torch.testing.assert_close(
            sampled_log_q,
            evaluated_log_q,
            rtol=1e-5,
            atol=1e-5,
        )

    def test_reverse_kl_path_has_finite_gradients(self) -> None:
        torch.manual_seed(11)
        model = RealNVP(self.config)
        epsilon = model.sample_epsilon(16)
        z, log_q = model.forward_and_log_prob(epsilon)
        log_p = -0.5 * (z**2).sum(dim=-1)
        loss = (log_q - log_p).mean()
        loss.backward()

        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad
        ]
        self.assertTrue(all(gradient is not None for gradient in gradients))
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))

    def test_log_density_preserves_leading_dimensions(self) -> None:
        model = RealNVP(self.config)
        z = torch.randn(3, 5, 2)
        self.assertEqual(model.logp(z).shape, (3, 5))


class RebuttalBenchmarkConfigTest(unittest.TestCase):

    def test_metric_frequency_suppresses_evaluation_without_using_zero(self) -> None:
        args = Namespace(
            epochs=100,
            w2_samples=10,
            w2_projections=2,
            elbo_samples=10,
            mode_samples=10,
            output_dir=Path("results/test_nfvi_rebuttal"),
        )
        variant = Variant(
            "DIVI",
            PROJECT_ROOT / "configs/dsivi_8_gaussians.yaml",
        )
        config = configure_run(variant, 42, args, torch.device("cpu"))

        self.assertEqual(config.train.log.metric_log_freq, 101)

    def test_compact_csv_reports_use_unix_newlines(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "summary.csv"
            write_summary(path, [{"method": "NFVI-4", "n_seeds": 1}])

            self.assertNotIn(b"\r\n", path.read_bytes())


if __name__ == "__main__":
    unittest.main()
