from __future__ import annotations

import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.annealing import annealing
from utils.kernels import (
    BaseKernel,
    ComponentAdaptiveLaplaceL2Kernel,
    GaussianKernel,
)
from utils.mcmc_kernels import mala_transition
from utils.mmd import configure_kernel_bandwidth, mmd2_v_statistic


REPO_ROOT = Path(__file__).resolve().parents[1]


class RecordingKernel(BaseKernel):
    def __init__(self, h: float = 1.0):
        super().__init__(h=h, name="RecordingKernel")
        self.fit_samples: torch.Tensor | None = None

    def fit_h(self, samples: torch.Tensor) -> float:
        self.fit_samples = samples.clone()
        self.h = 1.0
        return self.h

    def pair_eval(
        self,
        samples_x: torch.Tensor,
        samples_y: torch.Tensor | None = None,
        fit_h: bool = False,
        detach_h: bool = True,
    ) -> torch.Tensor:
        del fit_h, detach_h
        if samples_y is None:
            samples_y = samples_x
        distances = torch.cdist(samples_x, samples_y)
        return torch.exp(-distances / self.h)


class KDVIBandwidthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.x = torch.tensor([[0.0, 0.0], [1.0, 0.0]], requires_grad=True)
        self.y = torch.tensor([[0.0, 1.0], [1.0, 1.0]])

    def test_x_bandwidth_fits_only_variational_samples(self) -> None:
        kernel = RecordingKernel()

        loss, _ = mmd2_v_statistic(self.x, self.y, kernel, "x")

        self.assertTrue(torch.equal(kernel.fit_samples, self.x.detach()))
        loss.backward()
        self.assertIsNotNone(self.x.grad)

    def test_xy_bandwidth_fits_pooled_samples(self) -> None:
        kernel = RecordingKernel()

        mmd2_v_statistic(self.x, self.y, kernel, "xy")

        expected = torch.cat([self.x.detach(), self.y], dim=0)
        self.assertTrue(torch.equal(kernel.fit_samples, expected))

    def test_fixed_bandwidth_takes_precedence_over_fit_source(self) -> None:
        kernel = RecordingKernel(h=-1.0)

        fit_source = configure_kernel_bandwidth(
            kernel,
            fit_bandwidth_on="y",
            kernel_bandwidth=0.75,
        )
        mmd2_v_statistic(self.x, self.y, kernel, fit_source)

        self.assertIsNone(fit_source)
        self.assertEqual(kernel.h, 0.75)
        self.assertIsNone(kernel.fit_samples)

    def test_fixed_bandwidth_must_be_positive(self) -> None:
        for invalid in (0.0, -0.5):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "must be positive"):
                    configure_kernel_bandwidth(
                        GaussianKernel(), kernel_bandwidth=invalid)

    def test_removed_adaptive_modes_are_rejected(self) -> None:
        for removed in ("y", "ivi", "none"):
            with self.subTest(removed=removed):
                with self.assertRaisesRegex(ValueError, "must be 'x' or 'xy'"):
                    configure_kernel_bandwidth(
                        GaussianKernel(), fit_bandwidth_on=removed)


    def test_component_adaptive_laplace_l2_fits_coordinate_bandwidths(self) -> None:
        samples = torch.tensor([
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
        ])
        kernel = ComponentAdaptiveLaplaceL2Kernel()

        kernel.fit_h(samples)
        values = kernel.pair_eval(samples, samples)

        self.assertTrue(torch.allclose(kernel._h_vec, torch.tensor([1.0, 10.0])))
        self.assertAlmostEqual(kernel.h, 5.5)
        self.assertEqual(tuple(values.shape), (3, 3))
        self.assertTrue(torch.isfinite(values).all())

    def test_component_adaptive_laplace_l2_respects_fixed_bandwidth(self) -> None:
        samples = torch.tensor([
            [0.0, 0.0],
            [1.0, 10.0],
            [2.0, 20.0],
        ])
        kernel = ComponentAdaptiveLaplaceL2Kernel()

        fit_source = configure_kernel_bandwidth(
            kernel,
            fit_bandwidth_on="x",
            kernel_bandwidth=2.0,
        )
        values = kernel.pair_eval(samples, samples)

        self.assertIsNone(fit_source)
        self.assertEqual(kernel.h, 2.0)
        self.assertIsNone(kernel._h_vec)
        self.assertEqual(tuple(values.shape), (3, 3))


class KDVIScheduleAndMALATests(unittest.TestCase):
    def test_offset_linear_annealing(self) -> None:
        values = [
            annealing(t, warm_up_interval=100, scheme="offset_linear", anneal=True)
            for t in (0, 45, 90, 150)
        ]
        self.assertEqual(values[0], 0.1)
        self.assertAlmostEqual(values[1], 0.55)
        self.assertEqual(values[2], 1.0)
        self.assertEqual(values[3], 1.0)

    def test_old_ivi_annealing_name_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown annealing scheme"):
            annealing(1, warm_up_interval=100, scheme="ivi", anneal=True)

    def test_mala_uses_supplied_analytic_score(self) -> None:
        score_calls = 0

        def score_fn(z: torch.Tensor) -> torch.Tensor:
            nonlocal score_calls
            score_calls += 1
            self.assertFalse(torch.is_grad_enabled())
            return -z

        def log_prob_fn(z: torch.Tensor) -> torch.Tensor:
            # If MALA tried to obtain its score through autograd, this callback
            # would run under enable_grad and fail the assertion.
            self.assertFalse(torch.is_grad_enabled())
            return -0.5 * (z ** 2).sum(dim=-1)

        torch.manual_seed(7)
        output = mala_transition(
            z_init=torch.zeros(8, 2),
            log_prob_fn=log_prob_fn,
            score_fn=score_fn,
            step_size=0.01,
            n_steps=2,
        )

        self.assertEqual(score_calls, 4)
        self.assertEqual(tuple(output.z.shape), (8, 2))
        self.assertTrue(torch.isfinite(output.z).all())


class KDVIConfigSmokeTests(unittest.TestCase):
    def test_all_active_kdvi_configs_use_clean_interface(self) -> None:
        config_paths = sorted((REPO_ROOT / "configs").glob("kdvi_*.yaml"))
        self.assertTrue(config_paths)

        for config_path in config_paths:
            with self.subTest(config=config_path.name):
                config = OmegaConf.load(config_path)
                self.assertEqual(config.runner_type, "KDVI")
                self.assertNotIn("parity_rng_isolation", config.train)
                self.assertNotEqual(config.train.annealing.scheme, "ivi")

                kdvi = config.train.kdvi
                self.assertNotIn("loss_form", kdvi)
                self.assertIn(kdvi.get("fit_bandwidth_on", "x"), ("x", "xy"))

                fixed = kdvi.get("kernel_bandwidth", None)
                if fixed is not None:
                    self.assertGreater(float(fixed), 0.0)

                vi_config_path = config.get("vi_model_config_path", None)
                if vi_config_path is not None:
                    self.assertTrue((REPO_ROOT / vi_config_path).is_file())


if __name__ == "__main__":
    unittest.main()
