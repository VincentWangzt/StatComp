from __future__ import annotations

import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from utils.annealing import annealing
from utils.kernels import (
    BaseKernel,
    GaussianKernel,
    GaussianKernelMMD,
    IMQKernel,
    LaplaceL2Kernel,
)
from utils.mcmc_kernels import (
    mala_transition,
    sgld_transition_differentiable,
)
from utils.mmd import (
    configure_kernel_bandwidth,
    mmd2_v_statistic,
    mmd2_v_statistic_per_dim,
    paired_l2_loss,
    sliced_w2_loss,
)


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


class KDVILossTypeTests(unittest.TestCase):
    def test_mmd_can_backpropagate_through_refined_samples(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [1.0, 0.0]],
            requires_grad=True,
        )
        y = torch.tensor(
            [[0.0, 1.0], [1.0, 1.5]],
            requires_grad=True,
        )

        loss, _ = mmd2_v_statistic(x, y, GaussianKernelMMD(h=1.0), None)

        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNotNone(y.grad)
        self.assertTrue(torch.isfinite(y.grad).all())
        self.assertGreater(y.grad.abs().sum().item(), 0.0)

    def test_paired_l2_value_and_gradient(self) -> None:
        x = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        y = torch.tensor(
            [[2.0, 0.0], [1.0, 5.0]],
            requires_grad=True,
        )

        loss, info = paired_l2_loss(x, y)

        expected = torch.tensor((1.0 + 4.0 + 4.0 + 1.0) / 2.0)
        self.assertTrue(torch.allclose(loss, expected))
        self.assertAlmostEqual(info["paired_l2_mean"], expected.item())
        loss.backward()
        self.assertTrue(torch.allclose(x.grad, (2.0 / 2.0) * (x - y.detach())))
        self.assertIsNone(y.grad)

    def test_sliced_w2_value_and_gradient(self) -> None:
        x = torch.tensor(
            [[0.0], [2.0], [4.0]],
            requires_grad=True,
        )
        y = torch.tensor(
            [[1.0], [3.0], [5.0]],
            requires_grad=True,
        )

        torch.manual_seed(3)
        loss, info = sliced_w2_loss(x, y, num_projections=4)

        self.assertTrue(torch.allclose(loss, torch.tensor(1.0)))
        self.assertAlmostEqual(info["sliced_w2"], 1.0)
        self.assertEqual(info["sliced_w2_num_projections"], 4.0)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertIsNone(y.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0.0)

    def test_per_dim_gaussian_mmd_matches_hand_computed_kernel(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]],
            requires_grad=True,
        )
        y = torch.tensor([[1.0, 1.5], [3.0, 4.5], [5.0, 7.5]])
        kernel = GaussianKernelMMD()

        loss, info = mmd2_v_statistic_per_dim(x, y, kernel, "x")

        bandwidths = torch.tensor([2.0, 3.0])

        def pair_eval(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            scaled = (a[:, None, :] - b[None, :, :]) / bandwidths
            return torch.exp(-0.5 * scaled.square().sum(dim=-1))

        expected = (
            pair_eval(x, x).mean()
            + pair_eval(y, y).mean()
            - 2.0 * pair_eval(x, y).mean()
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertAlmostEqual(info["kernel_bandwidth_mean"], 2.5)
        self.assertAlmostEqual(info["kernel_bandwidth_min"], 2.0)
        self.assertAlmostEqual(info["kernel_bandwidth_max"], 3.0)
        loss.backward()
        self.assertIsNotNone(x.grad)

    def test_per_dim_laplace_l2_matches_hand_computed_kernel(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]],
            requires_grad=True,
        )
        y = torch.tensor([[1.0, 1.5], [3.0, 4.5], [5.0, 7.5]])
        kernel = LaplaceL2Kernel()

        loss, info = mmd2_v_statistic_per_dim(x, y, kernel, "x")

        bandwidths = torch.tensor([2.0, 3.0])

        def pair_eval(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            scaled = (a[:, None, :] - b[None, :, :]) / bandwidths
            return torch.exp(-0.5 * scaled.square().sum(dim=-1).sqrt())

        expected = (
            pair_eval(x, x).mean()
            + pair_eval(y, y).mean()
            - 2.0 * pair_eval(x, y).mean()
        )
        self.assertTrue(torch.allclose(loss, expected))
        self.assertAlmostEqual(info["kernel_bandwidth_mean"], 2.5)
        loss.backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())

    def test_per_dim_bandwidth_xy_uses_pooled_samples(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]],
            requires_grad=True,
        )
        y = torch.tensor([[100.0, 0.0], [102.0, 3.0], [104.0, 6.0]])

        _, info_x = mmd2_v_statistic_per_dim(x, y, GaussianKernelMMD(), "x")
        _, info_xy = mmd2_v_statistic_per_dim(
            x, y, GaussianKernelMMD(), "xy")

        self.assertAlmostEqual(info_x["kernel_bandwidth_mean"], 2.5)
        self.assertGreater(info_xy["kernel_bandwidth_mean"], 2.5)

    def test_per_dim_fixed_scalar_bandwidth_broadcasts(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]],
            requires_grad=True,
        )
        y = torch.tensor([[1.0, 1.5], [3.0, 4.5], [5.0, 7.5]])
        kernel = GaussianKernelMMD()
        fit_source = configure_kernel_bandwidth(
            kernel,
            fit_bandwidth_on="x",
            kernel_bandwidth=0.75,
        )

        _, info = mmd2_v_statistic_per_dim(x, y, kernel, fit_source)

        self.assertIsNone(fit_source)
        self.assertAlmostEqual(info["kernel_bandwidth_mean"], 0.75)
        self.assertAlmostEqual(info["kernel_bandwidth_min"], 0.75)
        self.assertAlmostEqual(info["kernel_bandwidth_max"], 0.75)

    def test_per_dim_rejects_unsupported_kernels(self) -> None:
        x = torch.tensor(
            [[0.0, 0.0], [2.0, 3.0], [4.0, 6.0]],
            requires_grad=True,
        )
        y = torch.tensor([[1.0, 1.5], [3.0, 4.5], [5.0, 7.5]])

        with self.assertRaisesRegex(ValueError, "supports only"):
            mmd2_v_statistic_per_dim(x, y, IMQKernel(), "x")


class KDVIScheduleAndMALATests(unittest.TestCase):
    def test_differentiable_sgld_backpropagates_to_initial_particles(self) -> None:
        z_init = torch.tensor(
            [[1.0, -2.0], [0.5, 3.0]],
            requires_grad=True,
        )
        step_size = 0.2
        n_steps = 2

        torch.manual_seed(11)
        output = sgld_transition_differentiable(
            z_init=z_init,
            score_fn=lambda z: -z,
            step_size=step_size,
            n_steps=n_steps,
        )

        self.assertTrue(output.z.requires_grad)
        output.z.sum().backward()
        expected_factor = (1.0 - 0.5 * step_size) ** n_steps
        self.assertTrue(
            torch.allclose(
                z_init.grad,
                torch.full_like(z_init, expected_factor),
            )
        )

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
                loss_type = kdvi.get("loss_type", "mmd")
                self.assertIn(
                    loss_type,
                    (
                        "mmd",
                        "paired_l2",
                        "mmd_per_dim",
                        "mmd_no_detach",
                        "sliced_w2",
                    ),
                )
                self.assertEqual(loss_type, "mmd")
                self.assertIn(kdvi.get("fit_bandwidth_on", "x"), ("x", "xy"))

                fixed = kdvi.get("kernel_bandwidth", None)
                if fixed is not None:
                    self.assertGreater(float(fixed), 0.0)

                vi_config_path = config.get("vi_model_config_path", None)
                if vi_config_path is not None:
                    self.assertTrue((REPO_ROOT / vi_config_path).is_file())


if __name__ == "__main__":
    unittest.main()
