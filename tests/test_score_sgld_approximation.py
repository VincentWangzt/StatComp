from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from finalization.artifacts import RunRecord
from finalization.score_approximation import CellSpec, compute_score_metrics
from finalization.score_sgld_approximation import (
    SGLD_IMPLEMENTATION_VERSION,
    _assess_sgld_quality,
    load_sgld_score_config,
    posterior_sgld_group_scores,
    sgld_chunk_path,
    streamed_posterior_sgld_reference_scores,
)


class LinearGaussianVI(torch.nn.Module):
    """q(epsilon)=N(0,1), q(z|epsilon)=N(epsilon, variance)."""

    def __init__(self, conditional_variance: float = 0.5) -> None:
        super().__init__()
        self.conditional_variance = conditional_variance
        self.epsilon_dim = 1

    def log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        return -0.5 * epsilon.square().sum(dim=-1)

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        return -0.5 * (
            (z - epsilon).square() / self.conditional_variance
        ).sum(dim=-1)

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        return -(z - epsilon) / self.conditional_variance


class StopAfterCheckpoint(RuntimeError):
    pass


class ScoreSGLDApproximationTests(unittest.TestCase):

    def test_one_step_uses_repository_langevin_convention(self) -> None:
        model = LinearGaussianVI().to(dtype=torch.float64)
        z = torch.ones(1, 1, dtype=torch.float64)
        generating = torch.zeros_like(z)
        with patch(
            "finalization.score_sgld_approximation.torch.randn_like",
            side_effect=lambda value: torch.zeros_like(value),
        ):
            scores, diagnostics = posterior_sgld_group_scores(
                model,
                z,
                generating,
                num_groups=2,
                chains_per_group=3,
                num_steps=1,
                step_size=0.1,
                init_jitter_scale=0.0,
                finite_check_interval=1,
            )
        # grad log p(epsilon|z) at epsilon=0 is 2, hence the update is
        # epsilon=0 + 0.5*0.1*2 = 0.1 and the conditional score is -1.8.
        torch.testing.assert_close(
            scores,
            torch.full_like(scores, -1.8),
        )
        self.assertAlmostEqual(diagnostics["sgld_langevin_time"], 0.1)

    def test_group_scores_approximate_linear_gaussian_marginal_score(
        self,
    ) -> None:
        torch.manual_seed(321)
        model = LinearGaussianVI().to(dtype=torch.float64)
        z = torch.tensor([[-1.0], [0.0], [1.0]], dtype=torch.float64)
        posterior_variance = 1.0 / 3.0
        posterior_mean = z / 1.5
        generating = (
            posterior_mean
            + posterior_variance**0.5 * torch.randn_like(z)
        )
        scores, diagnostics = posterior_sgld_group_scores(
            model,
            z,
            generating,
            num_groups=4,
            chains_per_group=128,
            num_steps=2500,
            step_size=0.005,
            init_jitter_scale=1.0,
            diagnostic_steps=[1000, 2500],
            finite_check_interval=250,
        )
        expected = -z / 1.5
        torch.testing.assert_close(
            scores.mean(dim=0),
            expected,
            rtol=0.0,
            atol=0.09,
        )
        self.assertEqual(tuple(scores.shape), (4, 3, 1))
        self.assertEqual(
            diagnostics["sgld_total_terminal_particles_per_z"],
            512,
        )
        self.assertIn(
            "sgld_score_drift_step_1000_to_2500_l2",
            diagnostics,
        )

    def test_active_state_resume_preserves_exact_rng_stream(self) -> None:
        model = LinearGaussianVI().to(dtype=torch.float64)
        z = torch.tensor([[0.25]], dtype=torch.float64)
        generating = torch.tensor([[0.1]], dtype=torch.float64)
        kwargs = {
            "num_groups": 2,
            "chains_per_group": 4,
            "num_steps": 8,
            "step_size": 0.01,
            "init_jitter_scale": 1.0,
            "diagnostic_steps": [4, 8],
            "finite_check_interval": 2,
        }

        captured: dict[str, object] = {}

        def stop(state: dict[str, object]) -> None:
            captured.update(state)
            raise StopAfterCheckpoint

        torch.manual_seed(999)
        with self.assertRaises(StopAfterCheckpoint):
            posterior_sgld_group_scores(
                model,
                z,
                generating,
                checkpoint_interval=4,
                checkpoint_callback=stop,
                **kwargs,
            )
        resumed_scores, resumed_diagnostics = (
            posterior_sgld_group_scores(
                model,
                z,
                generating,
                resume_state=captured,
                **kwargs,
            )
        )

        torch.manual_seed(999)
        full_scores, full_diagnostics = posterior_sgld_group_scores(
            model,
            z,
            generating,
            **kwargs,
        )
        torch.testing.assert_close(
            resumed_scores,
            full_scores,
            rtol=0.0,
            atol=0.0,
        )
        self.assertEqual(
            resumed_diagnostics[
                "sgld_score_drift_step_4_to_8_l2"
            ],
            full_diagnostics["sgld_score_drift_step_4_to_8_l2"],
        )

    def test_rejects_single_group_for_internal_l2(self) -> None:
        model = LinearGaussianVI()
        z = torch.zeros(1, 1)
        with self.assertRaisesRegex(ValueError, "at least two groups"):
            posterior_sgld_group_scores(
                model,
                z,
                z.clone(),
                num_groups=1,
                chains_per_group=2,
                num_steps=2,
                step_size=0.01,
                init_jitter_scale=1.0,
            )

    def test_quality_uses_latest_preterminal_snapshot(self) -> None:
        diagnostics = {
            "sgld_terminal_nonfinite_fraction": 0.0,
            "sgld_score_drift_step_5000_to_20000_l2": 100.0,
            "sgld_score_drift_step_10000_to_20000_l2": 0.2,
        }
        metrics = {"reference_mean_mcse_l2": 0.1}
        status, issues = _assess_sgld_quality(
            diagnostics,
            metrics,
            OmegaConf.create({
                "max_nonfinite_fraction": 0.0,
                "max_horizon_drift_to_mcse": 4.0,
            }),
        )
        self.assertEqual(status, "pass")
        self.assertEqual(issues, [])
        self.assertEqual(
            diagnostics["sgld_latest_horizon_drift_l2"],
            0.2,
        )

    def test_production_config_locks_requested_budget(self) -> None:
        cfg = load_sgld_score_config(None)
        reference = cfg.evaluation.reference
        self.assertEqual(list(cfg.selection.methods), ["DSIVI"])
        self.assertEqual(
            list(cfg.selection.targets),
            ["x_shaped", "8_gaussians"],
        )
        self.assertEqual(
            list(cfg.selection.seeds),
            [42, 43, 45, 49, 50],
        )
        self.assertEqual(list(cfg.selection.checkpoint_progress), [1.0])
        self.assertEqual(int(cfg.evaluation.forward_batch_size), 1024)
        self.assertEqual(int(reference.num_groups), 10)
        self.assertEqual(int(reference.chains_per_group), 1000)
        self.assertEqual(int(reference.num_steps), 5000)
        self.assertEqual(float(reference.step_size), 0.0001)
        self.assertEqual(float(reference.init_jitter_scale), 1.0)
        self.assertEqual(int(reference.z_chunk_size), 800)
        self.assertEqual(
            str(reference.implementation_version),
            SGLD_IMPLEMENTATION_VERSION,
        )

    def test_group_replicates_give_requested_internal_l2_and_mcse(
        self,
    ) -> None:
        references = torch.tensor([
            [[0.0], [0.0]],
            [[2.0], [2.0]],
            [[0.0], [2.0]],
            [[2.0], [0.0]],
        ])
        metrics = compute_score_metrics(
            torch.ones(2, 1),
            references,
        )
        self.assertEqual(metrics["method_l2"], 0.0)
        self.assertEqual(metrics["reference_internal_l2"], 1.0)
        self.assertAlmostEqual(
            metrics["reference_mean_mcse_l2"],
            1.0 / 3.0,
        )

    def test_streamed_chunks_resume_and_no_resume_recomputes(self) -> None:
        model = LinearGaussianVI().to(dtype=torch.float64)
        z = torch.tensor([[-0.5], [0.5]], dtype=torch.float64)
        generating = torch.tensor([[-0.25], [0.25]], dtype=torch.float64)
        record = RunRecord(
            run_id="run-1",
            seed=42,
            method="DSIVI",
            target="x_shaped",
            runner_type="DSIVI",
            config_path=Path("config.yaml"),
            result_path=Path("results"),
            duration_sec=None,
            status="completed",
            entry={},
        )
        spec = CellSpec(
            record=record,
            progress=1.0,
            epoch=10000,
            checkpoint_dir=Path("checkpoint"),
        )
        kwargs = {
            "spec": spec,
            "fingerprint": "abc",
            "reference_seed": 123,
            "num_groups": 2,
            "chains_per_group": 3,
            "num_steps": 3,
            "step_size": 0.01,
            "init_jitter_scale": 1.0,
            "z_chunk_size": 1,
            "diagnostic_steps": [3],
            "finite_check_interval": 1,
            "checkpoint_interval": 0,
            "accumulator_dtype": torch.float64,
            "implementation_version": SGLD_IMPLEMENTATION_VERSION,
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            run_root = Path(temp_dir)
            first, _ = streamed_posterior_sgld_reference_scores(
                model,
                z,
                generating,
                run_root=run_root,
                **kwargs,
            )
            with patch(
                "finalization.score_sgld_approximation."
                "posterior_sgld_group_scores",
                side_effect=AssertionError("completed chunks were recomputed"),
            ):
                resumed, _ = streamed_posterior_sgld_reference_scores(
                    model,
                    z,
                    generating,
                    run_root=run_root,
                    resume=True,
                    **kwargs,
                )
            torch.testing.assert_close(first, resumed)

            corrupt_path = sgld_chunk_path(run_root, spec, 0, 1)
            corrupt = torch.load(
                corrupt_path,
                map_location="cpu",
                weights_only=False,
            )
            corrupt["group_scores"][0, 0, 0] = torch.nan
            torch.save(corrupt, corrupt_path)
            with self.assertRaisesRegex(RuntimeError, "non-finite"):
                streamed_posterior_sgld_reference_scores(
                    model,
                    z,
                    generating,
                    run_root=run_root,
                    resume=True,
                    **kwargs,
                )

            with patch(
                "finalization.score_sgld_approximation."
                "posterior_sgld_group_scores",
                wraps=posterior_sgld_group_scores,
            ) as recompute:
                fresh, _ = streamed_posterior_sgld_reference_scores(
                    model,
                    z,
                    generating,
                    run_root=run_root,
                    resume=False,
                    **kwargs,
                )
            self.assertEqual(recompute.call_count, 2)
            torch.testing.assert_close(first, fresh)


if __name__ == "__main__":
    unittest.main()
