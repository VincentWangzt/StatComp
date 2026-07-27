from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from finalization.artifacts import RunRecord
from finalization.score_approximation import CellSpec
from finalization.score_jitter_ablation import (
    _render_plot,
    evaluate_seed,
    load_jitter_config,
    pairwise_reference_l2,
    summarize_jitter_rows,
    summarize_pairwise_rows,
)


class ScoreJitterAblationTest(unittest.TestCase):

    def test_evaluate_seed_accepts_string_runner_device(self) -> None:
        cfg = load_jitter_config(
            None,
            [
                "evaluation.device=cpu",
                "evaluation.forward_batch_size=2",
                "evaluation.reference.total_samples=4",
                "evaluation.reference.num_chains=2",
            ],
        )
        vi_model = SimpleNamespace(
            sampling=lambda num: (
                torch.zeros(num, 1),
                torch.zeros(num, 2),
            ),
        )
        runner = SimpleNamespace(
            device="cpu",
            vi_model=vi_model,
            target_model=SimpleNamespace(
                score=lambda z: torch.zeros_like(z),
            ),
        )
        record = RunRecord(
            run_id="run-1",
            seed=42,
            method="DSIVI",
            target="8_gaussians",
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

        def fake_reference(
            _model: object,
            z: torch.Tensor,
            _epsilon: torch.Tensor,
            **_kwargs: object,
        ) -> tuple[torch.Tensor, dict[str, float]]:
            return (
                torch.zeros(
                    2,
                    z.shape[0],
                    z.shape[1],
                    dtype=torch.float64,
                ),
                {},
            )

        with (
            patch(
                "finalization.score_approximation._load_checkpoint",
            ),
            patch(
                "finalization.score_approximation.method_native_score",
                return_value=(torch.zeros(2, 2), {}),
            ),
            patch(
                "finalization.score_approximation."
                "posterior_hmc_reference_scores",
                side_effect=fake_reference,
            ),
            patch(
                "finalization.score_approximation."
                "assess_hmc_reference_quality",
                return_value=("pass", []),
            ),
        ):
            result = evaluate_seed(
                runner,
                spec,
                cfg,
                fingerprint="fingerprint",
            )
        self.assertEqual(len(result["jitter_metrics"]), 4)
        self.assertEqual(len(result["pairwise_reference_l2"]), 6)

    def test_pairwise_reference_l2_uses_per_sample_vector_loss(self) -> None:
        references = {
            0.0: torch.zeros(2, 2, dtype=torch.float64),
            0.1: torch.eye(2, dtype=torch.float64),
        }
        [row] = pairwise_reference_l2(references)
        self.assertEqual(row["jitter_a"], 0.0)
        self.assertEqual(row["jitter_b"], 0.1)
        self.assertAlmostEqual(row["reference_mean_l2"], 1.0)
        self.assertAlmostEqual(row["reference_mean_rms"], 1.0)
        self.assertAlmostEqual(
            row["relative_reference_mean_l2"],
            2.0,
        )

    def test_three_seed_summaries_use_sample_standard_deviation(self) -> None:
        jitter_rows = []
        pairwise_rows = []
        for seed_index, value in enumerate((1.0, 2.0, 3.0)):
            for jitter_scale in (0.0, 0.01):
                row = {
                    "jitter_scale": jitter_scale,
                    "reference_quality_status": "pass",
                }
                for metric in (
                    "method_l2",
                    "method_relative_l2",
                    "method_target_l2",
                    "reference_target_l2",
                    "reference_internal_l2",
                    "reference_mean_mcse_l2",
                    "reference_mean_score_sq_norm",
                    "reference_runtime_sec",
                    "diagnostic_hmc_score_rhat_p95",
                    "diagnostic_hmc_epsilon_rhat_p95",
                    "diagnostic_hmc_post_burn_acceptance_rate",
                    "diagnostic_hmc_post_burn_acceptance_min",
                    "diagnostic_hmc_divergence_fraction",
                    "diagnostic_hmc_final_step_size_median",
                    "diagnostic_hmc_mean_squared_jump_distance",
                ):
                    row[metric] = value + jitter_scale
                jitter_rows.append(row)
            pairwise_rows.append({
                "seed": 42 + seed_index,
                "jitter_a": 0.0,
                "jitter_b": 0.01,
                "reference_mean_l2": value,
                "reference_mean_rms": value,
                "relative_reference_mean_l2": value,
            })

        jitter_summary = summarize_jitter_rows(jitter_rows)
        pairwise_summary = summarize_pairwise_rows(pairwise_rows)
        self.assertEqual(len(jitter_summary), 2)
        self.assertEqual(jitter_summary[0]["n_seeds"], 3)
        self.assertAlmostEqual(
            jitter_summary[0]["method_l2_mean"],
            2.0,
        )
        self.assertAlmostEqual(
            jitter_summary[0]["method_l2_sd"],
            1.0,
        )
        self.assertEqual(len(pairwise_summary), 1)
        self.assertAlmostEqual(
            pairwise_summary[0]["reference_mean_l2_mean"],
            2.0,
        )
        self.assertAlmostEqual(
            pairwise_summary[0]["reference_mean_l2_sd"],
            1.0,
        )

    def test_ablation_plot_renders(self) -> None:
        jitter_summary = []
        for index, jitter_scale in enumerate(
            (0.0, 0.0001, 0.001, 0.01),
            start=1,
        ):
            row = {"jitter_scale": jitter_scale}
            for metric in (
                "method_l2",
                "reference_internal_l2",
                "diagnostic_hmc_score_rhat_p95",
            ):
                row[f"{metric}_mean"] = float(index)
                row[f"{metric}_sd"] = float(index) / 10.0
            jitter_summary.append(row)
        pairwise_summary = [
            {
                "jitter_a": 0.0,
                "jitter_b": jitter_scale,
                "reference_mean_l2_mean": float(index),
                "reference_mean_l2_sd": float(index) / 10.0,
            }
            for index, jitter_scale in enumerate(
                (0.0001, 0.001, 0.01),
                start=1,
            )
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            paths = _render_plot(
                Path(temp_dir),
                jitter_summary,
                pairwise_summary,
            )
            self.assertEqual(len(paths), 2)
            self.assertTrue(
                all(path.stat().st_size > 0 for path in paths)
            )


if __name__ == "__main__":
    unittest.main()
