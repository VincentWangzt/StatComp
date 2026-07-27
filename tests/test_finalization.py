from __future__ import annotations

import unittest
from pathlib import Path

import torch
from finalization.artifacts import RunRecord, normalize_target
from omegaconf import OmegaConf

from finalization.plots import langevin_panel_labels
from finalization.runner_eval import _append_langevin_sgld_if_needed, constrained_w2, prepare_config, summarize, truncated_w2_metric_name, warning_rows_from_run_rows
from finalization.tables import render_bnn_table, render_langevin_table, render_toy_method_grid


class FinalizationTests(unittest.TestCase):
    def test_target_aliases(self) -> None:
        self.assertEqual(normalize_target("multi_model"), "multimodal")
        self.assertEqual(normalize_target("8_gaussian"), "8_gaussians")
        self.assertEqual(normalize_target("banana"), "banana")

    def test_langevin_panel_labels_place_sgld_first_and_dsivi_bottom_left(self) -> None:
        self.assertEqual(
            langevin_panel_labels(["SIVI", "UIVI", "AISIVI", "DSIVI", "KSIVI"]),
            ["SGLD", "SIVI", "UIVI", "DSIVI", "AISIVI", "KSIVI"],
        )
        self.assertEqual(
            langevin_panel_labels(["SIVI", "UIVI", "AISIVI", "DSIVI"], {"UIVI", "DSIVI"}),
            ["SGLD", "UIVI", "DSIVI"],
        )

    def test_truncated_w2_metric_names_use_absolute_width(self) -> None:
        self.assertEqual(truncated_w2_metric_name(6), "w2_trunc_abs_6")
        self.assertEqual(truncated_w2_metric_name(8.0), "w2_trunc_abs_8")
        self.assertEqual(truncated_w2_metric_name(6.5), "w2_trunc_abs_6_5")

    def test_constrained_w2_falls_back_to_edge_length_when_max_draws_is_reached(self) -> None:
        class FakeVI:
            def sampling(self, num: int):
                return None, torch.full((num, 2), 2.0)

        class FakeTarget:
            def sample(self, num: int):
                return torch.zeros((num, 2))

        class FakeRunner:
            vi_model = FakeVI()
            target_model = FakeTarget()
            target_type = "fake"
            device = "cpu"

        warnings: list[str] = []
        cfg = OmegaConf.create(
            {
                "accepted_samples": 2,
                "sample_batch_size": 3,
                "max_draws": 5,
                "num_projections": 1,
            }
        )

        value = constrained_w2(
            FakeRunner(),
            1.0,
            cfg,
            warning_callback=warnings.append,
            warning_context={"run_id": "run-a", "metric": "w2_trunc_abs_1"},
        )

        self.assertEqual(value, 1.0)
        self.assertEqual(len(warnings), 1)
        self.assertIn("sampling process failed", warnings[0])
        self.assertIn("using fallback W2=edge length 1", warnings[0])

    def test_warning_rows_from_run_rows_expands_warning_json(self) -> None:
        rows = [
            {
                "run_id": "run-a",
                "seed": 42,
                "method": "UIVI",
                "target": "student_uc",
                "checkpoint_epoch": 100,
                "warnings": '{"w2_trunc_abs_8": "sampling process failed"}',
            }
        ]

        [warning] = warning_rows_from_run_rows(rows)

        self.assertEqual(warning["run_id"], "run-a")
        self.assertEqual(warning["metric"], "w2_trunc_abs_8")
        self.assertEqual(warning["warning"], "sampling process failed")

    def test_summarize_computes_mean_and_standard_error(self) -> None:
        rows = [
            {
                "target": "banana",
                "method": "SIVI",
                "seed": 42,
                "elbo": 1.0,
                "duration_sec": 10.0,
                "checkpoint_epoch": 100,
            },
            {
                "target": "banana",
                "method": "SIVI",
                "seed": 43,
                "elbo": 3.0,
                "duration_sec": 14.0,
                "checkpoint_epoch": 100,
            },
        ]
        [summary] = summarize(rows)
        self.assertEqual(summary["variant"], "default")
        self.assertEqual(summary["seed_count"], 2)
        self.assertAlmostEqual(summary["elbo_mean"], 2.0)
        self.assertAlmostEqual(summary["elbo_se"], 1.0)
        self.assertAlmostEqual(summary["duration_sec_mean"], 12.0)
        self.assertAlmostEqual(summary["duration_sec_se"], 2.0)

    def test_summarize_accepts_csv_string_numbers_and_wall_clock(self) -> None:
        rows = [
            {
                "target": "banana",
                "method": "UIVI",
                "seed": "42",
                "elbo": "1.0",
                "duration_sec": "",
                "wall_clock_sec": "20.0",
                "checkpoint_epoch": "100",
            },
            {
                "target": "banana",
                "method": "UIVI",
                "seed": "43",
                "elbo": "3.0",
                "duration_sec": "",
                "wall_clock_sec": "24.0",
                "checkpoint_epoch": "100",
            },
        ]
        [summary] = summarize(rows)
        self.assertAlmostEqual(summary["elbo_mean"], 2.0)
        self.assertAlmostEqual(summary["wall_clock_sec_mean"], 22.0)
        self.assertAlmostEqual(summary["duration_sec_mean"], 22.0)

    def test_summarize_keeps_scheduler_variants_separate(self) -> None:
        rows = [
            {"target": "banana", "method": "DSIVI", "variant": "baseline", "seed": 42, "elbo": 1.0},
            {"target": "banana", "method": "DSIVI", "variant": "baseline", "seed": 43, "elbo": 3.0},
            {"target": "banana", "method": "DSIVI", "variant": "reverse_steps_1", "seed": 42, "elbo": 5.0},
            {"target": "banana", "method": "DSIVI", "variant": "reverse_steps_1", "seed": 43, "elbo": 7.0},
        ]

        summaries = {row["variant"]: row for row in summarize(rows)}

        self.assertEqual(set(summaries), {"baseline", "reverse_steps_1"})
        self.assertEqual(summaries["baseline"]["seed_count"], 2)
        self.assertAlmostEqual(summaries["baseline"]["elbo_mean"], 2.0)
        self.assertAlmostEqual(summaries["reverse_steps_1"]["elbo_mean"], 6.0)

    def test_prepare_config_applies_manifest_overrides_and_seed(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        rec = RunRecord(
            run_id="seed46_dsivi_banana_reverse_steps_1",
            seed=46,
            method="DSIVI",
            target="banana",
            runner_type="DSIVI",
            config_path=repo_root / "configs" / "dsivi_banana.yaml",
            result_path=repo_root / "results" / "unused",
            duration_sec=None,
            status="completed",
            entry={
                "variant": "reverse_steps_1",
                "extra_overrides": [
                    "train.reverse.epochs=1",
                    "train.reverse.batch_size=512",
                ],
            },
        )

        cfg = prepare_config(
            rec,
            device="cpu",
            scratch_results="results/test_finalization",
            scratch_tb="tb_logs/test_finalization",
        )

        self.assertEqual(cfg.seed, 46)
        self.assertEqual(cfg.train.reverse.epochs, 1)
        self.assertEqual(cfg.train.reverse.batch_size, 512)
        self.assertEqual(rec.variant, "reverse_steps_1")

    def test_sgld_baseline_is_not_added_without_langevin_runs(self) -> None:
        rows = [
            {
                "run_id": "seed42_dsivi_banana_baseline",
                "seed": 42,
                "variant": "baseline",
                "method": "DSIVI",
                "target": "banana",
            }
        ]
        cfg = OmegaConf.create(
            {
                "evaluation": {
                    "langevin_kde_elm": {
                        "enabled": True,
                        "sgld": {"enabled": True},
                    }
                }
            }
        )

        self.assertEqual(_append_langevin_sgld_if_needed(rows, [], cfg), rows)

    def test_toy_method_grid_filters_targets_methods_and_pools_training_time(self) -> None:
        rows = [
            {
                "target": "student_uc",
                "method": "UIVI",
                "elbo_mean": "-2.0",
                "elbo_se": "0.1",
                "w2_trunc_abs_8_mean": "0.8",
                "w2_trunc_abs_8_se": "0.01",
                "wall_clock_sec_mean": "30",
            },
            {
                "target": "8_gaussians",
                "method": "UIVI",
                "elbo_mean": "-0.8",
                "elbo_se": "0.2",
                "w2_trunc_abs_6_mean": "0.6",
                "w2_trunc_abs_6_se": "0.02",
                "wall_clock_sec_mean": "60",
            },
            {
                "target": "x_shaped",
                "method": "UIVI",
                "elbo_mean": "-0.1",
                "elbo_se": "0.3",
                "w2_mean": "0.4",
                "w2_se": "0.03",
                "wall_clock_sec_mean": "90",
            },
            {
                "target": "student_uc",
                "method": "AISIVI",
                "elbo_mean": "-3.0",
                "w2_trunc_abs_8_mean": "0.9",
                "wall_clock_sec_mean": "15",
            },
            {
                "target": "8_gaussians",
                "method": "AISIVI",
                "elbo_mean": "-1.8",
                "w2_trunc_abs_6_mean": "0.7",
                "wall_clock_sec_mean": "30",
            },
            {
                "target": "x_shaped",
                "method": "AISIVI",
                "elbo_mean": "-1.1",
                "w2_mean": "0.5",
                "wall_clock_sec_mean": "45",
            },
            {
                "target": "student_uc",
                "method": "DSIVI",
                "elbo_mean": "-1.0",
                "elbo_se": "0.001",
                "w2_trunc_abs_8_mean": "0.3",
                "wall_clock_sec_mean": "12",
            },
            {
                "target": "8_gaussians",
                "method": "DSIVI",
                "elbo_mean": "-0.4",
                "w2_trunc_abs_6_mean": "0.2",
                "wall_clock_sec_mean": "24",
            },
            {
                "target": "x_shaped",
                "method": "DSIVI",
                "elbo_mean": "-0.05",
                "elbo_se": "0.000799",
                "w2_mean": "0.1",
                "wall_clock_sec_mean": "36",
            },
            {
                "target": "banana",
                "method": "UIVI",
                "elbo_mean": "99",
                "w2_mean": "99",
                "wall_clock_sec_mean": "99",
            },
            {
                "target": "student_uc",
                "method": "SIVI",
                "elbo_mean": "99",
                "w2_trunc_abs_8_mean": "99",
                "wall_clock_sec_mean": "99",
            },
        ]
        cfg = OmegaConf.create({"tables": {"value_precision": 1, "se_precision": 2}})

        table = render_toy_method_grid(rows, cfg)

        self.assertIn("Target & Metric & UIVI & AISIVI & DSIVI", table)
        self.assertIn("UIVI", table)
        self.assertIn("AISIVI", table)
        self.assertIn("DSIVI", table)
        self.assertIn("$D_{\\mathrm{KL}}$", table)
        self.assertNotIn("banana", table)
        self.assertIn("Wall-clock time (s)", table)
        self.assertIn("& W2 &", table)
        self.assertNotIn("W2 $|x|<8$", table)
        self.assertNotIn("W2 $|x|<6$", table)
        self.assertIn("\\addlinespace[2pt]", table)
        self.assertIn("\\textbf{0.1} $\\pm$ {\\footnotesize \\textbf{0.00}}", table)
        self.assertIn("\\multicolumn{2}{l}{Wall-clock time (s)} & 60 & 30 & \\textbf{24}", table)

    def test_langevin_table_renders_sgld_separator_and_compact_iterations(self) -> None:
        rows = [
            {"target": "Langevin_post", "method": "SGLD", "kde_elm_mean": "80.0", "kde_elm_se": "0"},
            {
                "target": "Langevin_post",
                "method": "UIVI",
                "kde_elm_mean": "70.0",
                "kde_elm_se": "0.1",
                "wall_clock_sec_mean": "12",
                "wall_clock_sec_se": "1",
                "training_iterations_mean": "10000",
            },
        ]
        cfg = OmegaConf.create({"tables": {"value_precision": 1, "se_precision": 2}})

        table = render_langevin_table(rows, ["UIVI"], cfg)

        self.assertIn("Conditioned diffusion process results for expected log marginal likelihood and wall-clock time.", table)
        self.assertIn("Method & ELM & wall-clock time (s) & iterations", table)
        self.assertIn("SGLD & 80.0 $\\pm$ {\\footnotesize 0.00} & -- & --", table)
        self.assertIn("\\midrule\nUIVI", table)
        self.assertIn("UIVI & \\textbf{70.0} $\\pm$ {\\footnotesize 0.10} & \\textbf{12} & 10K", table)
        self.assertIn("10K", table)
        self.assertNotIn("1.000e+04", table)

    def test_bnn_table_uses_vertical_layout_and_wall_clock_summary(self) -> None:
        rows = [
            {
                "target": "Bnn_boston",
                "method": "SIVI",
                "rmse_mean": "3.0",
                "rmse_se": "0.1",
                "nll_mean": "2.0",
                "nll_se": "0.01",
                "wall_clock_sec_mean": "101.0",
            },
            {
                "target": "Bnn_boston",
                "method": "DSIVI",
                "rmse_mean": "2.5",
                "rmse_se": "0.2",
                "nll_mean": "1.9",
                "nll_se": "0.02",
                "wall_clock_sec_mean": "51.0",
            },
            {
                "target": "Bnn_yacht",
                "method": "SIVI",
                "rmse_mean": "4.0",
                "rmse_se": "0.3",
                "nll_mean": "3.0",
                "nll_se": "0.03",
                "wall_clock_sec_mean": "201.0",
            },
            {
                "target": "Bnn_yacht",
                "method": "DSIVI",
                "rmse_mean": "3.5",
                "rmse_se": "0.4",
                "nll_mean": "2.9",
                "nll_se": "0.04",
                "wall_clock_sec_mean": "61.0",
            },
        ]
        cfg = OmegaConf.create({"tables": {"value_precision": 1, "se_precision": 2}})

        table = render_bnn_table(rows, ["Bnn_boston", "Bnn_yacht"], ["SIVI", "DSIVI"], cfg)

        self.assertIn("Dataset & Metric & SIVI & DSIVI", table)
        self.assertIn("\\multirow{2}{*}{Boston}", table)
        self.assertIn("& RMSE & 3.0 $\\pm$ {\\footnotesize 0.10} & \\textbf{2.5} $\\pm$ {\\footnotesize 0.20}", table)
        self.assertIn("& NLL & 2.0 $\\pm$ {\\footnotesize 0.01} & \\textbf{1.9} $\\pm$ {\\footnotesize 0.02}", table)
        self.assertIn("\\addlinespace[2pt]", table)
        self.assertIn("\\multicolumn{2}{l}{Avg. wall-clock time} & 151 & \\textbf{56}", table)
        self.assertIn("Boston", table)
        self.assertNotIn("Bnn_boston", table)
        self.assertIn("{\\footnotesize", table)


if __name__ == "__main__":
    unittest.main()
