from __future__ import annotations

import unittest

from finalization.artifacts import normalize_target
from omegaconf import OmegaConf

from finalization.plots import langevin_panel_labels
from finalization.runner_eval import summarize, truncated_w2_metric_name
from finalization.tables import render_toy_method_grid


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

    def test_toy_method_grid_filters_targets_methods_and_averages_training_time(self) -> None:
        rows = [
            {
                "target": "student_uc",
                "method": "UIVI",
                "elbo_mean": "-2.0",
                "elbo_se": "0.1",
                "w2_trunc_abs_8_mean": "0.8",
                "w2_trunc_abs_8_se": "0.01",
                "training_time_sec_mean": "30",
            },
            {
                "target": "8_gaussians",
                "method": "UIVI",
                "elbo_mean": "-0.8",
                "elbo_se": "0.2",
                "w2_trunc_abs_6_mean": "0.6",
                "w2_trunc_abs_6_se": "0.02",
                "training_time_sec_mean": "60",
            },
            {
                "target": "x_shaped",
                "method": "UIVI",
                "elbo_mean": "-0.1",
                "elbo_se": "0.3",
                "w2_mean": "0.4",
                "w2_se": "0.03",
                "training_time_sec_mean": "90",
            },
            {
                "target": "student_uc",
                "method": "AISIVI",
                "elbo_mean": "-3.0",
                "w2_trunc_abs_8_mean": "0.9",
                "training_time_sec_mean": "15",
            },
            {
                "target": "8_gaussians",
                "method": "AISIVI",
                "elbo_mean": "-1.8",
                "w2_trunc_abs_6_mean": "0.7",
                "training_time_sec_mean": "30",
            },
            {
                "target": "x_shaped",
                "method": "AISIVI",
                "elbo_mean": "-1.1",
                "w2_mean": "0.5",
                "training_time_sec_mean": "45",
            },
            {
                "target": "student_uc",
                "method": "DSIVI",
                "elbo_mean": "-1.0",
                "w2_trunc_abs_8_mean": "0.3",
                "training_time_sec_mean": "12",
            },
            {
                "target": "8_gaussians",
                "method": "DSIVI",
                "elbo_mean": "-0.4",
                "w2_trunc_abs_6_mean": "0.2",
                "training_time_sec_mean": "24",
            },
            {
                "target": "x_shaped",
                "method": "DSIVI",
                "elbo_mean": "-0.05",
                "w2_mean": "0.1",
                "training_time_sec_mean": "36",
            },
            {
                "target": "banana",
                "method": "UIVI",
                "elbo_mean": "99",
                "w2_mean": "99",
                "training_time_sec_mean": "99",
            },
            {
                "target": "student_uc",
                "method": "SIVI",
                "elbo_mean": "99",
                "w2_trunc_abs_8_mean": "99",
                "training_time_sec_mean": "99",
            },
        ]
        cfg = OmegaConf.create({"tables": {"value_precision": 1, "se_precision": 2}})

        table = render_toy_method_grid(rows, cfg)

        self.assertIn("Method & \\multicolumn{3}{c}{ELBO} & \\multicolumn{3}{c}{W2} & Avg. training time (s)", table)
        self.assertIn("UIVI", table)
        self.assertIn("AISIVI", table)
        self.assertIn("DSIVI", table)
        self.assertNotRegex(table, r"(?m)^SIVI\s*&")
        self.assertNotIn("banana", table)
        self.assertIn("60.0 $\\pm$ 17.32", table)
        self.assertIn("24.0 $\\pm$ 6.93", table)


if __name__ == "__main__":
    unittest.main()
