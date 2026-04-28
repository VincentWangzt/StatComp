from __future__ import annotations

import unittest

from finalization.artifacts import normalize_target
from finalization.plots import langevin_panel_labels
from finalization.runner_eval import summarize


class FinalizationTests(unittest.TestCase):
    def test_target_aliases(self) -> None:
        self.assertEqual(normalize_target("multi_model"), "multimodal")
        self.assertEqual(normalize_target("8_gaussian"), "8_gaussians")
        self.assertEqual(normalize_target("banana"), "banana")

    def test_langevin_panel_labels_add_sgld_only_for_odd_method_count(self) -> None:
        self.assertEqual(
            langevin_panel_labels(["SIVI", "UIVI", "AISIVI"]),
            ["SIVI", "UIVI", "AISIVI", "SGLD"],
        )
        self.assertEqual(
            langevin_panel_labels(["SIVI", "UIVI", "AISIVI", "DSIVI"]),
            ["SIVI", "UIVI", "AISIVI", "DSIVI"],
        )

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


if __name__ == "__main__":
    unittest.main()

