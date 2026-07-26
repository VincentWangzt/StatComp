import unittest
from pathlib import Path

from scripts.plot_nfvi_rebuttal_checkpoints import (
    RunRecord,
    select_median_w2_records,
)


class RepresentativeCheckpointSelectionTests(unittest.TestCase):
    def test_selects_observed_median_w2_per_method(self) -> None:
        records = [
            RunRecord("DIVI", 42, -0.1, 0.1, Path("divi-42")),
            RunRecord("DIVI", 43, -0.2, 0.5, Path("divi-43")),
            RunRecord("DIVI", 44, -0.3, 0.3, Path("divi-44")),
            RunRecord("NFVI-4", 42, -0.4, 0.6, Path("nfvi-42")),
            RunRecord("NFVI-4", 43, -0.5, 0.2, Path("nfvi-43")),
            RunRecord("NFVI-4", 44, -0.6, 0.4, Path("nfvi-44")),
        ]

        selected = select_median_w2_records(records, ["DIVI", "NFVI-4"])

        self.assertEqual(
            [(record.method, record.seed) for record in selected],
            [("DIVI", 44), ("NFVI-4", 44)],
        )

    def test_missing_method_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "NFVI-16"):
            select_median_w2_records([], ["NFVI-16"])


if __name__ == "__main__":
    unittest.main()
