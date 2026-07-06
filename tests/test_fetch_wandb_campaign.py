from __future__ import annotations

import unittest

from writeup.scripts.fetch_wandb_campaign import method_target_from_group


class WandbCampaignParsingTests(unittest.TestCase):
    def test_kdvi_group_is_labeled_as_kdvi_mmd(self) -> None:
        self.assertEqual(
            method_target_from_group("KDVI-banana"),
            ("KDVI-MMD", "banana"),
        )

    def test_kdvi_w2_group_preserves_full_method_prefix(self) -> None:
        self.assertEqual(
            method_target_from_group("KDVI-W2-banana"),
            ("KDVI-W2", "banana"),
        )

    def test_dsivi_group_is_unchanged(self) -> None:
        self.assertEqual(
            method_target_from_group("DSIVI-banana"),
            ("DSIVI", "banana"),
        )


if __name__ == "__main__":
    unittest.main()
