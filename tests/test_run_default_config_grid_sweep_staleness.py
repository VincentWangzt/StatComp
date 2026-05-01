from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from omegaconf import OmegaConf

from scripts import run_default_config_grid_sweep as sweep


class DefaultConfigGridStalenessTests(unittest.TestCase):
    def test_effective_hash_ignores_cuda_slot_and_output_roots(self) -> None:
        config_path = Path("configs/sivi_banana.yaml")

        left = sweep.effective_config_hash(
            config_path,
            seed=42,
            extra_overrides=[
                "cuda_visible_devices=0",
                "output.results_dir=results/a",
                "output.tb_dir=tb_logs/a",
            ],
        )
        right = sweep.effective_config_hash(
            config_path,
            seed=42,
            extra_overrides=[
                "cuda_visible_devices=7",
                "output.results_dir=/root/autodl-tmp/results/b",
                "output.tb_dir=/root/autodl-tmp/tb_logs/b",
            ],
        )

        self.assertEqual(left, right)

    def test_effective_hash_changes_for_seed_and_training_knobs(self) -> None:
        config_path = Path("configs/sivi_banana.yaml")

        baseline = sweep.effective_config_hash(config_path, seed=42, extra_overrides=[])
        different_seed = sweep.effective_config_hash(config_path, seed=43, extra_overrides=[])
        different_lr = sweep.effective_config_hash(
            config_path,
            seed=42,
            extra_overrides=["train.vi.lr=0.000123"],
        )

        self.assertNotEqual(baseline, different_seed)
        self.assertNotEqual(baseline, different_lr)

    def test_enqueue_pending_entries_respects_stale_and_retry_flags(self) -> None:
        fresh_entry = {"run_id": "fresh", "config_hash": "aaa"}
        stale_entry = {"run_id": "stale", "config_hash": "bbb"}
        legacy_entry = {"run_id": "legacy", "config_hash": "ccc"}
        failed_entry = {"run_id": "failed", "config_hash": "ddd"}
        new_entry = {"run_id": "new", "config_hash": "eee"}
        entries = [fresh_entry, stale_entry, legacy_entry, failed_entry, new_entry]
        statuses = {
            "fresh": {"status": "completed", "config_hash": "aaa"},
            "stale": {"status": "completed", "config_hash": "old"},
            "legacy": {"status": "completed"},
            "failed": {"status": "failed", "config_hash": "ddd"},
        }

        pending, stale, legacy = sweep.enqueue_pending_entries(
            entries,
            statuses,
            retry_failed=False,
            rerun_stale=False,
        )
        self.assertEqual([entry["run_id"] for entry in pending], ["new"])
        self.assertEqual([item["entry"]["run_id"] for item in stale], ["stale"])
        self.assertEqual([item["entry"]["run_id"] for item in legacy], ["legacy"])

        pending, _, _ = sweep.enqueue_pending_entries(
            entries,
            statuses,
            retry_failed=True,
            rerun_stale=True,
        )
        self.assertEqual([entry["run_id"] for entry in pending], ["stale", "failed", "new"])

    def test_current_counts_only_fresh_completed_and_finalized_runs(self) -> None:
        entries = [
            {"run_id": "fresh", "config_hash": "aaa"},
            {"run_id": "stale", "config_hash": "bbb"},
            {"run_id": "new", "config_hash": "ccc"},
        ]
        statuses = {
            "fresh": {"status": "completed", "config_hash": "aaa"},
            "stale": {"status": "completed", "config_hash": "old"},
        }
        finalize_statuses = {
            "fresh": {"status": "finalize_completed"},
            "stale": {"status": "finalize_completed"},
        }

        with tempfile.TemporaryDirectory() as tmp:
            current_path = Path(tmp) / "current.json"

            sweep.write_current(
                current_path,
                entries,
                active={},
                statuses=statuses,
                gpus=[0],
                finalize_statuses=finalize_statuses,
            )

            current = json.loads(current_path.read_text(encoding="utf-8"))

        self.assertEqual(current["total_runs"], 3)
        self.assertEqual(current["completed_runs"], 1)
        self.assertEqual(current["finalized_runs"], 1)

    def test_summary_markdown_counts_only_fresh_completed_runs(self) -> None:
        entries = [
            {
                "run_id": "fresh",
                "method": "SIVI",
                "method_slug": "sivi",
                "target": "banana",
                "target_slug": "banana",
                "seed": 42,
                "config_path": "configs/sivi_banana.yaml",
                "config_hash": "aaa",
                "config_hash_version": sweep.CONFIG_HASH_VERSION,
            },
            {
                "run_id": "stale",
                "method": "SIVI",
                "method_slug": "sivi",
                "target": "banana",
                "target_slug": "banana",
                "seed": 42,
                "config_path": "configs/sivi_banana.yaml",
                "config_hash": "bbb",
                "config_hash_version": sweep.CONFIG_HASH_VERSION,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            events = [
                {
                    "run_id": "fresh",
                    "status": "completed",
                    "config_hash": "aaa",
                    "tb_path": str(root / "tb" / "fresh"),
                },
                {
                    "run_id": "stale",
                    "status": "completed",
                    "config_hash": "old",
                    "tb_path": str(root / "tb" / "stale"),
                },
            ]
            report_dir = Path(tmp) / "reports"
            sweep.write_summary(report_dir, entries, events)
            summary_md = (report_dir / "summary.md").read_text(encoding="utf-8")

        self.assertIn("- Total manifest runs: 2", summary_md)
        self.assertIn("- Recorded completed runs: 1", summary_md)

    def test_artifact_config_hash_inventory_hashes_full_config_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_dir = root / "results" / "SIVI" / "banana" / "run"
            result_dir.mkdir(parents=True)
            full_config_path = result_dir / "full_config.yaml"
            OmegaConf.save(
                OmegaConf.create(
                    {
                        "seed": 42,
                        "cuda_visible_devices": "0",
                        "device": "cuda",
                        "runner_type": "SIVI",
                        "target_type": "banana",
                        "output": {"results_dir": "ignored"},
                        "train": {"epochs": 10},
                    }
                ),
                full_config_path,
            )

            statuses = {
                "hashed_run": {"status": "completed", "result_path": str(result_dir)},
                "missing_run": {"status": "completed", "result_path": str(root / "missing")},
                "failed_run": {"status": "failed", "result_path": str(result_dir)},
            }
            json_path = root / "runtime" / "artifact_config_hashes.json"
            csv_path = root / "runtime" / "artifact_config_hashes.csv"

            rows = sweep.write_artifact_hash_inventory(statuses, json_path, csv_path)

            self.assertEqual([row["run_id"] for row in rows], ["hashed_run", "missing_run"])
            self.assertEqual(rows[0]["status"], "hashed")
            self.assertEqual(rows[0]["artifact_config_hash"], sweep.artifact_config_hash(full_config_path))
            self.assertEqual(rows[1]["status"], "missing_full_config")
            self.assertTrue(json_path.exists())
            self.assertTrue(csv_path.exists())

            persisted = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted[0]["run_id"], "hashed_run")


if __name__ == "__main__":
    unittest.main()
