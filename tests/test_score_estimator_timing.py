from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from finalization.score_approximation import build_frozen_estimators, method_native_score, select_checkpoints, shared_input_bank
from finalization.score_estimator_timing import ESTIMATORS, aggregate_timings, benchmark_estimator, load_timing_config, run_benchmark
from tests import test_score_approximation as fixtures


class TimingBoundaryTest(unittest.TestCase):
    def test_warmup_and_input_generation_are_outside_timer(self) -> None:
        runner = SimpleNamespace(device="cpu", z_dim=2)
        bank = torch.arange(12, dtype=torch.float32).reshape(3, 2, 2)
        seen = []

        def estimator(runner, z, epsilon):
            seen.append(z.clone())
            return z

        with patch("finalization.score_estimator_timing.time.perf_counter_ns", side_effect=[0, 2_000_000, 5_000_000, 10_000_000]):
            elapsed, peak = benchmark_estimator(runner, estimator, z_bank=bank, epsilon_bank=bank,
                                                warmup_calls=1, timed_calls=2)
        self.assertEqual(elapsed, [2, 5])
        self.assertEqual(peak, 0)
        torch.testing.assert_close(torch.stack(seen), bank)

    def test_nonfinite_scores_are_rejected(self) -> None:
        runner = SimpleNamespace(device="cpu", z_dim=1)
        bank = torch.zeros(2, 1, 1)
        with self.assertRaises(FloatingPointError):
            benchmark_estimator(runner, lambda runner, z, epsilon: z + torch.nan,
                                z_bank=bank, epsilon_bank=bank, warmup_calls=0, timed_calls=2)

    def test_uncertainty_is_over_seed_means(self) -> None:
        rows = [{"target": "banana", "epoch": 1, "method": "DSIVI", "batch_size": 2,
                 "latency_ms_mean": mean} for mean in (1.0, 3.0)]
        summary = aggregate_timings(rows)[0]
        self.assertEqual(summary["seed_count"], 2)
        self.assertEqual(summary["latency_ms_mean"], 2)
        self.assertEqual(summary["latency_ms_se"], 1)


class FrozenTimingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        fixtures.FrozenCheckpointTest.setUpClass()
        cls.cfg = OmegaConf.merge(load_timing_config(), fixtures.FrozenCheckpointTest.cfg,
                                 {"evaluation": {"batch_sizes": [1, 3], "warmup_calls": 1, "timed_calls": 2}})

    @classmethod
    def tearDownClass(cls) -> None:
        fixtures.FrozenCheckpointTest.tearDownClass()

    def test_timed_implementations_match_accuracy_estimators(self) -> None:
        checkpoint = select_checkpoints(self.cfg)[0]
        runners = build_frozen_estimators(self.cfg, checkpoint)
        epsilon, z = shared_input_bank(next(iter(runners.values())), checkpoint, count=3, seed=42)
        for method, runner in runners.items():
            with self.subTest(method=method):
                torch.manual_seed(29)
                expected, _ = method_native_score(runner, method, z, epsilon)
                torch.manual_seed(29)
                actual = ESTIMATORS[method](runner, z, epsilon)
                torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)

    def test_benchmark_shares_inputs_and_reuses_accuracy_proposal(self) -> None:
        checkpoint = select_checkpoints(self.cfg)[0]
        runners = build_frozen_estimators(self.cfg, checkpoint)
        proposal_hash = runners["AISIVI"].score_refit_metadata["proposal_sha256"]
        seen = {}

        def record_inputs(runner, estimator, **kwargs):
            seen.setdefault(kwargs["z_bank"].shape[1], []).append(kwargs["z_bank"].clone())
            return benchmark_estimator(runner, estimator, **kwargs)

        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Use shared fitted proposal")), \
             patch("finalization.score_estimator_timing.benchmark_estimator", side_effect=record_inputs):
            rows = run_benchmark(self.cfg)
        self.assertEqual(len(rows), 8)
        self.assertEqual(len({row["vi_model_sha256"] for row in rows}), 1)
        for banks in seen.values():
            self.assertEqual(len(banks), 4)
            for bank in banks[1:]:
                torch.testing.assert_close(banks[0], bank, rtol=0, atol=0)
        metadata = json.loads((Path(self.cfg.output.report_dir) / "metadata.json").read_text())
        self.assertEqual(metadata["checkpoints"][0]["aisivi_refit"]["proposal_sha256"], proposal_hash)
        self.assertEqual(metadata["environment"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
