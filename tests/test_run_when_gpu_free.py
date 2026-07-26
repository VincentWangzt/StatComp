import unittest
import json
import tempfile
from pathlib import Path

from scripts.run_when_gpu_free import (
    manifest_has_nonterminal_status,
    parse_args,
    wait_until_gpu_is_free,
)


class RunWhenGpuFreeTest(unittest.TestCase):

    def test_requires_a_sustained_idle_window_after_busy_gpu(self) -> None:
        process_states = iter([
            ["100, python"],
            [],
            [],
            [],
        ])
        now = [0.0]
        reports: list[str] = []

        wait_until_gpu_is_free(
            process_query=lambda: next(process_states),
            blocker_query=lambda: False,
            poll_seconds=10.0,
            idle_seconds=20.0,
            report=reports.append,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

        self.assertEqual(now[0], 30.0)
        self.assertIn("GPU compute processes", reports[0])
        self.assertIn("starting command", reports[-1])

    def test_blocker_resets_the_idle_window(self) -> None:
        blocker_states = iter([True, False, False])
        now = [0.0]

        wait_until_gpu_is_free(
            process_query=lambda: [],
            blocker_query=lambda: next(blocker_states),
            poll_seconds=5.0,
            idle_seconds=5.0,
            report=lambda _message: None,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

        self.assertEqual(now[0], 10.0)

    def test_parser_strips_command_separator(self) -> None:
        args = parse_args([
            "--idle-seconds",
            "0",
            "--",
            "python",
            "job.py",
        ])

        self.assertEqual(args.command, ["python", "job.py"])

    def test_telemetry_must_remain_below_threshold(self) -> None:
        telemetry_states = iter([
            (99.0, 100.0),
            (0.0, 100.0),
            (0.0, 100.0),
        ])
        now = [0.0]

        wait_until_gpu_is_free(
            process_query=lambda: [],
            blocker_query=lambda: False,
            telemetry_query=lambda: next(telemetry_states),
            max_utilization=5.0,
            max_used_memory_mib=512.0,
            poll_seconds=5.0,
            idle_seconds=5.0,
            report=lambda _message: None,
            clock=lambda: now[0],
            sleep=lambda seconds: now.__setitem__(0, now[0] + seconds),
        )

        self.assertEqual(now[0], 10.0)

    def test_manifest_pending_status_blocks_launch(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = Path(temp_dir) / "manifest.json"
            manifest.write_text(
                json.dumps([{"status": "completed"}, {"status": "pending"}]),
                encoding="utf-8",
            )
            self.assertTrue(
                manifest_has_nonterminal_status(manifest, {"pending", "running"})
            )
            manifest.write_text(
                json.dumps([{"status": "completed"}, {"status": "failed"}]),
                encoding="utf-8",
            )
            self.assertFalse(
                manifest_has_nonterminal_status(manifest, {"pending", "running"})
            )


if __name__ == "__main__":
    unittest.main()
