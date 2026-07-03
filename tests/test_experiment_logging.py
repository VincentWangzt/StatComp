from __future__ import annotations

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from finalization.artifacts import RunRecord, load_grad_norm_series, resolve_tb_metrics_csv
from utils.experiment_logging import ExperimentLogger, metric, timer


class _FakeRun:
    def __init__(self) -> None:
        self.id = "run123"
        self.url = "https://wandb.invalid/run123"
        self.path = ("entity", "StatComp", "run123")
        self.logged: list[dict] = []
        self.saved: list[str] = []
        self.finished: int | None = None

    def define_metric(self, *args, **kwargs) -> None:
        pass

    def log(self, values: dict) -> None:
        self.logged.append(values)

    def save(self, path: str, **kwargs) -> None:
        self.saved.append(path)

    def finish(self, *, exit_code: int = 0) -> None:
        self.finished = exit_code


class _MetricOwner:
    def __init__(self, logger: ExperimentLogger) -> None:
        self.experiment_logger = logger

    @metric("metric/test/scalar")
    def scalar(self, epoch: int) -> float:
        return 2.5

    @metric("metric/test/left", "metric/test/right")
    def pair(self, epoch: int) -> tuple[float, float]:
        return 1.0, 3.0

    @metric(prefix="metric/test")
    def mapping(self, epoch: int) -> dict[str, float]:
        return {"alpha": 4.0, "beta": 5.0}

    @timer("decorated")
    def timed(self, epoch: int) -> str:
        return "unchanged"

    @timer("failure")
    def failing(self, epoch: int) -> None:
        raise RuntimeError("boom")


class ExperimentLoggingTests(unittest.TestCase):
    def _make_logger(self, root: Path, fake: _FakeRun) -> ExperimentLogger:
        with patch.dict(os.environ, {"WANDB_API_KEY": "secret-key"}, clear=False), patch(
            "wandb.init", return_value=fake
        ) as init:
            logger = ExperimentLogger(
                save_path=root,
                config={"tracking": {"enabled": True}, "seed": 7},
                runner_name="SIVI",
                target_type="banana",
                vi_model_type="ConditionalGaussian",
                seed=7,
                time_avg_window=2,
            )
        init_config = init.call_args.kwargs["config"]
        self.assertNotIn("secret-key", repr(init_config))
        self.assertEqual(logger.wandb_path, "entity/StatComp/run123")
        return logger

    def test_metric_timer_and_csv_routing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeRun()
            logger = self._make_logger(Path(tmp), fake)
            owner = _MetricOwner(logger)

            self.assertEqual(owner.scalar(4), 2.5)
            self.assertEqual(owner.pair(4), (1.0, 3.0))
            self.assertEqual(owner.mapping(4), {"alpha": 4.0, "beta": 5.0})
            self.assertEqual(owner.timed(4), "unchanged")
            with self.assertRaisesRegex(RuntimeError, "boom"):
                owner.failing(5)
            with logger.timer("context", step=6):
                pass
            logger.log_scalars(
                {
                    "train/loss": 9.0,
                    "diagnostic/norm": 8.0,
                    "metric/test/kept": 7.0,
                },
                step=6,
            )
            logger.finish()

            with (Path(tmp) / "metrics.csv").open(newline="", encoding="utf-8") as fh:
                rows = list(csv.DictReader(fh))
            tags = {row["tag"] for row in rows}
            self.assertIn("metric/test/scalar", tags)
            self.assertIn("metric/test/alpha", tags)
            self.assertIn("time_avg/decorated", tags)
            self.assertIn("time_avg/failure", tags)
            self.assertIn("time_avg/context", tags)
            self.assertIn("metric/test/kept", tags)
            self.assertNotIn("train/loss", tags)
            self.assertNotIn("diagnostic/norm", tags)
            self.assertNotIn("time/decorated", tags)

    def test_online_failure_retries_offline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            fake = _FakeRun()
            with patch.dict(os.environ, {"WANDB_API_KEY": "secret-key"}, clear=False), patch(
                "wandb.init", side_effect=[RuntimeError("network"), fake]
            ) as init, patch("wandb.finish"):
                logger = ExperimentLogger(
                    save_path=tmp,
                    config={"tracking": {"enabled": True}},
                    runner_name="SIVI",
                    target_type="banana",
                    vi_model_type="ConditionalGaussian",
                    seed=42,
                )
            self.assertEqual(init.call_args_list[0].kwargs["mode"], "online")
            self.assertEqual(init.call_args_list[1].kwargs["mode"], "offline")
            self.assertEqual(logger.wandb_mode, "offline")
            logger.finish()

    def test_campaign_name_group_and_tags(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            save_path = Path(tmp) / "20260624_143000"
            fake = _FakeRun()
            with patch.dict(os.environ, {"WANDB_API_KEY": "secret-key"}, clear=False), patch(
                "wandb.init", return_value=fake
            ) as init:
                logger = ExperimentLogger(
                    save_path=save_path,
                    config={"tracking": {"campaign": "default_config_grid"}},
                    runner_name="SIVI",
                    target_type="banana",
                    vi_model_type="ConditionalGaussian",
                    seed=42,
                )
            self.assertEqual(
                logger.run_name,
                "default_config_grid-SIVI-banana-seed42-20260624_143000",
            )
            self.assertEqual(init.call_args.kwargs["group"], "default_config_grid")
            self.assertIn("method:SIVI", logger.tags)
            self.assertIn("target:banana", logger.tags)
            self.assertIn("campaign:default_config_grid", logger.tags)
            logger.finish()

    def test_disabled_tracking_still_writes_filtered_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            logger = ExperimentLogger(
                save_path=tmp,
                config={"tracking": {"enabled": False}},
                runner_name="SIVI",
                target_type="banana",
                vi_model_type="ConditionalGaussian",
                seed=42,
            )
            logger.log_scalars({"metric/test/value": 1.0}, step=1)
            logger.finish()
            self.assertTrue((Path(tmp) / "metrics.csv").is_file())
            self.assertEqual(logger.wandb_mode, "disabled")

    def test_metric_resolution_prefers_live_csv_and_keeps_legacy_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result_path = root / "results" / "SIVI" / "banana" / "stamp"
            legacy = root / "tb" / "SIVI" / "banana" / "stamp" / "extracted"
            result_path.mkdir(parents=True)
            legacy.mkdir(parents=True)
            live_csv = result_path / "metrics.csv"
            legacy_csv = legacy / "metrics.csv"
            live_csv.write_text("tag,step,wall_time,value\n", encoding="utf-8")
            legacy_csv.write_text("tag,step,wall_time,value\n", encoding="utf-8")
            record = RunRecord(
                run_id="run",
                seed=1,
                method="SIVI",
                target="banana",
                runner_type="SIVI",
                config_path=root / "config.yaml",
                result_path=result_path,
                duration_sec=1.0,
                status="completed",
                entry={"tb_dir": str(root / "tb")},
            )
            self.assertEqual(resolve_tb_metrics_csv(record), live_csv)
            live_csv.unlink()
            self.assertEqual(resolve_tb_metrics_csv(record), legacy_csv)

    def test_diagnostic_history_can_come_from_wandb(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            root.mkdir(exist_ok=True)
            (root / "wandb_run.json").write_text(
                '{"run_path":"entity/StatComp/run123","mode":"online"}',
                encoding="utf-8",
            )
            record = RunRecord(
                run_id="run123",
                seed=1,
                method="SIVI",
                target="banana",
                runner_type="SIVI",
                config_path=root / "config.yaml",
                result_path=root,
                duration_sec=1.0,
                status="completed",
                entry={},
            )

            class FakeApiRun:
                def scan_history(self, **kwargs):
                    tag = "diagnostic/vi_model/grad_norm"
                    return iter(
                        [
                            {tag: 3.0, "epoch": 1, "_timestamp": 10.0},
                            {tag: 2.0, "epoch": 2, "_timestamp": 11.0},
                        ]
                    )

            class FakeApi:
                def run(self, path):
                    self.path = path
                    return FakeApiRun()

            with patch("wandb.Api", return_value=FakeApi()):
                series = load_grad_norm_series(record)
            self.assertIsNotNone(series)
            assert series is not None
            self.assertEqual(series[0].tolist(), [1.0, 2.0])
            self.assertEqual(series[2].tolist(), [3.0, 2.0])


if __name__ == "__main__":
    unittest.main()
