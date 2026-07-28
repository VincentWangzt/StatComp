from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from omegaconf import OmegaConf

from finalization.artifacts import RunRecord
from finalization.shared_checkpoint_score import (
    ArtifactPaths,
    SharedCheckpointSpec,
    aggregate_shared_results,
    artifact_paths,
    atomic_torch_save,
    build_shared_checkpoint_specs,
    method_artifact_fingerprint,
    refit_aisivi_flow,
    select_shared_checkpoint_specs,
    summarize_shared_results,
)


def make_record(
    *,
    method: str,
    config_path: Path,
    result_path: Path,
) -> RunRecord:
    return RunRecord(
        run_id=f"seed42_{method.lower()}_x_shaped",
        seed=42,
        method=method,
        target="x_shaped",
        runner_type=method,
        config_path=config_path,
        result_path=result_path,
        duration_sec=None,
        status="completed",
        entry={},
    )


def reference_config() -> dict:
    return {
        "estimator": "posterior_hmc",
        "total_samples": 100000,
        "num_chains": 20,
        "burn_in_steps": 1000,
        "thinning": 1,
        "step_size": 0.002,
        "leapfrog_steps": 20,
        "init_jitter_scale": 0.01,
        "adapt_step_size": True,
        "target_acceptance": 0.9,
        "adaptation_rate": 0.3,
        "min_step_size": 1.0e-5,
        "max_step_size": 0.01,
        "divergence_threshold": 1000.0,
        "accumulator_dtype": "float64",
        "quality": {
            "max_divergence_fraction": 0.01,
            "max_score_rhat_p95": 1.1,
            "max_epsilon_rhat_p95": 2.0,
            "min_post_burn_acceptance_rate": 0.6,
            "min_worst_chain_acceptance_rate": 0.05,
        },
    }


class TinyVI:

    def sampling(self, num: int) -> tuple[torch.Tensor, torch.Tensor]:
        epsilon = torch.randn(num, 1)
        z = epsilon + 0.1 * torch.randn_like(epsilon)
        return epsilon, z


class TinyReverse(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.0))

    def log_prob(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        return -((epsilon - self.weight * z) ** 2).sum(dim=-1)


class SharedCheckpointScoreTest(unittest.TestCase):

    def test_worker_filters_select_configured_cells(self) -> None:
        records = []
        specs = []
        for seed in (42, 43):
            record = RunRecord(
                run_id=f"seed{seed}_dsivi_x_shaped",
                seed=seed,
                method="DSIVI",
                target="x_shaped",
                runner_type="DSIVI",
                config_path=Path(f"seed_{seed}.yaml"),
                result_path=Path(f"seed_{seed}"),
                duration_sec=None,
                status="completed",
                entry={},
            )
            records.append(record)
            for epoch in (2000, 10000):
                specs.append(SharedCheckpointSpec(
                    source_record=record,
                    method_records=(record,),
                    progress=epoch / 10000,
                    epoch=epoch,
                    checkpoint_dir=Path(
                        f"seed_{seed}/checkpoints/epoch_{epoch}"
                    ),
                ))
        selected = select_shared_checkpoint_specs(
            specs,
            seeds=[43],
            epochs=[2000],
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0].source_record.seed, 43)
        self.assertEqual(selected[0].epoch, 2000)
        with self.assertRaisesRegex(ValueError, "not configured"):
            select_shared_checkpoint_specs(specs, seeds=[44])

    def test_method_fingerprint_tracks_config_and_dsivi_reverse(
        self,
    ) -> None:
        paths = ArtifactPaths(
            input_fingerprint="input",
            reference_fingerprint="reference",
            analysis_fingerprint="analysis",
            forward_bank=Path("forward.pt"),
            hmc_reference=Path("hmc.pt"),
            run_root=Path("run"),
        )
        baseline = method_artifact_fingerprint(
            paths=paths,
            method="DSIVI",
            estimator_config_fingerprint="config-a",
            dsivi_reverse_sha256="reverse-a",
        )
        changed_config = method_artifact_fingerprint(
            paths=paths,
            method="DSIVI",
            estimator_config_fingerprint="config-b",
            dsivi_reverse_sha256="reverse-a",
        )
        changed_reverse = method_artifact_fingerprint(
            paths=paths,
            method="DSIVI",
            estimator_config_fingerprint="config-a",
            dsivi_reverse_sha256="reverse-b",
        )
        self.assertNotEqual(baseline, changed_config)
        self.assertNotEqual(baseline, changed_reverse)

    def test_build_specs_uses_only_dsivi_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            manifest_path = root / "manifest.json"
            entries = []
            dsivi_checkpoint: Path | None = None
            for method in ("SIVI", "UIVI", "AISIVI", "DSIVI"):
                config_path = root / f"{method.lower()}.yaml"
                config_path.write_text(
                    "train:\n  epochs: 10000\n",
                    encoding="utf-8",
                )
                result_path = root / "results" / method
                result_path.mkdir(parents=True)
                if method == "DSIVI":
                    dsivi_checkpoint = (
                        result_path
                        / "checkpoints"
                        / "epoch_10000"
                    )
                    dsivi_checkpoint.mkdir(parents=True)
                    (dsivi_checkpoint / "vi_model.pt").write_bytes(b"vi")
                    (dsivi_checkpoint / "reverse_model.pt").write_bytes(
                        b"reverse"
                    )
                entries.append({
                    "run_id": f"seed42_{method.lower()}_x_shaped",
                    "seed": 42,
                    "method": method,
                    "target": "x_shaped",
                    "runner_type": method,
                    "config_path": str(config_path),
                    "result_path": str(result_path),
                    "status": "completed",
                })
            manifest_path.write_text(
                json.dumps(entries),
                encoding="utf-8",
            )
            cfg = OmegaConf.create({
                "campaign": {"manifest_path": str(manifest_path)},
                "selection": {
                    "source_method": "DSIVI",
                    "methods": [
                        "SIVI",
                        "UIVI",
                        "AISIVI",
                        "DSIVI",
                    ],
                    "targets": ["x_shaped"],
                    "seeds": [42],
                    "checkpoint_progress": [1.0],
                },
            })
            specs = build_shared_checkpoint_specs(cfg)
            self.assertEqual(len(specs), 1)
            self.assertEqual(specs[0].checkpoint_dir, dsivi_checkpoint)
            self.assertEqual(
                {
                    record.method
                    for record in specs[0].method_records
                },
                {"SIVI", "UIVI", "AISIVI", "DSIVI"},
            )

    def test_aisivi_refit_runs_exact_steps_and_resumes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "aisivi.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            record = make_record(
                method="AISIVI",
                config_path=config_path,
                result_path=root,
            )
            spec = SharedCheckpointSpec(
                source_record=record,
                method_records=(record,),
                progress=1.0,
                epoch=10000,
                checkpoint_dir=root,
            )
            paths = ArtifactPaths(
                input_fingerprint="input",
                reference_fingerprint="reference",
                analysis_fingerprint="analysis",
                forward_bank=root / "forward.pt",
                hmc_reference=root / "hmc.pt",
                run_root=root / "run",
            )
            cfg = OmegaConf.create({
                "evaluation": {
                    "aisivi_refit": {
                        "steps": 3,
                        "batch_size": 8,
                        "checkpoint_every": 1,
                        "log_every": 1,
                    },
                },
            })

            def make_runner() -> SimpleNamespace:
                reverse = TinyReverse()
                optimizer = torch.optim.Adam(
                    reverse.parameters(),
                    lr=0.01,
                )
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=2,
                    gamma=0.5,
                )
                return SimpleNamespace(
                    vi_model=TinyVI(),
                    reverse_model=reverse,
                    training_reverse_optimizer=optimizer,
                    training_reverse_scheduler=scheduler,
                    rev_train_cfg=OmegaConf.create({
                        "scheduler": {
                            "type": "StepLR",
                            "step_size": 2,
                            "gamma": 0.5,
                        },
                    }),
                    rev_batch_size=8,
                    reverse_lr=0.01,
                    reverse_weight_decay=0.0,
                    grad_clip=10.0,
                    device="cpu",
                    config=OmegaConf.create({
                        "config_path": str(config_path),
                    }),
                )

            first = refit_aisivi_flow(
                make_runner(),
                cfg,
                spec,
                paths,
                resume=True,
                estimator_config_fingerprint="estimator-config",
            )
            second = refit_aisivi_flow(
                make_runner(),
                cfg,
                spec,
                paths,
                resume=True,
                estimator_config_fingerprint="estimator-config",
            )
            self.assertEqual(first["completed_steps"], 3)
            self.assertEqual(second["completed_steps"], 3)
            self.assertTrue(paths.aisivi_flow(spec).is_file())

    def test_aggregate_reports_method_hmc_and_uivi_acceptance(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            config_path = root / "method.yaml"
            config_path.write_text("{}\n", encoding="utf-8")
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            torch.save({"weight": torch.tensor(1.0)}, checkpoint / "vi_model.pt")
            records = tuple(
                make_record(
                    method=method,
                    config_path=config_path,
                    result_path=root,
                )
                for method in ("SIVI", "UIVI", "AISIVI", "DSIVI")
            )
            spec = SharedCheckpointSpec(
                source_record=records[-1],
                method_records=records,
                progress=1.0,
                epoch=10000,
                checkpoint_dir=checkpoint,
            )
            cfg = OmegaConf.create({
                "selection": {
                    "source_method": "DSIVI",
                    "methods": [
                        "SIVI",
                        "UIVI",
                        "AISIVI",
                        "DSIVI",
                    ],
                    "targets": ["x_shaped"],
                    "seeds": [42],
                    "checkpoint_progress": [1.0],
                },
                "evaluation": {
                    "device": "cpu",
                    "forward_batch_size": 2,
                    "reference": reference_config(),
                },
                "output": {
                    "runtime_dir": str(root / "runtime"),
                    "report_dir": str(root / "report"),
                    "scratch_results_dir": str(root / "scratch-results"),
                    "scratch_tb_dir": str(root / "scratch-tb"),
                },
            })
            paths = artifact_paths(cfg, spec)
            atomic_torch_save(paths.forward_bank, {
                "schema_version": 1,
                "input_fingerprint": paths.input_fingerprint,
                "z": torch.zeros(2, 1),
                "generating_epsilon": torch.zeros(2, 1),
            })
            hmc_scores = torch.tensor([
                [[0.0], [0.0]],
                [[2.0], [2.0]],
            ])
            atomic_torch_save(paths.hmc_reference, {
                "schema_version": 1,
                "reference_fingerprint": paths.reference_fingerprint,
                "input_fingerprint": paths.input_fingerprint,
                "chain_score_means": hmc_scores,
                "runtime_sec": 1.0,
                "quality_status": "pass",
                "quality_issues": [],
                "diagnostics": {
                    "hmc_acceptance_rate": 0.9,
                    "hmc_post_burn_acceptance_rate": 0.91,
                    "hmc_score_rhat_p95": 1.01,
                },
            })
            for method in cfg.selection.methods:
                diagnostics = {"native_auxiliary_samples": 5}
                if method == "UIVI":
                    diagnostics["uivi_hmc_acceptance_rate"] = 0.375
                atomic_torch_save(
                    paths.method_score(spec, str(method)),
                    {
                        "schema_version": 1,
                        "analysis_fingerprint": paths.analysis_fingerprint,
                        "input_fingerprint": paths.input_fingerprint,
                        "estimator_run_id": (
                            f"seed42_{str(method).lower()}_x_shaped"
                        ),
                        "estimator_config_path": str(config_path),
                        "method_score": torch.ones(2, 1),
                        "diagnostics": diagnostics,
                        "aisivi_refit": None,
                        "runtime_sec": 0.1,
                        "completed_at": "now",
                    },
                )
            rows, report_dir = aggregate_shared_results(cfg, [spec])
            self.assertEqual(len(rows), 4)
            self.assertTrue(
                (report_dir / "score_approximation_table.md").is_file()
            )
            self.assertTrue(
                (report_dir / "checkpoint_summary.csv").is_file()
            )
            self.assertTrue(
                (report_dir / "hmc_checkpoint_summary.csv").is_file()
            )
            self.assertTrue(
                (report_dir / "method_hmc_l2_table.md").is_file()
            )
            self.assertTrue(
                (report_dir / "hmc_internal_l2_table.md").is_file()
            )
            for row in rows:
                self.assertAlmostEqual(row["method_hmc_l2"], 0.0)
                self.assertAlmostEqual(row["hmc_internal_l2"], 1.0)
            uivi = next(row for row in rows if row["method"] == "UIVI")
            self.assertAlmostEqual(
                uivi["uivi_average_acceptance_rate"],
                0.375,
            )

    def test_summaries_use_sample_sd_and_deduplicate_hmc(
        self,
    ) -> None:
        rows: list[dict[str, object]] = []
        for seed, method_l2, internal_l2 in (
            (42, 1.0, 2.0),
            (43, 3.0, 4.0),
        ):
            for method in ("SIVI", "UIVI"):
                rows.append({
                    "target": "x_shaped",
                    "progress": 1.0,
                    "epoch": 10000,
                    "seed": seed,
                    "method": method,
                    "method_hmc_l2": method_l2,
                    "method_hmc_relative_l2": method_l2 / 10.0,
                    "method_runtime_sec": 1.0,
                    "native_auxiliary_samples": 5,
                    "uivi_average_acceptance_rate": (
                        0.2 * (seed - 41)
                        if method == "UIVI"
                        else None
                    ),
                    "hmc_internal_l2": internal_l2,
                    "hmc_mean_mcse_l2": internal_l2 / 20.0,
                    "hmc_runtime_sec": 2.0,
                    "hmc_quality_status": "pass",
                    "hmc_quality_issues": "[]",
                    "hmc_average_acceptance_rate": 0.9,
                    "hmc_post_burn_acceptance_rate": 0.8,
                    "hmc_score_rhat_p95": 1.01,
                    "hmc_reference_path": f"hmc_{seed}.pt",
                })
        method_summary, hmc_summary = summarize_shared_results(rows)
        self.assertEqual(len(method_summary), 2)
        self.assertEqual(len(hmc_summary), 1)
        sivi = next(
            row for row in method_summary if row["method"] == "SIVI"
        )
        self.assertAlmostEqual(sivi["method_hmc_l2_mean"], 2.0)
        self.assertAlmostEqual(
            sivi["method_hmc_l2_sd"],
            2.0 ** 0.5,
        )
        self.assertAlmostEqual(
            hmc_summary[0]["hmc_internal_l2_mean"],
            3.0,
        )
        self.assertAlmostEqual(
            hmc_summary[0]["hmc_internal_l2_sd"],
            2.0 ** 0.5,
        )


if __name__ == "__main__":
    unittest.main()
