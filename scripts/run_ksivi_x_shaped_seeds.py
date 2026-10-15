"""Run the canonical ten X-shaped KSIVI seeds in batches of three on GPU 0."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from finalization.artifacts import completed_runs, find_final_checkpoint, find_final_samples, load_sample_z
from finalization.runner_eval import write_csv
from scripts import run_default_config_grid_sweep as sweep

CAMPAIGN = "ksivi_x_shaped"
MANIFEST = f"campaigns/{CAMPAIGN}/manifest.json"
SEEDS = list(range(42, 52))
OVERRIDES = ["train.log.metric_log_freq=0", "train.plot.freq=1000000000"]


def validate_entry(entry: dict) -> None:
    run = sweep.repo_path(entry["result_path"])
    snapshot = run / "full_config.yaml"
    # BaseSIVIRunner resolves a toy target's mixing dimension to model.z_dim
    # and inserts the disabled Jacobian diagnostic into the saved snapshot.
    expected = sweep._resolved_effective_config(entry["config_path"], entry["seed"], OVERRIDES)
    expected["target"]["epsilon_dim"] = expected["target"]["z_dim"]
    expected["vi_model"]["epsilon_dim"] = expected["target"]["z_dim"]
    expected["metric"].setdefault("jacobian_spectral", dict(enabled=False, num_samples=128))
    actual_hash = sweep.artifact_config_hash(snapshot)
    assert actual_hash == sweep._hash_normalized_payload(expected), entry["seed"]
    entry["artifact_config_hash"] = actual_hash
    config = OmegaConf.load(snapshot)
    assert config.seed == entry["seed"] and config.device == "cuda"
    assert config.train.epochs == 50000 and config.train.batch_size == 128
    assert config.train.vi.lr == 1e-3 and config.train.vi.var_lr == 1e-5
    assert config.vi_model.variance_init == 1.313
    assert config.target.epsilon_dim == config.vi_model.epsilon_dim == 2
    assert config.train.annealing.enabled and config.train.annealing.steps == 25000
    assert config.train.ksivi.kernel == "gaussian" and not config.train.ksivi.detach_kernel
    checkpoint, epoch = find_final_checkpoint(run)
    weights = torch.load(checkpoint / "vi_model.pt", map_location="cpu", weights_only=True)
    assert epoch == 50000 and all(torch.isfinite(value).all() for value in weights.values())
    optimizer = torch.load(checkpoint / "vi_optim.pt", map_location="cpu", weights_only=True)
    assert {int(value["step"]) for value in optimizer["state"].values()} == {50000}
    sample_path, sample_epoch = find_final_samples(run)
    samples = load_sample_z(sample_path)
    assert sample_epoch == epoch and samples.shape == (10000, 2) and torch.isfinite(samples).all()


def canonical_records() -> list:
    entries = json.loads((ROOT / MANIFEST).read_text(encoding="utf-8"))
    assert len(entries) == 10 and sorted(entry["seed"] for entry in entries) == SEEDS
    assert all(entry["status"] == "completed" for entry in entries)
    for entry in entries:
        assert (entry["method"], entry["target"]) == ("KSIVI", "x_shaped")
        assert entry["config_hash"] == sweep.effective_config_hash(entry["config_path"], entry["seed"], OVERRIDES)
        validate_entry(entry)
    return sorted(completed_runs(entries), key=lambda record: record.seed)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.check_only:
        canonical_records()
        print("Validated all ten 50,000-step checkpoints, optimizers, configurations, and sample sets.")
        return
    manifest_args = argparse.Namespace(
        methods=["ksivi"], exclude_methods=[], targets=["x_shaped"], seeds=SEEDS,
        seed_override=[], campaign_slug=CAMPAIGN, results_dir=f"results/{CAMPAIGN}",
        tb_dir=f"tb_logs/{CAMPAIGN}", extra_override=OVERRIDES, limit=None,
    )
    entries = sweep.build_manifest_entries(manifest_args)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    for index, entry in enumerate(entries):
        entry.update(results_dir=f"results/{CAMPAIGN}/seed{entry['seed']}",
                     tb_dir=f"tb_logs/{CAMPAIGN}/seed{entry['seed']}",
                     batch=index // 3 + 1, source_revision=revision, concurrency_limit=3)
        entry["command_template"] = sweep.build_command(entry, 0, entry["results_dir"], entry["tb_dir"], OVERRIDES)
    if args.dry_run:
        for start in range(0, len(entries), 3):
            print(f"Batch {start // 3 + 1}: {[entry['seed'] for entry in entries[start:start + 3]]}")
        print(json.dumps(entries, indent=2))
        return

    import fcntl
    runtime = ROOT / "campaigns" / CAMPAIGN / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    lock = (runtime / "controller.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", PYTHONUNBUFFERED="1")
    torch.set_num_threads(1)
    manifest_path = ROOT / MANIFEST
    previous = {entry["run_id"]: entry for entry in sweep.read_json_or_default(manifest_path, [])}
    for entry in entries:
        old = previous.get(entry["run_id"])
        if old and (old["status"] == "completed" or old.get("exit_code") == 0):
            assert old["config_hash"] == entry["config_hash"], "Completed run has a different configuration"
            validate_entry(old)
            entry.update(old)
            entry.update(status="completed", failure_reason=None)
            if "training_time_sec" not in entry:
                times = re.findall(r"Training completed\. Total time: ([0-9.]+)s",
                                   (sweep.repo_path(entry["result_path"]) / "run.log").read_text())
                assert times
                entry.update(training_time_sec=float(times[-1]), iterations=50000)

    console_root = runtime / "console"
    console_root.mkdir(exist_ok=True)
    active = {}

    def save_state() -> None:
        sweep.write_json(manifest_path, entries)
        sweep.write_manifest_csv(manifest_path.with_suffix(".csv"), entries, {entry["run_id"]: entry for entry in entries})
        sweep.write_json(runtime / "current.json", dict(
            updated_at=sweep.utc_now(), max_concurrent_seeds=3,
            completed=[entry["seed"] for entry in entries if entry["status"] == "completed"],
            running=[run.entry["seed"] for run in active.values()],
            failed=[entry["seed"] for entry in entries if entry["status"] == "failed"],
        ))

    save_state()
    try:
        for start in range(0, len(entries), 3):
            batch = [entry for entry in entries[start:start + 3] if entry["status"] != "completed"]
            if not batch:
                continue
            print(f"Starting batch {start // 3 + 1}: {[entry['seed'] for entry in batch]}", flush=True)
            for entry in batch:
                run = sweep.launch_run(entry, 0, console_root)
                active[entry["run_id"]] = run
                entry.update(status="running", started_at=run.started_at, pid=run.process_pid, runtime_gpu=0)
                sweep.append_event(runtime / "events.jsonl", dict(status="started", seed=entry["seed"],
                                   batch=entry["batch"], pid=run.process_pid, started_at=run.started_at))
                save_state()
                time.sleep(2)
            while active:
                for run_id, run in list(active.items()):
                    sweep.poll_process_output(run)
                    if run.process.poll() is None:
                        continue
                    event = sweep.complete_process(run)
                    entry = run.entry
                    entry.update(event, status=event["run_status"])
                    if entry["status"] == "completed":
                        try:
                            validate_entry(entry)
                            times = re.findall(r"Training completed\. Total time: ([0-9.]+)s",
                                               (sweep.repo_path(entry["result_path"]) / "run.log").read_text())
                            assert times
                            entry["training_time_sec"] = float(times[-1])
                            entry["iterations"] = 50000
                        except Exception as exc:
                            entry.update(status="failed", failure_reason=repr(exc))
                    sweep.append_event(runtime / "events.jsonl", dict(entry))
                    active.pop(run_id)
                    save_state()
                    print(f"Seed {entry['seed']}: {entry['status']} ({entry['duration_sec']:.1f}s)", flush=True)
                if active:
                    time.sleep(2)
            assert all(entry["status"] == "completed" for entry in batch), "Batch failed; inspect runtime/console logs"
    finally:
        for run in active.values():
            run.process.terminate()
        for run in active.values():
            run.process.wait()
            run.console_fh.close()
            run.entry.update(status="failed", failure_reason="controller interrupted")
        active.clear()
        save_state()
        lock.close()
    report_dir = ROOT / "campaigns" / CAMPAIGN / "generated_reports"
    write_csv(report_dir / "summary.csv", [dict(
        run_id=entry["run_id"], seed=entry["seed"], method="KSIVI", target="x_shaped",
        wall_clock_sec=entry["duration_sec"], training_time_sec=entry["training_time_sec"], iterations=50000,
        result_path=entry["result_path"], source_revision=entry["source_revision"],
    ) for entry in entries])
    canonical_records()
    print("All ten seeds completed and validated.", flush=True)


if __name__ == "__main__":
    main()
