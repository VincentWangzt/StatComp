"""Run two sequential rounds of three parallel canonical KSIVI experiments."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from omegaconf import OmegaConf

CAMPAIGN_DIR = Path(__file__).resolve().parent
REPO = CAMPAIGN_DIR.parents[1]
CANONICAL = REPO / "configs/ksivi_x_shaped.yaml"


def specifications(results_root: Path, tb_root: Path,
                   variance_init: float | None = None) -> dict[str, list[dict]]:
    rounds = {
        "seeds": [(f"seed{seed}", f"Seed {seed}", seed, 0.001) for seed in (42, 43, 44)],
        "learning_rates": [("lr5e-4", "LR 5e-4 | seed 43", 43, 5e-4),
                           ("lr2e-3", "LR 2e-3 | seed 43", 43, 2e-3),
                           ("lr2e-4", "LR 2e-4 | seed 43", 43, 2e-4)],
    }
    plans = {}
    for name, entries in rounds.items():
        plans[name] = []
        for key, label, seed, lr in entries:
            row_results, row_tb = results_root / name / key, tb_root / name / key
            command = [sys.executable, "-u", str(REPO / "src.py"), "--config", str(CANONICAL),
                       f"seed={seed}", "train.log.metric_log_freq=0",
                       "metric.kl_ite.enabled=false", "metric.w2.enabled=false",
                       "metric.elbo.enabled=false", f"output.results_dir={row_results}",
                       f"output.tb_dir={row_tb}"]
            if name == "learning_rates":
                command.append(f"train.vi.lr={lr}")
            if variance_init is not None:
                command.append(f"vi_model.variance_init={variance_init}")
            plans[name].append({"key": key, "label": label, "seed": seed, "lr": lr,
                                "variance_init": variance_init,
                                "results_root": str(row_results), "tb_root": str(row_tb),
                                "command": command})
    return plans


def save_state(path: Path, state: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def execute_rounds(plans: dict[str, list[dict]], results_root: Path,
                   report_root: Path, campaign_validator=None, round_finalizer=None) -> None:
    """Run each round in parallel, then validate and render its saved outputs."""
    for specs in plans.values():
        for spec in specs:
            if Path(spec["results_root"]).exists():
                raise FileExistsError(f"Run output already exists: {spec['results_root']}")
    if round_finalizer is None:
        from plot_evolution import finalize_round
        round_finalizer = finalize_round

    source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    runtime = results_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    state_path = runtime / "state.json"
    state = {"status": "running", "source_commit": source_commit,
             "round_order": list(plans), "rounds": {}, "started_at": time.time()}
    save_state(state_path, state)
    try:
        for name, specs in plans.items():
            state["current_round"] = name
            state["rounds"][name] = {"status": "running", "runs": specs}
            save_state(state_path, state)
            jobs = []
            for spec in specs:
                row_runtime = Path(spec["results_root"]) / "runtime"
                row_runtime.mkdir(parents=True)
                (row_runtime / "command.json").write_text(json.dumps(spec, indent=2) + "\n")
                (row_runtime / "source_commit.txt").write_text(source_commit + "\n")
                output = (row_runtime / "console.log").open("w")
                process = subprocess.Popen(spec["command"], cwd=REPO, stdout=output,
                                           stderr=subprocess.STDOUT)
                jobs.append((spec, process, output, row_runtime))
                print(f"Started {name}/{spec['key']} (pid {process.pid})", flush=True)
            remaining = set(range(len(jobs)))
            while remaining:
                for index in list(remaining):
                    spec, process, output, row_runtime = jobs[index]
                    code = process.poll()
                    if code is not None:
                        output.close()
                        (row_runtime / "exit_code.txt").write_text(str(code) + "\n")
                        spec["exit_code"] = code
                        remaining.remove(index)
                        print(f"Finished {name}/{spec['key']}: exit {code}", flush=True)
                        save_state(state_path, state)
                if remaining:
                    time.sleep(1)
            if any(spec["exit_code"] != 0 for spec in specs):
                raise RuntimeError(f"A training job failed in round {name}")
            state["rounds"][name]["status"] = "rendering"
            save_state(state_path, state)
            report = round_finalizer(specs, report_root / name, source_commit, CANONICAL, name)
            state["rounds"][name]["status"] = "completed"
            state["rounds"][name]["grid"] = str(report_root / name / report["grid"])
            save_state(state_path, state)
            print(f"Completed {name}: {report['grid']}", flush=True)
        if campaign_validator is not None:
            state["checks"] = campaign_validator(plans, report_root)
        state["status"] = "completed"
        state["finished_at"] = time.time()
        save_state(state_path, state)
        (report_root / "campaign.json").write_text(json.dumps(state, indent=2) + "\n")
    except Exception as error:
        state["status"] = "failed"
        state["error"] = repr(error)
        save_state(state_path, state)
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--tb-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=CAMPAIGN_DIR / "generated_reports")
    parser.add_argument("--rounds", nargs="+", choices=("seeds", "learning_rates"),
                        default=["seeds", "learning_rates"])
    parser.add_argument("--variance-init", type=float,
                        help="Constant initial conditional variance; remains trainable")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    canonical = OmegaConf.load(CANONICAL)
    assert canonical.vi_model_type == "ConditionalGaussian"
    assert canonical.train.epochs == 50000 and canonical.train.batch_size == 128
    assert canonical.train.vi.lr == 0.001 and canonical.train.annealing.enabled
    assert canonical.train.plot.freq == canonical.train.sample.freq == 5000
    assert canonical.train.plot.num == canonical.train.sample.num == 10000
    plans = specifications(args.results_root, args.tb_root, args.variance_init)
    plans = {name: plans[name] for name in dict.fromkeys(args.rounds)}
    if args.dry_run:
        print(json.dumps(plans, indent=2))
        return
    execute_rounds(plans, args.results_root, args.report_root)


if __name__ == "__main__":
    main()
