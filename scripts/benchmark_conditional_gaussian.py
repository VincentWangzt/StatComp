"""Time the production KSIVI loop before and after the conditional-head split."""

import argparse
import json
import logging
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from finalization.runner_eval import remove_file_handlers
from models.vi_model import ConditionalGaussian
from runner.base_runner import VIModel
from runner.ksivi import KSIVIRunner
from utils.logging import get_logger


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--baseline', default='06d26f0^')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.steps < 5000 or args.repeats < 1:
        parser.error('Use 1--4999 steps and at least one repeat.')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA is unavailable.')
    torch.set_num_threads(1)
    get_logger().setLevel(logging.ERROR)
    args.output = args.output.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    baseline_source = subprocess.check_output(
        ['git', 'show', f'{args.baseline}:models/vi_model.py'], cwd=ROOT, text=True)
    baseline = types.ModuleType('_benchmark_baseline_vi_model')
    exec(compile(baseline_source, f'{args.baseline}:models/vi_model.py', 'exec'),
         baseline.__dict__)
    model_types = {'before': baseline.ConditionalGaussian, 'after': ConditionalGaussian}
    trials = []
    for repeat in range(args.repeats):
        order = ('before', 'after') if repeat % 2 == 0 else ('after', 'before')
        for label in order:
            config = OmegaConf.load(ROOT / 'configs/ksivi_x_shaped.yaml')
            config.config_path = str(ROOT / 'configs/ksivi_x_shaped.yaml')
            config.device = args.device
            config.use_cuda = args.device == 'cuda'
            config.train.epochs = args.steps
            # Equal LRs isolate implementation overhead from different trajectories.
            config.train.vi.var_lr = config.train.vi.lr
            config.vi_model.variance_init = None
            config.train.log.metric_log_freq = 0
            config.train.plot.freq = 1000000000
            trial_name = f'{args.device}_{label}_{repeat}'
            config.output = dict(
                results_dir=str(args.output.parent / trial_name),
                tb_dir=str(ROOT / 'tb_logs/conditional_gaussian_efficiency_20261015' / trial_name))
            torch.manual_seed(42)
            with patch.dict(VIModel, {'ConditionalGaussian': model_types[label]}):
                runner = KSIVIRunner(config)
            try:
                runner.log_config()
                for step in range(1, 21):
                    runner._compute_loss_and_step(step)
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                start = time.perf_counter()
                runner.learn()
                if args.device == 'cuda':
                    torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                trials.append(dict(version=label, repeat=repeat, seconds=elapsed,
                                   updates_per_second=args.steps / elapsed))
            finally:
                runner.writer.close()
                remove_file_handlers()
                del runner

    medians = {label: statistics.median(t['seconds'] for t in trials if t['version'] == label)
               for label in model_types}
    report = dict(
        host=platform.node(), python=platform.python_version(), torch=torch.__version__,
        device=args.device, gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        cpu_threads=1, timed_steps_per_trial=args.steps, repeats=args.repeats,
        baseline_revision=subprocess.check_output(
            ['git', 'rev-parse', args.baseline], cwd=ROOT, text=True).strip(),
        current_revision=subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        before_updates_per_second=args.steps / medians['before'],
        after_updates_per_second=args.steps / medians['after'],
        slowdown_percent=100 * (medians['after'] / medians['before'] - 1), trials=trials)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
