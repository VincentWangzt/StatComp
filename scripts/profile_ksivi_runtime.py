"""Separate KSIVI training-loop, scalar logging, and CUDA latency costs."""

import argparse
import json
import logging
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from omegaconf import OmegaConf

from finalization.runner_eval import remove_file_handlers
from runner.ksivi import KSIVIRunner
from utils.logging import get_logger


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=('cpu', 'cuda'), required=True)
    parser.add_argument('--steps', type=int, default=200)
    parser.add_argument('--repeats', type=int, default=2)
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

    def synchronize():
        if args.device == 'cuda':
            torch.cuda.synchronize()

    def create_runner(label, steps):
        config = OmegaConf.load(ROOT / 'configs/ksivi_x_shaped.yaml')
        config.config_path = str(ROOT / 'configs/ksivi_x_shaped.yaml')
        config.device = args.device
        config.use_cuda = args.device == 'cuda'
        config.train.epochs = steps
        config.train.log.metric_log_freq = 0
        config.train.plot.freq = 1000000000
        config.output = dict(
            results_dir=str(args.output.parent / f'{args.device}_{label}'),
            tb_dir=str(ROOT / 'tb_logs/ksivi_runtime_profile_20261015' / f'{args.device}_{label}'))
        torch.manual_seed(42)
        runner = KSIVIRunner(config)
        runner.log_config()
        for step in range(1, 21):
            runner._compute_loss_and_step(step)
        synchronize()
        return runner

    trials = []
    reference_state = None
    max_state_difference = 0.0
    for repeat in range(args.repeats):
        modes = ('full_loop', 'null_writer', 'step_only')
        for mode in modes if repeat % 2 == 0 else modes[::-1]:
            runner = create_runner(f'{mode}_{repeat}', args.steps)
            try:
                if mode == 'null_writer':
                    runner.writer.close()
                    runner.writer = NullWriter()
                start = time.perf_counter()
                if mode == 'step_only':
                    for step in range(1, args.steps + 1):
                        runner._compute_loss_and_step(step)
                else:
                    runner.learn()
                synchronize()
                elapsed = time.perf_counter() - start
                state = {key: value.detach().cpu().clone()
                         for key, value in runner.vi_model.state_dict().items()}
                if reference_state is None:
                    reference_state = state
                max_state_difference = max(max_state_difference, max(
                    (value - reference_state[key]).abs().max().item()
                    for key, value in state.items()))
                trials.append(dict(mode=mode, repeat=repeat, seconds=elapsed,
                                   updates_per_second=args.steps / elapsed))
            finally:
                runner.writer.close()
                remove_file_handlers()

    profile_steps = min(20, args.steps)
    runner = create_runner('profiler', profile_steps)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.device == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    try:
        with torch.profiler.profile(activities=activities) as profile:
            runner.learn()
            synchronize()
        gpu_events = [event for event in profile.events()
                      if event.device_type == torch.autograd.DeviceType.CUDA]
        kernel_us_per_step = sum(event.self_device_time_total for event in gpu_events) / profile_steps
        operators = [dict(name=event.key, calls_per_step=event.count / profile_steps,
                          self_cpu_us_per_step=event.self_cpu_time_total / profile_steps,
                          total_cpu_us_per_step=event.cpu_time_total / profile_steps,
                          total_device_us_per_step=event.device_time_total / profile_steps)
                     for event in profile.key_averages()]
        operators.sort(key=lambda event: event['self_cpu_us_per_step'], reverse=True)
    finally:
        runner.writer.close()
        remove_file_handlers()

    latency = {}
    if args.device == 'cuda':
        scalar = torch.ones(1, device='cuda')
        operations = {
            'scalar_item': lambda: scalar.item(),
            'add_async': lambda: scalar.add_(1),
            'add_then_synchronize': lambda: (scalar.add_(1), torch.cuda.synchronize()),
        }
        for name, operation in operations.items():
            for _ in range(20):
                operation()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(1000):
                operation()
            torch.cuda.synchronize()
            latency[name + '_us'] = 1e6 * (time.perf_counter() - start) / 1000

    report = dict(
        host=platform.node(), device=args.device, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
        steps=args.steps, repeats=args.repeats, max_parameter_difference_between_modes=max_state_difference,
        median_updates_per_second={mode: statistics.median(
            trial['updates_per_second'] for trial in trials if trial['mode'] == mode)
            for mode in ('full_loop', 'null_writer', 'step_only')},
        profiler_steps=profile_steps, cuda_events_per_step=len(gpu_events) / profile_steps,
        summed_cuda_event_us_per_step=kernel_us_per_step,
        top_cpu_operators=operators[:25],
        scalar_and_cuda_operators=[event for event in operators if any(
            key in event['name'].lower() for key in ('cuda', 'item', 'local_scalar_dense'))],
        scalar_latency=latency, trials=trials)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
