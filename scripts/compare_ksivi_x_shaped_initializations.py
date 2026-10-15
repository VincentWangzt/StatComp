"""Compare saved 50K-step KSIVI runs with random and constant variance init."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch

from finalization.artifacts import find_final_checkpoint, find_final_samples, load_sample_z
from finalization.plots import _draw_toy_contours, _scatter_generator, _take_points
from models.target_models import X_shaped


def comparable_config(config):
    result = OmegaConf.to_container(config, resolve=True)
    result.pop('output')
    result['vi_model'].pop('variance_init', None)
    result['cuda_visible_devices'] = str(result['cuda_visible_devices'])
    return result


def load_run(root, seed, variant, reference, output):
    paths = list((root / f'seed{seed}' / 'KSIVI/x_shaped').glob('*/full_config.yaml'))
    assert len(paths) == 1, paths
    config = OmegaConf.load(paths[0])
    run = paths[0].parent
    assert config.seed == seed and config.device == 'cuda'
    assert config.train.epochs == 50000 and config.train.vi.lr == 1e-3
    assert config.train.vi.var_lr == 1e-5
    expected_init = None if variant == 'random' else 1.313
    assert config.vi_model.get('variance_init') == expected_init
    sample_path, epoch = find_final_samples(run)
    checkpoint, checkpoint_epoch = find_final_checkpoint(run)
    assert epoch == checkpoint_epoch == 50000
    samples = load_sample_z(sample_path)
    assert samples.shape == (10000, 2) and torch.isfinite(samples).all()
    weights = torch.load(checkpoint / 'vi_model.pt', map_location='cpu', weights_only=True)
    assert all(torch.isfinite(value).all() for value in weights.values())
    optimizer = torch.load(checkpoint / 'vi_optim.pt', map_location='cpu', weights_only=True)
    steps = [int(state['step']) for state in optimizer['state'].values()]
    assert min(steps) == max(steps) == 50000
    angles = torch.linspace(0, math.pi, 129)[:-1]
    directions = torch.stack((angles.cos(), angles.sin()))
    error = (torch.sort(samples @ directions, dim=0).values
             - torch.sort(reference @ directions, dim=0).values).square().mean().sqrt().item()
    timing = re.findall(r'Training completed\. Total time: ([0-9.]+)s',
                        (run / 'run.log').read_text(errors='replace'))
    assert timing, run
    shutil.copyfile(paths[0], output / f'{variant}_seed_{seed}_config.yaml')
    record = dict(variant=variant, seed=seed, initial_variance=expected_init,
                  optimizer_steps=min(steps), mean_lr=1e-3, variance_lr=1e-5,
                  training_seconds=float(timing[-1]), result_path=str(run),
                  source_revision=(root / 'runtime/source_revision').read_text().strip(),
                  sample_sha256=hashlib.sha256(sample_path.read_bytes()).hexdigest(),
                  samples_saved=len(samples), samples_plotted=2000,
                  finite_parameters_and_samples=True, sw2_10k_128_directions=error,
                  positive_diagonal_fraction=((samples[:, 0] * samples[:, 1]) > 0).float().mean().item())
    points = _take_points(samples, 2000, generator=_scatter_generator(42, 'x_shaped', 'KSIVI'))
    return record, config, points


def draw(records, configs, points, seeds, variants, path):
    fig, axes = plt.subplots(len(variants), len(seeds),
                             figsize=(3.6 * len(seeds), 3.5 * len(variants) + .5),
                             squeeze=False, sharex=True, sharey=True)
    for row, variant in enumerate(variants):
        for column, seed in enumerate(seeds):
            key = (variant, seed)
            ax = axes[row, column]
            _draw_toy_contours(ax, 'x_shaped', list(configs[key].target.bbox))
            x = points[key]
            ax.plot(x[:, 0], x[:, 1], '.', color='#ff7f0e', markersize=3, alpha=.45)
            if row == 0:
                ax.set_title(f'Seed {seed}', fontsize=14)
            ax.text(.04, .95, f"SW2 = {records[key]['sw2_10k_128_directions']:.3f}",
                    transform=ax.transAxes, va='top', fontsize=10,
                    bbox=dict(facecolor='white', edgecolor='none', alpha=.85))
            if row == len(variants) - 1:
                ax.set_xlabel('$z_1$', fontsize=12)
            if column == 0:
                label = 'Random initial variance' if variant == 'random' else 'Constant initial variance: 1.313'
                ax.set_ylabel(label + '\n$z_2$', fontsize=11)
            ax.tick_params(labelsize=10)
    fig.suptitle('KSIVI on x_shaped | RTX 4090 | 50,000 updates\n'
                 'Variance-head LR $10^{-5}$; 2,000 plotted samples per panel', fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, .92 if len(variants) == 1 else .95))
    fig.savefig(path, dpi=220, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--random-root', type=Path, required=True)
    parser.add_argument('--constant-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[43, 44, 45])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(20261015)
        reference = X_shaped('cpu').sample(10000)
    records, configs, points = {}, {}, {}
    for variant, root in (('random', args.random_root), ('constant', args.constant_root)):
        for seed in args.seeds:
            key = (variant, seed)
            records[key], configs[key], points[key] = load_run(root, seed, variant, reference, args.output)
    for seed in args.seeds:
        assert comparable_config(configs['random', seed]) == comparable_config(configs['constant', seed]), seed
    draw(records, configs, points, args.seeds, ['constant'], args.output / 'final_samples.png')
    draw(records, configs, points, args.seeds, ['random', 'constant'], args.output / 'initialization_comparison.png')
    for seed in args.seeds:
        draw(records, configs, points, [seed], ['constant'], args.output / f'seed_{seed}.png')
    summaries = {}
    for variant in ('random', 'constant'):
        errors = [records[variant, seed]['sw2_10k_128_directions'] for seed in args.seeds]
        summaries[variant] = dict(mean_sw2=statistics.mean(errors),
                                  sd_sw2=statistics.stdev(errors) if len(errors) > 1 else None)
    report = dict(runs=list(records.values()), summary=summaries,
                  metric='Root mean squared projected quantile difference; 10K samples, 128 evenly spaced directions',
                  reference_seed=20261015, scatter_selection_seed=42,
                  configs_equal_except_variance_init_and_output=True)
    (args.output / 'runs.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
