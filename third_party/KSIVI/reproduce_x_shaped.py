"""Local CPU reproduction using the authors' SIMINet, target, and KSIVI runner.

Run from the parent workspace with its .venv Python. Statistical settings are
read from the unmodified configs/x_shaped.yml; only the four stated treatments
are applied. Raw experiment files and the generated report stay under results/.
"""
from __future__ import annotations

import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

from models.networks import SIMINet
from models.target_models import X_shaped
from sivistein_2d import KernelSIVI
from utils.kernels import gaussian_kernel

HERE = Path(__file__).resolve().parent
WORKSPACE = HERE.parent.parent
DEFAULT_ROOT = WORKSPACE / "results/ksivi_upstream_local_20261010"
TREATMENTS = {
    "canonical_cond": {"variance_mode": "conditional", "estimator": "conditional"},
    "canonical_global": {"variance_mode": "global", "estimator": "conditional"},
    "stein_cond": {"variance_mode": "conditional", "estimator": "stein",
                   "stein_detach_bandwidth": True},
    "floor02_cond": {"variance_mode": "conditional", "estimator": "conditional",
                     "variance_floor": 0.2},
}
LABELS = {
    "canonical_cond": "Conditional\noriginal loss",
    "canonical_global": "Global\noriginal loss",
    "stein_cond": "Conditional\nStein loss",
    "floor02_cond": "Conditional\nvariance floor 0.2",
}


def namespace(data):
    return argparse.Namespace(**{k: namespace(v) if isinstance(v, dict) else v
                                 for k, v in data.items()})


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".new")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def source_info():
    paths = ["models/networks.py", "sivistein_2d.py", "utils/kernels.py",
             "configs/x_shaped.yml", "reproduce_x_shaped.py"]
    return {
        "upstream": json.loads((HERE / "UPSTREAM.json").read_text())['commit'],
        "workspace_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=WORKSPACE, text=True).strip(),
        "source_sha256": {p: hashlib.sha256((HERE / p).read_bytes()).hexdigest() for p in paths},
        "python": sys.version, "torch": torch.__version__, "numpy": np.__version__,
        "platform": platform.platform(), "device": "cpu",
    }


def make_runner(condition, seed, batch=None):
    raw = yaml.safe_load((HERE / "configs/x_shaped.yml").read_text())
    raw["train"].update(TREATMENTS[condition])
    if batch is not None:
        raw["train"]["batchsize"] = batch
    config = namespace(raw)
    config.device = torch.device("cpu")
    torch.manual_seed(seed)
    runner = KernelSIVI(config)
    runner.target_model = X_shaped(config.device)
    runner.SemiVInet = SIMINet(config.train, config.device)
    return runner, raw


def optimizer_for(runner):
    cfg = runner.trainpara
    optimizer = torch.optim.Adam(runner.SemiVInet.parameters(), lr=cfg.lr_SIMI, betas=(.9, .99))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg.gamma_step, gamma=cfg.gamma)
    return optimizer, scheduler


def target_samples(n, seed=20261010):
    rng = np.random.default_rng(seed)
    z = rng.normal(size=(n, 2))
    branch = rng.choice([-1., 1.], n)
    a, b = math.sqrt(3.8) * z[:, 0], math.sqrt(.2) * z[:, 1]
    return np.stack([(a + b) / math.sqrt(2), branch * (a - b) / math.sqrt(2)], axis=1)


def sliced_w2(samples, reference):
    angles = np.arange(128) * np.pi / 128
    directions = np.stack([np.cos(angles), np.sin(angles)])
    a = np.sort(samples @ directions, axis=0)
    b = np.sort(reference @ directions, axis=0)
    return float(np.sqrt(np.mean((a - b)**2)))


def sample_and_diagnose(runner, seed, n=10000):
    # Diagnostics do not consume any part of the training random-number stream.
    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(300000 + seed)
        z = torch.randn(n, runner.trainpara.z_dim)
        x, neg_score = runner.SemiVInet(z)
        variance = runner.SemiVInet.getstd(z).square()
        if variance.ndim == 1:
            variance = variance.expand(n, -1)
        a = x.numpy().astype(np.float64)
        v = variance.numpy().astype(np.float64)
        if not np.isfinite(a).all() or not np.isfinite(v).all():
            raise FloatingPointError("Nonfinite generated samples or variances")
        metrics = {
            "sw2": sliced_w2(a, target_samples(n)),
            "mean": a.mean(0).tolist(), "covariance": np.cov(a.T).tolist(),
            "variance_mean": float(v.mean()), "variance_median": float(np.median(v)),
            "variance_q01": float(np.quantile(v, .01)), "variance_min": float(v.min()),
            "inverse_variance_mean": float((1 / v).mean()),
            "variance_floor": math.exp(runner.SemiVInet.log_var_min),
            "floor_fraction": float(np.mean(v <= math.exp(runner.SemiVInet.log_var_min) * 1.0001)),
            "component_score_second_moment": float(neg_score.square().sum(1).mean()),
            "outside_plot_fraction": float(np.mean(np.max(np.abs(a), axis=1) > 5)),
            "positive_diagonal_fraction": float(np.mean(a[:, 0] * a[:, 1] > 0)),
            "samples": n, "projection_directions": 128,
        }
        return x, metrics


def worker(args):
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    condition, seed = args.conditions[0], args.seeds[0]
    runner, config = make_runner(condition, seed, args.batch)
    total = args.steps or runner.num_iters
    directory = args.root / f"{condition}_s{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    specification = {"condition": condition, "seed": seed, "steps": total,
                     "threads": args.threads, "config": config, "source": source_info()}
    optimizer, scheduler = optimizer_for(runner)
    first_step, elapsed_before, trajectory = 1, 0., []
    if (directory / 'checkpoint.pt').exists():
        previous = json.loads((directory / 'specification.json').read_text())
        for key in ['condition', 'seed', 'steps', 'threads', 'config']:
            if previous[key] != specification[key]:
                raise RuntimeError(f'Checkpoint specification mismatch: {key}')
        if previous['source']['source_sha256'] != specification['source']['source_sha256']:
            raise RuntimeError('Checkpoint was generated by different source code')
        checkpoint = torch.load(directory / 'checkpoint.pt', map_location='cpu', weights_only=True)
        runner.SemiVInet.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        torch.random.set_rng_state(checkpoint['rng_state'])
        first_step = checkpoint['step'] + 1
        elapsed_before = checkpoint['elapsed_seconds']
        trajectory = checkpoint['trajectory']
        print(f'Resuming at step {first_step}', flush=True)
    else:
        x, metrics = sample_and_diagnose(runner, seed, n=2048)
        trajectory.append({"step": 0, "lr": optimizer.param_groups[0]['lr'], **metrics})
    write_json(directory / "specification.json", specification)
    start = time.perf_counter()
    last_loss = None
    for step in range(first_step, total + 1):
        loss, h = runner.training_loss(step)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"{condition} seed {seed}: nonfinite loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if step % 1000 == 0 or step == total:
            last_loss = float(loss.detach())
            progress = {"condition": condition, "seed": seed, "step": step, "total": total,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - start, "loss": last_loss,
                        "bandwidth": float(h.detach()) if torch.is_tensor(h) else h,
                        "lr": optimizer.param_groups[0]['lr'], "pid": os.getpid()}
            write_json(directory / "progress.json", progress)
            print(json.dumps(progress), flush=True)
        if step % 5000 == 0 or step == total:
            x, metrics = sample_and_diagnose(runner, seed, n=2048)
            trajectory.append({"step": step, "lr": optimizer.param_groups[0]['lr'], **metrics})
            write_json(directory / "trajectory.json", trajectory)
            checkpoint = directory / 'checkpoint.pt.new'
            torch.save({"model": runner.SemiVInet.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(), "step": step,
                        "rng_state": torch.random.get_rng_state(), "trajectory": trajectory,
                        "elapsed_seconds": elapsed_before + time.perf_counter() - start}, checkpoint)
            checkpoint.replace(directory / 'checkpoint.pt')
    x, metrics = sample_and_diagnose(runner, seed, n=10000)
    np.save(directory / "final_samples.npy", x.numpy())
    result = {"condition": condition, "seed": seed, "steps": total, "status": "completed",
              "elapsed_seconds": elapsed_before + time.perf_counter() - start, "final_loss": last_loss,
              "metrics": metrics, "specification": specification}
    write_json(directory / "result.json", result)
    print(json.dumps({"completed": condition, "seed": seed, "sw2": metrics['sw2']}), flush=True)


def check(args):
    torch.set_num_threads(1)
    results = {}
    # Execute the pristine model from the preceding import commit, not a second
    # hand-written model that could repeat the implementation's mistakes.
    vendor_commit = subprocess.check_output(
        ['git', 'log', '--diff-filter=A', '--format=%H', '--', 'third_party/KSIVI/models/networks.py'],
        cwd=WORKSPACE, text=True).splitlines()[-1]
    baseline = subprocess.check_output(
        ['git', 'show', f'{vendor_commit}:third_party/KSIVI/models/networks.py'], cwd=WORKSPACE, text=True)
    scope = {}
    exec(compile(baseline, '<pristine-upstream-networks>', 'exec'), scope)
    current, _ = make_runner('canonical_global', 42, 17)
    config = current.trainpara
    torch.manual_seed(42)
    pristine = scope['SIMINet'](config, torch.device('cpu'))
    for key, value in pristine.state_dict().items():
        torch.testing.assert_close(value, current.SemiVInet.state_dict()[key], rtol=0, atol=0)
    torch.manual_seed(99)
    z = torch.randn(17, 3)
    state = torch.random.get_rng_state()
    reference = pristine(z)
    torch.random.set_rng_state(state)
    actual = current.SemiVInet(z)
    for a, b in zip(reference, actual):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    results['global_model_matches_pristine_bytes_and_forward'] = True
    results['vendor_commit'] = vendor_commit

    global_runner, _ = make_runner('canonical_global', 43, 17)
    global_rng = torch.random.get_rng_state()
    conditional_runner, _ = make_runner('canonical_cond', 43, 17)
    assert torch.equal(global_rng, torch.random.get_rng_state())
    for p, q in zip(global_runner.SemiVInet.mu.parameters(), conditional_runner.SemiVInet.mu.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    torch.testing.assert_close(conditional_runner.SemiVInet.getstd(z), torch.ones(17, 2), rtol=0, atol=0)
    results['paired_initial_mean_variance_and_training_rng'] = True

    # Compare every parameter gradient to the upstream expression on identical
    # samples. The final reproduction uses the untouched upstream kernel.
    a, _ = make_runner('canonical_global', 44, 37)
    z1, z2 = torch.randn(37, 3), torch.randn(37, 3)
    x, b = a.SemiVInet(z1)
    y, c = a.SemiVInet(z2)
    reference = (((a.target_model.score(x) + b) @ (a.target_model.score(y) + c).T)
                 * gaussian_kernel(x, y, detach=False)).mean()
    parameters = list(a.SemiVInet.parameters())
    rg = torch.autograd.grad(reference, parameters, retain_graph=True)
    actual, _ = a.loss_from_samples(x, b, y, c)
    ag = torch.autograd.grad(actual, parameters)
    for p, q in zip(rg, ag):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    results['original_loss_refactor_exact'] = True

    # Derive the Stein kernel by automatic spatial differentiation of k rather
    # than repeating the closed-form expression used for training.
    a, _ = make_runner('stein_cond', 42, 2)
    a.trainpara.fixed_bandwidth = .75
    x = torch.tensor([[.3, -.8], [1.1, .2]], requires_grad=True)
    y = torch.tensor([[-.4, .6], [.7, -.1]], requires_grad=True)
    pair_values = []
    for i in range(2):
        for j in range(2):
            k = torch.exp(-(x[i] - y[j]).square().sum() / (2 * .75**2))
            kx, ky = torch.autograd.grad(k, (x, y), create_graph=True)
            trace = sum(torch.autograd.grad(kx[i, d], y, create_graph=True)[0][j, d] for d in range(2))
            sx, sy = a.target_model.score(x[i:i+1])[0], a.target_model.score(y[j:j+1])[0]
            pair_values.append(k * (sx * sy).sum() + (sx * ky[j]).sum() + (sy * kx[i]).sum() + trace)
    reference = torch.stack(pair_values).mean()
    actual, _ = a.loss_from_samples(x, torch.zeros_like(x), y, torch.zeros_like(y))
    torch.testing.assert_close(reference, actual, rtol=1e-5, atol=1e-6)
    rg = torch.autograd.grad(reference, (x, y), retain_graph=True)
    ag = torch.autograd.grad(actual, (x, y))
    for p, q in zip(rg, ag):
        torch.testing.assert_close(p, q, rtol=2e-5, atol=3e-6)
    results['stein_value_and_gradient_match_spatial_autodiff'] = True

    a, _ = make_runner('floor02_cond', 42, 17)
    with torch.no_grad():
        a.SemiVInet.variance_head.bias.fill_(-100)
    torch.testing.assert_close(a.SemiVInet.getstd(z).square(), torch.full((17, 2), .2), atol=3e-8, rtol=0)
    before = torch.random.get_rng_state()
    sample_and_diagnose(a, 42, 512)
    assert torch.equal(before, torch.random.get_rng_state())
    results['variance_floor_and_isolated_diagnostic_rng'] = True
    results['status'] = 'passed'
    write_json(args.root / 'checks.json', results)
    print(json.dumps(results, indent=2))


def benchmark(args):
    records = []
    for condition in ['canonical_global', 'stein_cond']:
        for threads in [1, 2, 4]:
            torch.set_num_threads(threads)
            runner, _ = make_runner(condition, 42, args.batch)
            optimizer, scheduler = optimizer_for(runner)
            start = time.perf_counter()
            for step in range(1, 201):
                loss, _ = runner.training_loss(step)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                scheduler.step()
            seconds = time.perf_counter() - start
            record = {'condition': condition, 'backend': 'upstream', 'threads': threads,
                      'steps': 200, 'seconds': seconds, 'ms_per_step': seconds * 5}
            records.append(record)
            print(json.dumps(record), flush=True)
    write_json(args.root / 'benchmark.json', records)


def run(args):
    args.root.mkdir(parents=True, exist_ok=True)
    queue = [(name, seed) for seed in args.seeds for name in args.conditions]
    active, completed = [], []
    for name, seed in queue.copy():
        existing = args.root / f'{name}_s{seed}' / 'result.json'
        if existing.exists():
            result = json.loads(existing.read_text())
            expected = args.steps or 50000
            if result['status'] == 'completed' and result['steps'] == expected:
                _, expected_config = make_runner(name, seed, args.batch)
                if result['specification']['config'] != expected_config:
                    raise RuntimeError(f'Refusing to reuse results from a different configuration: {existing}')
                if result['specification']['source']['source_sha256'] != source_info()['source_sha256']:
                    raise RuntimeError(f'Refusing to reuse results from different code: {existing}')
                completed.append((name, seed))
                queue.remove((name, seed))
    start, last_report = time.perf_counter(), 0.
    while queue or active:
        while queue and len(active) < args.jobs:
            name, seed = queue.pop(0)
            directory = args.root / f'{name}_s{seed}'
            directory.mkdir(parents=True, exist_ok=True)
            log = (directory / 'console.log').open('w', encoding='utf-8')
            command = [sys.executable, '-W', 'ignore::SyntaxWarning', str(Path(__file__).resolve()),
                       'worker', '--root', str(args.root), '--conditions', name, '--seeds', str(seed),
                       '--threads', str(args.threads)]
            if args.steps:
                command += ['--steps', str(args.steps)]
            if args.batch:
                command += ['--batch', str(args.batch)]
            process = subprocess.Popen(command, cwd=WORKSPACE, stdout=log, stderr=subprocess.STDOUT)
            active.append((process, log, name, seed))
            print(f'Started {name} seed {seed}, PID {process.pid}', flush=True)
        for job in active.copy():
            process, log, name, seed = job
            if process.poll() is not None:
                log.close()
                active.remove(job)
                if process.returncode != 0:
                    for other, other_log, _, _ in active:
                        other.terminate()
                        other.wait()
                        other_log.close()
                    raise RuntimeError(f'{name} seed {seed} failed; inspect {args.root / f"{name}_s{seed}" / "console.log"}')
                completed.append((name, seed))
                result = json.loads((args.root / f'{name}_s{seed}' / 'result.json').read_text())
                print(f'Completed {name} seed {seed}: SW2={result["metrics"]["sw2"]:.4f}', flush=True)
        now = time.perf_counter()
        if now - last_report >= 30:
            progress = []
            for _, _, name, seed in active:
                path = args.root / f'{name}_s{seed}' / 'progress.json'
                if path.exists():
                    p = json.loads(path.read_text())
                    progress.append(f'{name}/{seed}:{p["step"]}')
            print(json.dumps({'completed': len(completed), 'active': progress, 'queued': len(queue),
                              'elapsed_seconds': round(now-start)}), flush=True)
            last_report = now
        time.sleep(1)
    report(args)


def report(args):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    plt.rcParams.update({'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
                         'font.size': 12, 'axes.spines.top': False, 'axes.spines.right': False})
    output = args.root / 'report'
    output.mkdir(parents=True, exist_ok=True)
    rows = [json.loads((args.root / f'{name}_s{seed}' / 'result.json').read_text())
            for seed in args.seeds for name in args.conditions]
    samples = {(r['condition'], r['seed']): np.load(args.root / f"{r['condition']}_s{r['seed']}" / 'final_samples.npy')
               for r in rows}
    grid = np.linspace(-5, 5, 180)
    xx, yy = np.meshgrid(grid, grid)
    positions = torch.tensor(np.stack([xx.ravel(), yy.ravel()], axis=1), dtype=torch.float32)
    with torch.no_grad():
        density = X_shaped('cpu').logp(positions).exp().numpy().reshape(xx.shape)
    metrics = {(r['condition'], r['seed']): r for r in rows}

    def panel(ax, name, seed):
        x = samples[(name, seed)]
        collection = ax.hexbin(x[:,0], x[:,1], gridsize=60, extent=(-5,5,-5,5),
                               mincnt=1, cmap='Blues', norm=LogNorm(vmin=1, vmax=180))
        ax.contour(xx, yy, density, levels=[.005,.02,.07,.14], colors='#333333', linewidths=.7)
        ax.set(xlim=(-5,5), ylim=(-5,5), aspect='equal')
        ax.text(.025,.975,f"Seed {seed}\nSW$_2$={metrics[(name,seed)]['metrics']['sw2']:.3f}",
                transform=ax.transAxes, va='top', fontsize=12)
        return collection

    figure, axes = plt.subplots(len(args.seeds), len(args.conditions),
                                figsize=(3.6*len(args.conditions),3.6*len(args.seeds)),
                                squeeze=False, layout='constrained')
    for i, seed in enumerate(args.seeds):
        for j, name in enumerate(args.conditions):
            panel(axes[i,j], name, seed)
            if i == 0:
                axes[i,j].set_title(LABELS[name], fontsize=15)
            if j == 0:
                axes[i,j].set_ylabel('$x_2$')
            if i == len(args.seeds)-1:
                axes[i,j].set_xlabel('$x_1$')
    figure.savefig(output / 'final_samples.png', dpi=220)
    plt.close(figure)
    for seed in args.seeds:
        for name in args.conditions:
            figure, ax = plt.subplots(figsize=(4.8,4.8),layout='constrained')
            panel(ax, name, seed)
            ax.set(title=LABELS[name].replace('\n', ': '), xlabel='$x_1$', ylabel='$x_2$')
            figure.savefig(output / f'{name}_s{seed}.png', dpi=200)
            plt.close(figure)

    summaries = {}
    for name in args.conditions:
        selected = [r for r in rows if r['condition'] == name]
        values = [r['metrics']['sw2'] for r in selected]
        summaries[name] = {'n':len(values), 'sw2_mean':float(np.mean(values)),
                           'sw2_sd':float(np.std(values,ddof=1)) if len(values)>1 else None,
                           'sw2_by_seed':{str(r['seed']):r['metrics']['sw2'] for r in selected}}
    manifest = {'status':'completed', 'runs':len(rows), 'source':source_info(),
                'canonical_config':yaml.safe_load((HERE/'configs/x_shaped.yml').read_text()),
                'treatments':TREATMENTS, 'summary':summaries,
                'figures':{'combined':'final_samples.png', 'individual':[
                    f'{name}_s{seed}.png' for seed in args.seeds for name in args.conditions]},
                'notes':[
                    'Local CPU runs; no remote execution.',
                    'Original width-50 ReLU architecture, exponential variance, batch 500, no annealing, Adam and StepLR.',
                    'Conditional variance shares the original two hidden layers; its head starts at constant variance 1.',
                    'All methods have identical initial means, variances, and training RNG streams for each seed.',
                    'Original estimator differentiates median bandwidth; Stein variant detaches it, as in the earlier investigation.',
                    'Default variance floor is exp(-20), including conditional/Stein; the floor treatment uses variance 0.2.',
                    'Plot window is [-5,5]^2; SW2 includes all 10000 samples, including tails outside the plot.',
                    'SW2 is root mean squared projected quantile distance over 128 directions, not full 2D Wasserstein distance.',
                    'These are independent upstream-default experiments; they do not reuse the earlier remote samples.',
                ], 'results':rows}
    for name in ['checks.json','benchmark.json']:
        if (args.root/name).exists():
            manifest[name.removesuffix('.json')] = json.loads((args.root/name).read_text())
    write_json(output/'manifest.json',manifest)
    with (output/'metrics.csv').open('w',newline='',encoding='utf-8') as f:
        fields=['condition','seed','steps','sw2','variance_mean','variance_median','variance_q01',
                'inverse_variance_mean','floor_fraction','outside_plot_fraction','positive_diagonal_fraction']
        writer=csv.DictWriter(f,fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow({key:r[key] if key in r else r['metrics'][key] for key in fields})
    print(json.dumps(summaries,indent=2),flush=True)
    print(f'Report saved to {output}',flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['check','benchmark','run','worker','report'])
    parser.add_argument('--root',type=Path,default=DEFAULT_ROOT)
    parser.add_argument('--conditions',nargs='+',choices=list(TREATMENTS),default=list(TREATMENTS))
    parser.add_argument('--seeds',nargs='+',type=int,default=[42,43,44])
    parser.add_argument('--steps',type=int)
    parser.add_argument('--batch',type=int)
    parser.add_argument('--threads',type=int,default=1)
    parser.add_argument('--jobs',type=int,default=6)
    args=parser.parse_args()
    args.root=args.root.resolve()
    if args.jobs < 1 or args.threads < 1 or (args.steps is not None and args.steps < 1):
        parser.error('jobs, threads, and steps must be positive')
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.conditions)) != len(args.conditions):
        parser.error('seeds and conditions must be unique')
    if args.action == 'worker' and (len(args.conditions)!=1 or len(args.seeds)!=1):
        parser.error('worker requires one condition and one seed')
    globals()[args.action](args)


if __name__ == '__main__':
    main()
