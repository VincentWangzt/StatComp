"""Reproducible variance-mechanism investigation, using production KSIVI steps.

All random diagnostic draws preserve the training RNG. New paired runs start
with identical mean functions and constant variance, including Adam state.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.vi_model import ConditionalGaussian, ConditionalGaussianGlobal
from models.target_models import X_shaped
from runner.ksivi import KSIVIRunner
from utils.annealing import annealing


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@contextmanager
def preserve_rng(seed=None):
    with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
        if seed is not None:
            torch.manual_seed(seed)
        yield


def model_from_config(cfg):
    mc = OmegaConf.create(OmegaConf.to_container(cfg.vi_model, resolve=True))
    mc.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cls = ConditionalGaussianGlobal if cfg.vi_model_type == 'ConditionalGaussianGlobal' else ConditionalGaussian
    return cls(mc).to(mc.device)


def moments(model, target, n=10000):
    with preserve_rng(20261006), torch.no_grad():
        eps = model.sample_epsilon(n)
        x, neg = model(eps)
        mu, std = model.getmu(eps), model.getstd(eps)
        if std.ndim == 1:
            std = std.expand_as(mu)
        v = std.square()
        t = (x[:, 0] + x[:, 1]) / math.sqrt(2)
        u = (x[:, 0] - x[:, 1]) / math.sqrt(2)
        fourth = (t.square() * u.square()).mean().item()
        cov = torch.cov(x.T)
        return {
            'mean_norm': x.mean(0).norm().item(), 'var_x': cov[0, 0].item(),
            'var_y': cov[1, 1].item(), 'cov_xy': cov[0, 1].item(),
            'cross_fourth': fourth, 'radial_fourth': x.square().sum(1).square().mean().item(),
            'origin_mass': (x.square().sum(1) < 0.25).float().mean().item(),
            'tail_mass': (x.norm(dim=1) > 4).float().mean().item(),
            'v_min': v.min().item(), 'v_p01': torch.quantile(v, .01).item(),
            'v_median': v.median().item(), 'v_p99': torch.quantile(v, .99).item(),
            'v_mean': v.mean().item(), 'v_cv': (v.std() / v.mean()).item(),
            'floor_fraction': (v <= 1.0001e-4).float().mean().item(),
            'inverse_v': (1 / v).mean().item(),
            'mu_variance': torch.var(mu, dim=0).mean().item(),
            'conditional_score_second': neg.square().sum(1).mean().item(),
            'score_p_second': target.score(x).square().sum(1).mean().item(),
        }, x.detach(), eps.detach()


def sample_metrics(x, reference):
    n = min(len(x), len(reference), 10000)
    x, reference = x[:n], reference[:n]
    angles = torch.linspace(0, math.pi, 129, device=x.device)[:-1]
    directions = torch.stack([angles.cos(), angles.sin()])
    sw2 = (torch.sort(x @ directions, dim=0).values -
           torch.sort(reference @ directions, dim=0).values).square().mean().sqrt()
    return {'sw2': sw2.item()}


def audit(root, output):
    rows = []
    target = X_shaped(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    with preserve_rng(81273):
        ref = target.sample(10000)
    for path in sorted(Path(root).glob('ksivi_x_shaped*/**/full_config.yaml')):
        cfg = OmegaConf.load(path)
        if cfg.vi_model_type not in ('ConditionalGaussian', 'ConditionalGaussianGlobal'):
            continue
        model = model_from_config(cfg)
        checkpoints = sorted(path.parent.glob('checkpoints/epoch_*/vi_model.pt'),
                             key=lambda p: int(p.parent.name.split('_')[-1]))
        for cp in checkpoints:
            step = int(cp.parent.name.split('_')[-1])
            model.load_state_dict(torch.load(cp, map_location=target.device, weights_only=True))
            m, x, _ = moments(model, target)
            m.update(sample_metrics(x, ref))
            m.update({'path': str(path.parent), 'checkpoint': str(cp), 'step': step,
                      'seed': int(cfg.seed), 'family': cfg.vi_model_type,
                      'batch': int(cfg.train.batch_size), 'annealing': bool(cfg.train.annealing.enabled),
                      'lr': float(cfg.train.vi.lr), 'gamma': float(cfg.train.vi.scheduler.gamma),
                      'constant_init': cfg.vi_model.get('variance_init'),
                      'epsilon_dim': int(cfg.vi_model.epsilon_dim),
                      'width': int(cfg.vi_model.hidden_dim),
                      'detach_h': cfg.train.ksivi.get('detach_bandwidth', False)})
            rows.append(m)
        print(f'Audited {path.parent}: {len(checkpoints)} checkpoints', flush=True)
    write_json(output, {'source_commit': git_commit(), 'rows': rows})


def git_commit():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


def specs(stage, names=None):
    core = [dict(name='canonical_cond', family='cond'),
            dict(name='canonical_global', family='global'),
            dict(name='no_anneal_cond', family='cond', anneal=False),
            dict(name='no_anneal_global', family='global', anneal=False),
            dict(name='constant_lr_cond', family='cond', gamma=1.),
            dict(name='constant_lr_global', family='global', gamma=1.),
            dict(name='fast_warmup_cond', family='cond', warmup=2000),
            dict(name='fast_warmup_global', family='global', warmup=2000),
            dict(name='detach_h_cond', family='cond', detach_h=True),
            dict(name='detach_h_global', family='global', detach_h=True),
            dict(name='stein_cond', family='cond', estimator='stein', detach_h=True),
            dict(name='stein_global', family='global', estimator='stein', detach_h=True)]
    broad = [dict(name='batch512_cond', family='cond', batch=512),
             dict(name='batch512_global', family='global', batch=512),
             dict(name='floor002_cond', family='cond', floor=.02),
             dict(name='floor02_cond', family='cond', floor=.2),
             dict(name='separate_cond', family='separate'),
             dict(name='no_anneal_const_cond', family='cond', anneal=False, gamma=1.),
             dict(name='no_anneal_const_global', family='global', anneal=False, gamma=1.),
             dict(name='gaussian_cond', family='cond', rho=0.),
             dict(name='gaussian_global', family='global', rho=0.),
             dict(name='rho05_cond', family='cond', rho=.5),
             dict(name='rho05_global', family='global', rho=.5)]
    broad += [dict(name='fixed_h_cond', family='cond', fixed_h=.75),
              dict(name='fixed_h_global', family='global', fixed_h=.75),
              dict(name='fixed_h_stein_cond', family='cond', fixed_h=.75,
                   estimator='stein',detach_h=True)]
    if stage in ('screen','contrast'):
        entries = core + broad
    else:
        # Prespecified confirmation contrasts: reproduction, score integration,
        # bounded variance, and a stationary target with sustained step size.
        entries = [s for s in core if s['name'] in
                   ('canonical_cond','canonical_global','stein_cond','stein_global')]
        entries += [s for s in broad if s['name'] in
                    ('floor02_cond','no_anneal_const_cond','no_anneal_const_global')]
    if names:
        entries = [s for s in core+broad if s['name'] in names]
        assert set(names) == {s['name'] for s in entries}
    seeds = [42] if stage == 'screen' else [42, 43, 44]
    return [dict(s, seed=seed, steps=10000 if stage == 'screen' else 50000,
                 stage=stage) for seed in seeds for s in entries]


class CorrelatedTarget(X_shaped):
    def __init__(self, device, rho):
        super().__init__(device)
        scale = 4 * (1 - rho * rho)
        self.normalization_correction = .5*math.log(.76/scale)
        self._sigmasqinv_0 = torch.tensor([[2., -2*rho], [-2*rho, 2.]], device=device) / scale
        self._sigmasqinv_1 = torch.tensor([[2., 2*rho], [2*rho, 2.]], device=device) / scale
        self._cov_0 = torch.tensor([[2., 2*rho], [2*rho, 2.]], device=device)
        self._cov_1 = torch.tensor([[2., -2*rho], [-2*rho, 2.]], device=device)

    def logp(self, X):
        return super().logp(X) + self.normalization_correction


class SeparateConditional(ConditionalGaussian):
    def __init__(self, config, global_model):
        super().__init__(config)
        self.mean_net = copy.deepcopy(global_model.net)
        self.variance_net = copy.deepcopy(global_model.net)
        with torch.no_grad():
            self.variance_net[-1].weight.zero_()
            self.variance_net[-1].bias.fill_(1.)
        del self.net

    def getmu(self, epsilon):
        return self.mean_net(epsilon)

    def getstd(self, epsilon):
        return self._variance_from_raw(self.variance_net(epsilon))[0].sqrt()

    def forward(self, epsilon):
        return self.reparameterize(self.getmu(epsilon), self.variance_net(epsilon))


def loss_at(runner, epoch, estimator='conditional', h_fixed=None, batch=None):
    model, target = runner.vi_model, runner.target_model
    n = batch or runner.training_batch_size
    e1, e2 = model.sample_epsilon(n), model.sample_epsilon(n)
    x, b = model(e1)
    y, c = model(e2)
    a = annealing(epoch, runner.anneal_steps, anneal=runner.use_annealing,
                  scheme=runner.anneal_scheme)
    sx, sy = a * target.score(x), a * target.score(y)
    if h_fixed is not None:
        runner.kernel.h = h_fixed
    k = runner.kernel.pair_eval(x, y, fit_h=h_fixed is None,
                               detach_h=runner.detach_bandwidth)
    if estimator == 'conditional':
        return ((sx + b) @ (sy + c).T * k).mean()
    # Stein integration-by-parts estimator: same fixed-bandwidth population KSD.
    h2 = runner.kernel.h ** 2
    delta = x[:, None] - y[None, :]
    inner = sx @ sy.T + ((sx[:, None] - sy[None, :]) * delta).sum(-1)/h2
    inner = inner + x.shape[1]/h2 - delta.square().sum(-1)/(h2*h2)
    return (inner * k).mean()


def create_runner(spec, out, tb):
    seed = spec['seed']
    torch.manual_seed(seed)
    cfg = OmegaConf.load(REPO / 'configs/ksivi_x_shaped.yaml')
    cfg.config_path = str(REPO / 'configs/ksivi_x_shaped.yaml')
    cfg.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    cfg.seed = seed
    cfg.vi_model_type = 'ConditionalGaussianGlobal'
    cfg.train.epochs = spec['steps']
    cfg.train.batch_size = spec.get('batch', 128)
    cfg.train.annealing.enabled = spec.get('anneal', True)
    cfg.train.annealing.steps = spec.get('warmup', 25000)
    cfg.train.vi.scheduler.gamma = spec.get('gamma', .9)
    cfg.train.ksivi.detach_bandwidth = spec.get('detach_h', False)
    for metric in cfg.metric.values():
        metric.enabled = False
    cfg.output = {'results_dir': str(out), 'tb_dir': str(tb)}
    runner = KSIVIRunner(cfg)
    global_model = runner.vi_model
    if spec['family'] != 'global':
        mc = OmegaConf.create(OmegaConf.to_container(runner.config.vi_model, resolve=True))
        mc.variance_init = 1.31326162815094
        mc.var_min = spec.get('floor', 1e-4)
        if spec['family'] == 'separate':
            model = SeparateConditional(mc, global_model).to(cfg.device)
        else:
            model = ConditionalGaussian(mc).to(cfg.device)
            with torch.no_grad():
                for dst, src in zip(model.net[:-1].parameters(), global_model.net[:-1].parameters()):
                    dst.copy_(src)
                model.net[-1].weight[:2].copy_(global_model.net[-1].weight)
                model.net[-1].bias[:2].copy_(global_model.net[-1].bias)
        runner.vi_model = model
        runner.optimizer_vi = torch.optim.Adam(model.parameters(), lr=.001, betas=(.9, .999))
        runner.scheduler_vi = torch.optim.lr_scheduler.StepLR(runner.optimizer_vi, 1000,
                                                             gamma=spec.get('gamma', .9))
    if 'rho' in spec:
        runner.target_model = CorrelatedTarget(torch.device(cfg.device), spec['rho'])
    # Start paired training with identical noise draws despite construction costs.
    torch.manual_seed(seed + 100000)
    return runner


def train_one(spec, root, tbroot):
    torch.set_num_threads(1)
    out = Path(root)/spec['stage']/f"{spec['name']}_s{spec['seed']}"
    if (out/'summary.json').exists():
        return
    source_commit = git_commit()
    runner = create_runner(spec, out, Path(tbroot)/spec['stage']/out.name)
    write_json(out/'spec.json', dict(spec, source_commit=source_commit))
    with preserve_rng(81273):
        reference = runner.target_model.sample(10000)
    rows = []
    snapshots = {0, 100, 500, 1000, 2000, 5000, 10000, 25000, 50000}
    snapshots.add(spec['steps'])
    start = time.time()
    for step in range(spec['steps'] + 1):
        if step > 0:
            # Equivalent production objective/update without per-step logging or
            # CUDA scalar synchronization; validate() checks its gradient.
            loss = loss_at(runner, step, spec.get('estimator', 'conditional'),
                           h_fixed=spec.get('fixed_h'))
            runner.optimizer_vi.zero_grad()
            loss.backward()
            runner.optimizer_vi.step()
            runner.scheduler_vi.step()
        if step in snapshots:
            if step and not torch.isfinite(loss):
                raise FloatingPointError(f'{out.name} step {step}')
            m, x, eps = moments(runner.vi_model, runner.target_model)
            m.update(sample_metrics(x, reference))
            m.update(step=step, alpha=annealing(step, runner.anneal_steps, anneal=runner.use_annealing),
                     lr=runner.optimizer_vi.param_groups[0]['lr'], elapsed=time.time()-start)
            rows.append(m)
            torch.save({'model': runner.vi_model.state_dict(), 'x': x.cpu(), 'eps': eps.cpu()},
                       out/f'snapshot_{step}.pt')
            write_json(out/'trajectory.json', rows)
            print(json.dumps(dict(run=out.name, **m)), flush=True)
    runner.writer.close()
    write_json(out/'summary.json', {'spec': spec, 'source_commit': source_commit, 'trajectory': rows})


def campaign(stage, root, tbroot, workers, names=None):
    pending = specs(stage,names)
    active = []
    state = {'source_commit': git_commit(), 'stage': stage, 'started': time.time(),
             'completed': [], 'status': 'running', 'specs': pending.copy()}
    state_path = Path(root)/stage/'state.json'
    try:
        while pending or active:
            while pending and len(active) < workers:
                s = pending.pop(0)
                out = Path(root)/stage/f"{s['name']}_s{s['seed']}"
                out.mkdir(parents=True, exist_ok=True)
                f = (out/'console.log').open('w')
                cmd = [sys.executable, '-u', str(Path(__file__).resolve()), 'train',
                       '--root', str(root), '--tbroot', str(tbroot), '--spec', json.dumps(s)]
                p = subprocess.Popen(cmd, cwd=REPO, stdout=f, stderr=subprocess.STDOUT)
                active.append((s, p, f))
                print('Started', s, flush=True)
            for s, p, f in active.copy():
                if p.poll() is not None:
                    f.close()
                    active.remove((s, p, f))
                    if p.returncode != 0:
                        raise RuntimeError(f"{s['name']}: exit {p.returncode}")
                    state['completed'].append(s)
                    print('Completed', s['name'], s['seed'], flush=True)
            write_json(state_path, state)
            if active:
                time.sleep(2)
        state.update(status='completed', finished=time.time())
    except Exception as exc:
        state.update(status='failed', error=repr(exc))
        raise
    finally:
        write_json(state_path, state)


def validate():
    torch.set_num_threads(1)
    # Score and normalization checks, population identity, matched functions,
    # and analytic fourth moments detect scientific implementation mistakes.
    t = X_shaped(torch.device('cpu'))
    x = torch.randn(200, 2, dtype=torch.float32, requires_grad=True)
    automatic = torch.autograd.grad(t.logp(x).sum(), x)[0]
    assert torch.allclose(automatic, t.score(x), atol=2e-6)
    cfg = OmegaConf.create(dict(device='cpu', z_dim=2, epsilon_dim=2, hidden_dim=128, num_layers=2))
    torch.manual_seed(42)
    g = ConditionalGaussianGlobal(cfg)
    c = ConditionalGaussian(OmegaConf.merge(cfg, {'variance_init': 1.31326162815094}))
    with torch.no_grad():
        for dst, src in zip(c.net[:-1].parameters(), g.net[:-1].parameters()): dst.copy_(src)
        c.net[-1].weight[:2].copy_(g.net[-1].weight)
        c.net[-1].bias[:2].copy_(g.net[-1].bias)
    e = torch.randn(100, 2)
    assert torch.equal(c.getmu(e), g.getmu(e))
    assert torch.allclose(c.getstd(e), g.getstd(e))
    with preserve_rng(1234):
        y = t.sample(200000)
    cross = (((y[:,0]+y[:,1])/math.sqrt(2)).square()*
             ((y[:,0]-y[:,1])/math.sqrt(2)).square()).mean().item()
    assert abs(cross-.76) < .015, cross
    # Independent-batch conditional and integrated Stein objectives have equal
    # expectations for a bandwidth chosen independently of the draws.
    from types import SimpleNamespace
    from utils.kernels import GaussianKernel
    mock = SimpleNamespace(vi_model=c, target_model=t, training_batch_size=128,
                           anneal_steps=25000, use_annealing=True, anneal_scheme='linear',
                           kernel=GaussianKernel(), detach_bandwidth=False)
    import tempfile
    validation_root=Path('/root/ruivi/results') if Path('/root/ruivi/results').exists() else REPO/'results'
    validation_root.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=validation_root) as d:
        production = create_runner(dict(seed=42, family='cond', steps=1), Path(d)/'r', Path(d)/'tb')
        torch.manual_seed(452)
        raw = loss_at(production, 1)
        gradients = torch.autograd.grad(raw, production.vi_model.parameters())
        torch.manual_seed(452)
        production._compute_loss_and_step(1)
        for p, grad in zip(production.vi_model.parameters(), gradients):
            assert torch.equal(p.grad, grad)
        production.writer.close()
        from utils.logging import get_logger
        import logging
        logger = get_logger()
        for handler in logger.handlers.copy():
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
                handler.close()
    print(json.dumps(dict(score_autograd_agrees=True, paired_mean_identical=True,
                          paired_variance_equal_within_float32=True, empirical_cross_fourth=cross,
                          theoretical_cross_fourth=.76, production_gradient_identical=True)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['audit','train','campaign','validate'])
    p.add_argument('--root', default='/root/ruivi/results/ksivi_variance_investigation_20261006')
    p.add_argument('--tbroot', default='/root/ruivi/tb_logs/ksivi_variance_investigation_20261006')
    p.add_argument('--output', default='campaigns/ksivi_x_shaped_evolution_20261006/investigation/existing_audit.json')
    p.add_argument('--spec')
    p.add_argument('--stage', choices=['screen','confirm','contrast','fixed'], default='screen')
    p.add_argument('--names',nargs='+')
    p.add_argument('--workers', type=int, default=3)
    a = p.parse_args()
    if a.action == 'validate': validate()
    elif a.action == 'audit': audit(a.root, a.output)
    elif a.action == 'train': train_one(json.loads(a.spec), a.root, a.tbroot)
    else: campaign(a.stage, a.root, a.tbroot, a.workers,a.names)


if __name__ == '__main__':
    main()
