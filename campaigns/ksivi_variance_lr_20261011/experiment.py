"""Atomic variance-learning-rate experiments using StatComp's production models.

All primary contrasts use 50,000 updates, paired initialization/noise streams,
and the original StatComp kernel, score annealing, Adam and StepLR policy.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from contextlib import contextmanager

import numpy as np
import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from models.vi_model import ConditionalGaussian, ConditionalGaussianGlobal, ConditionalGaussianHeads
from models.target_models import X_shaped
from utils.annealing import annealing
from utils.kernels import GaussianKernel

ROOT = Path('/root/ruivi/results/ksivi_variance_lr_20261011') if os.name != 'nt' else REPO/'results/ksivi_variance_lr_20261011'
RATIOS = [0., .001, .003, .01, .03, .1, .3, 1., 3.]


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def revision():
    return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip()


@contextmanager
def rng(seed=None):
    with torch.random.fork_rng(devices=[0] if torch.cuda.is_available() else []):
        if seed is not None:
            torch.manual_seed(seed)
        yield


def label(architecture, ratio):
    return f'{architecture}_r{ratio:g}'.replace('.', 'p')


def specifications(phase, root=ROOT):
    seeds = [42,43,44]
    entries = []
    if phase == 'sweep':
        for architecture in ['shared','separate']:
            entries += [dict(name=label(architecture,r), architecture=architecture, ratio=r) for r in RATIOS]
        entries += [dict(name='legacy', architecture='legacy', ratio=1.),
                    dict(name='global', architecture='global', ratio=1.),
                    dict(name='stein', architecture='shared', ratio=1., estimator='stein'),
                    dict(name='floor02', architecture='shared', ratio=1., floor=.2)]
    elif phase == 'architecture':
        for architecture, extra in [('narrow',dict(variance_width=32,variance_depth=2)),
                                     ('shallow',dict(variance_width=128,variance_depth=1)),
                                     ('linear',dict(variance_width=128,variance_depth=0)),
                                     ('stopgrad',{})]:
            entries += [dict(name=label(architecture,r), architecture=architecture, ratio=r, **extra) for r in [.01,.1,1.]]
    elif phase in ['confirmation','ablation','extension']:
        selection = json.loads((Path(root)/'selection.json').read_text())
        selected = selection['selected']
        if phase == 'confirmation':
            seeds = [45,46,47,48,49]
            entries = selected + [dict(name='shared_r1',architecture='shared',ratio=1.),
                                  dict(name='separate_r1',architecture='separate',ratio=1.),
                                  dict(name='global',architecture='global',ratio=1.),
                                  dict(name='stein',architecture='shared',ratio=1.,estimator='stein'),
                                  dict(name='floor02',architecture='shared',ratio=1.,floor=.2)]
        elif phase == 'ablation':
            shared, separate = selected[:2]
            # Exactly one change from the corresponding sweep configuration.
            parents = [dict(name='shared_r1',architecture='shared',ratio=1.), shared, separate,
                       dict(name='global',architecture='global',ratio=1.),
                       dict(name='stein',architecture='shared',ratio=1.,estimator='stein')]
            entries = [dict(s, name=s['name']+'_noanneal', anneal=False, parent=s['name']) for s in parents]
            initial = [dict(name='shared_r1',architecture='shared',ratio=1.),
                       dict(name='shared_r0',architecture='shared',ratio=0.), shared, separate,
                       dict(name='global',architecture='global',ratio=1.)]
            entries += [dict(s, name=s['name']+'_init02', initial_variance=.2, parent=s['name']) for s in initial]
            entries += [dict(shared,name=shared['name']+'_stein',estimator='stein',parent=shared['name']),
                        dict(shared,name=shared['name']+'_floor02',floor=.2,parent=shared['name'])]
            entries += [dict(s,name=s['name']+'_detach',detach_h=True,parent=s['name']) for s in
                        [dict(name='shared_r1',architecture='shared',ratio=1.),shared]]
        else:
            parents = selected[:2] + [dict(name='shared_r1',architecture='shared',ratio=1.),
                                      dict(name='global',architecture='global',ratio=1.),
                                      dict(name='stein',architecture='shared',ratio=1.,estimator='stein')]
            entries=[]
            for s in parents:
                for policy in ['decay','restart1e-4']:
                    entries.append(dict(s,name=s['name']+'_'+policy,steps=200000,extension_policy=policy,
                                        parent=s['name']))
    else:
        raise ValueError(phase)
    # A selected configuration may coincide with a listed control.
    entries = list({s['name']:s for s in entries}.values())
    return [dict(s, seed=seed, phase=phase, steps=s.get('steps',50000)) for s in entries for seed in seeds]


def create_model(spec, device):
    torch.manual_seed(spec['seed'])
    cfg = OmegaConf.create(dict(device=str(device), z_dim=2, epsilon_dim=2, hidden_dim=128,
                                num_layers=2, variance_init=spec.get('initial_variance',1.31326162815094),
                                var_min=spec.get('floor',1e-4), activation='silu',
                                global_variance_init=spec.get('initial_variance',None)))
    global_model = ConditionalGaussianGlobal(cfg)
    architecture = spec['architecture']
    if architecture == 'global':
        model = global_model
    elif architecture == 'legacy':
        model = ConditionalGaussian(cfg)
        with torch.no_grad():
            for dst,src in zip(model.net[:-1].parameters(),global_model.net[:-1].parameters()): dst.copy_(src)
            model.net[-1].weight[:2].copy_(global_model.net[-1].weight)
            model.net[-1].bias[:2].copy_(global_model.net[-1].bias)
    else:
        cfg.variance_backbone = architecture if architecture in ('shared','stopgrad') else 'separate'
        cfg.variance_hidden_dim = spec.get('variance_width',128)
        cfg.variance_num_layers = spec.get('variance_depth',2)
        model = ConditionalGaussianHeads(cfg)
        with torch.no_grad():
            for dst,src in zip(model.mean_backbone.parameters(),global_model.net[:-1].parameters()): dst.copy_(src)
            model.mean_head.load_state_dict(global_model.net[-1].state_dict())
            if cfg.variance_backbone == 'separate' and cfg.variance_hidden_dim == 128 and cfg.variance_num_layers == 2:
                model.variance_net.load_state_dict(global_model.net[:-1].state_dict())
    model = model.to(device)
    if isinstance(model,ConditionalGaussianHeads):
        groups = model.optimizer_parameter_groups(.001,.001*spec['ratio'])
    elif architecture == 'global':
        groups = [dict(params=list(model.net.parameters()),lr=.001,name='mean'),
                  dict(params=[model.var_raw],lr=.001*spec['ratio'],name='variance')]
    else:
        groups = [dict(params=list(model.parameters()),lr=.001,name='legacy')]
    optimizer = torch.optim.Adam(groups,betas=(.9,.999))
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,1000,gamma=.9)
    return model,optimizer,scheduler


def loss_from_samples(x,b,y,c,target,alpha,estimator,detach_h=False,fixed_h=None):
    sx,sy=alpha*target.score(x),alpha*target.score(y)
    kernel=GaussianKernel(-1 if fixed_h is None else fixed_h)
    k=kernel.pair_eval(x,y,fit_h=fixed_h is None,detach_h=detach_h or estimator=='stein')
    if estimator=='original':
        return (((sx+b)@(sy+c).T)*k).mean()
    h2=kernel.h**2
    delta=x[:,None]-y[None,:]
    inner=sx@sy.T+((sx[:,None]-sy[None,:])*delta).sum(-1)/h2
    return (k*(inner+2/h2-delta.square().sum(-1)/h2**2)).mean()


def loss_at(model,target,spec,step,fixed_h=None):
    e1,e2=model.sample_epsilon(128),model.sample_epsilon(128)
    x,b=model(e1);y,c=model(e2)
    alpha=annealing(step,25000,anneal=spec.get('anneal',True))
    return loss_from_samples(x,b,y,c,target,alpha,spec.get('estimator','original'),
                             spec.get('detach_h',False),fixed_h)


def sw2(x,y):
    angles=torch.linspace(0,math.pi,129,device=x.device)[:-1]
    directions=torch.stack([angles.cos(),angles.sin()])
    return (torch.sort(x@directions,dim=0).values-torch.sort(y@directions,dim=0).values).square().mean().sqrt().item()


def diagnostics(model,target,n=10000):
    with rng(20261011),torch.no_grad():
        eps=model.sample_epsilon(n)
        mu=model.getmu(eps)
        v=model.getstd(eps).square()
        if v.ndim==1:v=v.expand_as(mu)
        x,b=model(eps)
        reference=target.sample(n)
        t=(x[:,0]+x[:,1])/math.sqrt(2);u=(x[:,0]-x[:,1])/math.sqrt(2)
        cov=torch.cov(x.T)
        result=dict(sw2=sw2(x,reference),inverse_v=(1/v).mean().item(),v_min=v.min().item(),
                    v_q01=torch.quantile(v,.01).item(),v_median=v.median().item(),v_mean=v.mean().item(),
                    v_q99=torch.quantile(v,.99).item(),floor_fraction=(v<=1.0001e-4).float().mean().item(),
                    mean_norm=x.mean(0).norm().item(),var_x=cov[0,0].item(),var_y=cov[1,1].item(),
                    cross_fourth=(t.square()*u.square()).mean().item(),origin_mass=(x.norm(dim=1)<.5).float().mean().item(),
                    tail_mass=(x.norm(dim=1)>4).float().mean().item(),score_second=b.square().mean().item())
        return result,x.cpu()


def checkpoint(path,model,optimizer,scheduler,step,spec):
    state=dict(model=model.state_dict(),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict(),
               step=step,spec=spec,cpu_rng=torch.get_rng_state(),
               cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])
    tmp=path.with_suffix('.tmp');torch.save(state,tmp);tmp.replace(path)


def run_one(spec,root,limit=None):
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    out=Path(root)/spec['phase']/f"{spec['name']}_s{spec['seed']}"
    out.mkdir(parents=True,exist_ok=True)
    final=out/'result.json'
    if final.exists():return
    model,optimizer,scheduler=create_model(spec,device)
    target=X_shaped(device)
    train_spec=dict(spec)
    if 'extension_policy' in spec:
        train_spec['steps']=50000
        train_spec['name']=spec['parent']
        train_spec['phase']='sweep'
        parent=Path(root)/'sweep'/f"{spec['parent']}_s{spec['seed']}"/'checkpoint_50000.pt'
        state=torch.load(parent,map_location=device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        start=50000
        if spec['extension_policy']=='restart1e-4':
            for group in optimizer.param_groups:
                group['lr']=1e-4*(spec['ratio'] if group.get('name')=='variance' else 1.)
    else:
        torch.manual_seed(spec['seed']+100000);start=0
    # Resume interrupted jobs without rerunning completed optimizer updates.
    saved=sorted(out.glob('checkpoint_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
    rows=[]
    if saved:
        state=torch.load(saved[-1],map_location=device,weights_only=True)
        model.load_state_dict(state['model']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
        torch.set_rng_state(state['cpu_rng'].cpu())
        if state['cuda_rng']:torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda_rng']])
        start=state['step'];rows=json.loads((out/'trajectory.json').read_text())
        rows=[r for r in rows if r['step']<=start]
    source=dict(revision=revision(),torch=torch.__version__,device=str(device),parameters=sum(p.numel() for p in model.parameters()),
                model_config=OmegaConf.to_container(model.config,resolve=True),spec=spec,
                source_sha256={p:hashlib.sha256((REPO/p).read_bytes()).hexdigest() for p in
                               ['models/vi_model.py','utils/kernels.py','utils/annealing.py',str(Path(__file__).relative_to(REPO)).replace('\\','/')]})
    write_json(out/'specification.json',source)
    snapshots={0,1000,5000,10000,25000,50000,75000,100000,150000,200000}
    last=min(spec['steps'],limit) if limit else spec['steps'];snapshots.add(last)
    began=time.monotonic()
    for step in range(start,last+1):
        if step>start:
            loss=loss_at(model,target,spec,step)
            optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
            if spec.get('extension_policy')!='restart1e-4':scheduler.step()
            if step%1000==0 and not torch.isfinite(loss):raise FloatingPointError((spec,step))
        if step in snapshots and (not rows or rows[-1]['step']!=step):
            metrics,samples=diagnostics(model,target)
            metrics.update(step=step,lr=[g['lr'] for g in optimizer.param_groups],
                           alpha=annealing(step,25000,anneal=spec.get('anneal',True)),elapsed=time.monotonic()-began)
            rows.append(metrics);write_json(out/'trajectory.json',rows)
            checkpoint(out/f'checkpoint_{step}.pt',model,optimizer,scheduler,step,spec)
            if step in (50000,100000,200000):np.save(out/f'samples_{step}.npy',samples.numpy())
            print(json.dumps(dict(run=out.name,**metrics)),flush=True)
    if last==spec['steps']:
        write_json(final,dict(spec=spec,source=source,final=rows[-1],trajectory=rows))


def campaign(phase,root,jobs,limit=None):
    pending=specifications(phase,root);active=[]
    state=dict(phase=phase,source_commit=revision(),started=time.time(),total=len(pending),completed=[],failed=[])
    runtime=Path(root)/'runtime'/phase;runtime.mkdir(parents=True,exist_ok=True)
    while pending or active:
        while pending and len(active)<jobs:
            spec=pending.pop(0)
            path=Path(root)/phase/f"{spec['name']}_s{spec['seed']}"/'result.json'
            if path.exists():state['completed'].append(spec);continue
            stream=(runtime/f"{spec['name']}_s{spec['seed']}.log").open('a')
            cmd=[sys.executable,'-u',str(Path(__file__).resolve()),'worker','--root',str(root),'--spec',json.dumps(spec)]
            if limit:cmd+=['--limit',str(limit)]
            process=subprocess.Popen(cmd,cwd=REPO,stdout=stream,stderr=subprocess.STDOUT)
            active.append((spec,process,stream));print('START',spec,flush=True)
        for spec,process,stream in active.copy():
            if process.poll() is not None:
                stream.close();active.remove((spec,process,stream))
                state['failed' if process.returncode else 'completed'].append(dict(spec,exit_code=process.returncode))
                print('END',spec['name'],spec['seed'],process.returncode,flush=True)
        state.update(active=[s for s,_,_ in active],pending=len(pending),updated=time.time())
        write_json(runtime/'state.json',state)
        if active:time.sleep(2)
    state['status']='failed' if state['failed'] else 'complete';write_json(runtime/'state.json',state)
    if state['failed']:raise RuntimeError(f"{len(state['failed'])} failed jobs")


def select(root):
    rows=[]
    for phase in ['sweep','architecture']:
        rows += [json.loads(p.read_text()) for p in (Path(root)/phase).glob('*/result.json')]
    selected=[];rankings={}
    for group,architectures in [('shared',['shared']),('separate',['separate']),('alternative',['narrow','shallow','linear','stopgrad'])]:
        configs={}
        for row in rows:
            s=row['spec']
            if s['architecture'] in architectures and s.get('estimator','original')=='original' and 'floor' not in s and s['ratio']>0:
                configs.setdefault(s['name'],[]).append(row)
        ranking=[]
        for name,rr in configs.items():
            if sorted(r['spec']['seed'] for r in rr)!=[42,43,44]:continue
            ranking.append(dict(name=name,mean_sw2=float(np.mean([r['final']['sw2'] for r in rr])),spec=rr[0]['spec']))
        ranking.sort(key=lambda r:(r['mean_sw2'],r['name']))
        if not ranking:raise RuntimeError('Incomplete discovery phase: '+group)
        rankings[group]=ranking
        chosen={k:v for k,v in ranking[0]['spec'].items() if k not in ('seed','phase','steps')}
        selected.append(chosen)
    selection=dict(rule='minimum arithmetic mean final SW2 on discovery seeds 42,43,44; positive LR; original loss',
                   selected=selected,rankings=rankings,validation_seeds=[45,46,47,48,49],source_revision=revision())
    write_json(Path(root)/'selection.json',selection)
    write_json(REPO/'campaigns/ksivi_variance_lr_20261011/report/selection.json',selection)
    print(json.dumps(selected,indent=2))


def validate(root):
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    a=dict(seed=42,architecture='legacy',ratio=1.)
    legacy,lo,ls=create_model(a,device)
    shared,so,ss=create_model(dict(a,architecture='shared'),device)
    eps=torch.randn(64,2,device=device)
    assert torch.equal(legacy.getmu(eps),shared.getmu(eps))
    assert torch.equal(legacy.getstd(eps),shared.getstd(eps))
    target=X_shaped(device)
    errors=[]
    for step in [1,25,25000,50000]:
        with rng(998+step):ll=loss_at(legacy,target,a,step)
        with rng(998+step):sl=loss_at(shared,target,a,step)
        assert torch.equal(ll,sl)
        lo.zero_grad();so.zero_grad();ll.backward();sl.backward()
        assert all(torch.equal(x.grad,y.grad) for x,y in zip(legacy.net[:-1].parameters(),shared.mean_backbone.parameters()))
        joined=torch.cat([shared.mean_head.weight.grad,shared.variance_head.weight.grad])
        assert torch.equal(legacy.net[-1].weight.grad,joined)
        lo.step();so.step();ls.step();ss.step()
        error=(legacy.getmu(eps)-shared.getmu(eps)).abs().max().item();errors.append(error)
        assert error<2e-6,error
    # LR=0 is a true parameter freeze, not gradient scaling (Adam cancels scaling).
    frozen,fo,fs=create_model(dict(a,architecture='shared',ratio=0.),device)
    initial=frozen.variance_head.weight.detach().clone()
    initial_bias=frozen.variance_head.bias.detach().clone()
    for _ in range(4):
        fo.zero_grad();loss_at(frozen,target,a,25000).backward();fo.step()
    assert torch.equal(initial,frozen.variance_head.weight)
    assert torch.equal(initial_bias,frozen.variance_head.bias)
    coverage={}
    for architecture in ['shared','stopgrad','separate','narrow','shallow','linear']:
        spec=dict(a,architecture=architecture)
        if architecture=='narrow':spec.update(variance_width=32,variance_depth=2)
        if architecture=='shallow':spec.update(variance_width=128,variance_depth=1)
        if architecture=='linear':spec.update(variance_width=128,variance_depth=0)
        m,o,_=create_model(spec,device)
        params=[id(p) for g in o.param_groups for p in g['params']]
        assert len(params)==len(set(params))==len(list(m.parameters()))
        assert set(params)=={id(p) for p in m.parameters()}
        reference,_,_=create_model(dict(a,architecture='global'),device)
        assert torch.allclose(m.getmu(eps),reference.getmu(eps),atol=2e-6,rtol=2e-6)
        assert torch.allclose(m.getstd(eps).square(),torch.full((64,2),1.31326162815094,device=device))
        z,_=m(eps)
        assert torch.isfinite(m.logp(z,eps)).all()
        assert torch.allclose(m.score(z,eps),-(z-m.getmu(eps))/m.getstd(eps).square(),atol=1e-5)
        coverage[architecture]=sum(p.numel() for p in m.parameters())
    # Validate the production runner's actual optimizer update, including graph order.
    from runner.ksivi import KSIVIRunner
    cfg=OmegaConf.load(REPO/'configs/ksivi_x_shaped.yaml')
    cfg.config_path=str(REPO/'configs/ksivi_x_shaped.yaml');cfg.device=str(device)
    cfg.vi_model_type='ConditionalGaussianHeads';cfg.train.vi.var_lr=1e-5
    cfg.output=dict(results_dir=str(Path(root)/'validation_production'),
                    tb_dir=str(REPO/'tb_logs/ksivi_variance_lr_20261011/validation'))
    for metric in cfg.metric.values():metric.enabled=False
    runner=KSIVIRunner(cfg);cloned=copy.deepcopy(runner.vi_model)
    other=torch.optim.Adam(cloned.optimizer_parameter_groups(.001,1e-5),betas=(.9,.999))
    assert [g['lr'] for g in runner.optimizer_vi.param_groups]==[.001,1e-5]
    with rng(54321):production_loss=runner._compute_loss_and_step(5000)['loss']
    with rng(54321):campaign_loss=loss_at(cloned,runner.target_model,{},5000)
    other.zero_grad();campaign_loss.backward();other.step()
    production_error=max((p-q).abs().max().item() for p,q in zip(runner.vi_model.parameters(),cloned.parameters()))
    assert torch.equal(production_loss,campaign_loss)
    assert production_error<2e-7,production_error
    runner.writer.close()
    write_json(Path(root)/'validation.json',dict(device=str(device),legacy_shared_update_max_errors=errors,parameter_counts=coverage,
                                                loss_and_gradient_equal=True,zero_lr_freeze=True,optimizer_coverage=True,
                                                production_update_max_error=production_error,revision=revision()))
    print('VALIDATION PASSED',device,errors,coverage)


def benchmark(root,steps):
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=dict(seed=42,architecture='shared',ratio=.1)
    m,o,s=create_model(spec,device);target=X_shaped(device)
    for _ in range(30):
        o.zero_grad();loss_at(m,target,spec,30000).backward();o.step()
    if device.type=='cuda':torch.cuda.synchronize()
    began=time.perf_counter()
    for _ in range(steps):
        o.zero_grad();loss_at(m,target,spec,30000).backward();o.step();s.step()
    if device.type=='cuda':torch.cuda.synchronize()
    result=dict(device=str(device),steps=steps,seconds=time.perf_counter()-began,torch=torch.__version__)
    write_json(Path(root)/'benchmark.json',result);print(result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['validate','benchmark','campaign','worker','select','plan'])
    p.add_argument('--phase',default='sweep',choices=['sweep','architecture','confirmation','ablation','extension'])
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--jobs',type=int,default=8)
    p.add_argument('--spec');p.add_argument('--limit',type=int);p.add_argument('--benchmark-steps',type=int,default=1000)
    a=p.parse_args()
    if a.action=='validate':validate(a.root)
    elif a.action=='benchmark':benchmark(a.root,a.benchmark_steps)
    elif a.action=='campaign':campaign(a.phase,a.root,a.jobs,a.limit)
    elif a.action=='worker':run_one(json.loads(a.spec),a.root,a.limit)
    elif a.action=='select':select(a.root)
    else:print(json.dumps(specifications(a.phase,a.root),indent=2))


if __name__=='__main__':main()
