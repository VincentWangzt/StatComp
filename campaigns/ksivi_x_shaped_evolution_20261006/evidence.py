"""Independent diagnostics and exact-marginal mechanism experiments."""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from omegaconf import OmegaConf

from investigate import (REPO, X_shaped, ConditionalGaussian, ConditionalGaussianGlobal,
                         SeparateConditional, CorrelatedTarget, model_from_config,
                         preserve_rng, write_json, git_commit, moments, sample_metrics)


def stein_matrix(x, y, target, h):
    dx = x[:, None] - y[None]
    d2 = dx.square().sum(-1)
    k = (-d2/(2*h*h)).exp()
    sx, sy = target.score(x), target.score(y)
    inner = sx @ sy.T + ((sx[:,None]-sy[None])*dx).sum(-1)/(h*h)
    return k * (inner + 2/(h*h) - d2/(h**4))


def density_log(model, points, eps):
    with torch.no_grad():
        mu, std = model.getmu(eps), model.getstd(eps)
        if std.ndim == 1: std = std.expand_as(mu)
        var = std.square()
        result = []
        for batch in points.split(64):
            ll = -.5 * (((batch[:,None]-mu[None]).square()/var[None]).sum(-1)
                         + var.log().sum(-1)[None] + 2*math.log(2*math.pi))
            result.append(torch.logsumexp(ll, dim=1)-math.log(len(eps)))
        return torch.cat(result)


def independent_metrics(model, target, high_precision=False):
    with preserve_rng(46829), torch.no_grad():
        ref = target.sample(4096)
        p_log = target.logp(ref).flatten()
        eps = model.sample_epsilon(262144 if high_precision else 16384)
        kl = {}
        for n in (4096, 8192, 16384):
            ell = p_log - density_log(model, ref, eps[:n])
            kl[f'kl_pq_{n}'] = ell.mean().item()
            kl[f'kl_pq_se_{n}'] = (ell.std()/math.sqrt(len(ell))).item()
        best_n=16384
        if high_precision and abs(kl['kl_pq_8192']-kl['kl_pq_16384'])>.015:
            for n in (65536,262144):
                ell=p_log-density_log(model,ref,eps[:n])
                kl[f'kl_pq_{n}']=ell.mean().item()
                kl[f'kl_pq_se_{n}']=(ell.std()/math.sqrt(len(ell))).item()
            best_n=262144
        kl['kl_pq_best']=kl[f'kl_pq_{best_n}']
        kl['kl_mixture_size']=best_n
        kl['kl_last_change']=abs(kl[f'kl_pq_{best_n}']-kl['kl_pq_65536' if best_n==262144 else 'kl_pq_8192'])
        # Bandwidth is fixed independently of evaluation draws. Two separate
        # sample batches remove diagonal bias; standard errors are over repeats.
        for h in (.5, .75, 1.5):
            vals = []
            for _ in range(16):
                x, _ = model(model.sample_epsilon(256))
                y, _ = model(model.sample_epsilon(256))
                vals.append(stein_matrix(x,y,target,h).mean().item())
            kl[f'ksd2_h{h}'] = float(np.mean(vals))
            kl[f'ksd2_se_h{h}'] = float(np.std(vals,ddof=1)/4)
        m, x, _ = moments(model,target, n=50000)
        reference = target.sample(50000)
        m.update(sample_metrics(x,reference))
        # A second independent sample pair estimates Monte Carlo error in SW2.
        y, _ = model(model.sample_epsilon(10000))
        m['sw2_independent'] = sample_metrics(y,target.sample(10000))['sw2']
        return dict(m, **kl)


def trained_directional_noise(model, target, repeats=128):
    records=[]
    with preserve_rng(72183):
        for _ in range(repeats):
            with torch.no_grad():
                x0,b0=model(model.sample_epsilon(128))
                y0,c0=model(model.sample_epsilon(128))
            theta=torch.tensor(0.,device=x0.device,requires_grad=True)
            x,y=x0*theta.exp(),y0*theta.exp()
            b,c=b0*theta.neg().exp(),c0*theta.neg().exp()
            k=(-(x[:,None]-y[None]).square().sum(-1)/(2*.75**2)).exp()
            cond=((target.score(x)+b)@(target.score(y)+c).T*k).mean()
            stein=stein_matrix(x,y,target,.75).mean()
            gc=torch.autograd.grad(cond,theta,retain_graph=True)[0].item()
            gs=torch.autograd.grad(stein,theta)[0].item()
            records.append((gc,gs))
    a=np.array(records)
    return {'direction_repeats':repeats,'direction_cond_grad_mean':float(a[:,0].mean()),
            'direction_stein_grad_mean':float(a[:,1].mean()),
            'direction_cond_grad_var':float(a[:,0].var(ddof=1)),
            'direction_stein_grad_var':float(a[:,1].var(ddof=1))}


def load_new(path):
    spec = json.loads((path/'spec.json').read_text())
    cfg = OmegaConf.create(dict(device='cuda' if torch.cuda.is_available() else 'cpu',
                               z_dim=2,epsilon_dim=2,hidden_dim=128,num_layers=2,
                               variance_init=1.31326162815094,var_min=spec.get('floor',1e-4)))
    if spec['family'] == 'global': model = ConditionalGaussianGlobal(cfg)
    elif spec['family'] == 'separate': model = SeparateConditional(cfg, ConditionalGaussianGlobal(cfg))
    else: model = ConditionalGaussian(cfg)
    model.to(cfg.device)
    cp = path / f"snapshot_{spec['steps']}.pt"
    state = torch.load(cp,map_location=cfg.device,weights_only=True)
    model.load_state_dict(state['model'])
    target = CorrelatedTarget(torch.device(cfg.device),spec.get('rho',.9))
    return model, target, spec


def evaluate(root, output):
    torch.set_num_threads(1)
    out = Path(output)
    previous = json.loads(out.read_text()) if out.exists() else {'rows':[]}
    previous['rows']=[r for r in previous['rows'] if r['spec']['steps']!=50000 or 'kl_pq_best' in r]
    done = {r['path'] for r in previous['rows']}
    for summary in sorted(Path(root).glob('*/*/summary.json')):
        path = summary.parent
        if str(path) in done: continue
        model,target,spec = load_new(path)
        m = independent_metrics(model,target,high_precision=spec['steps']==50000)
        m.update(trained_directional_noise(model,target))
        m.update(path=str(path), spec=spec)
        previous['rows'].append(m)
        previous['source_commit'] = git_commit()
        write_json(out,previous)
        print(spec['stage'],spec['name'],spec['seed'],round(m['sw2'],3),
              round(m['kl_pq_best'],3),round(m['ksd2_h0.75'],5),flush=True)


def evaluate_old(audit_path, output):
    torch.set_num_threads(1)
    rows=json.loads(Path(audit_path).read_text())['rows']
    selected=[]
    seen=set()
    for row in rows:
        if (row['step'] != 50000 or row['batch'] != 128 or not row['annealing']
            or row['lr'] != .001 or row['epsilon_dim'] != 2 or row['width'] != 128): continue
        key=(row['family'],row['seed'],row['constant_init'])
        if key in seen: continue
        seen.add(key)
        selected.append(row)
    data={'source_commit':git_commit(),'rows':[]}
    for row in selected:
        cfg=OmegaConf.load(Path(row['path'])/'full_config.yaml')
        model=model_from_config(cfg)
        device=next(model.parameters()).device
        model.load_state_dict(torch.load(row['checkpoint'],map_location=device,weights_only=True))
        target=X_shaped(device)
        m=independent_metrics(model,target,high_precision=True)
        m.update(trained_directional_noise(model,target))
        data['rows'].append(dict(row,**m))
        write_json(output,data)
        print('historical',row['family'],row['seed'],row['constant_init'],round(m['sw2'],3),
              round(m['kl_pq_16384'],3),flush=True)


def exact_batch(n, v, theta, target, heterogeneous=False):
    device = theta.device
    signs = torch.where(torch.rand(n,device=device)>.5,1.,-1.)
    if heterogeneous:
        # Every value of v gives exactly the SAME target p. The heterogeneous
        # representation therefore also has marginal p by the tower property.
        var = torch.where(torch.rand(n,device=device)>.5,.02,.2)
    else: var = torch.full((n,),v,device=device)
    z = torch.randn(n,2,device=device)
    a = (3.8-var).sqrt()*z[:,0]
    b = (.2-var).clamp_min(0).sqrt()*z[:,1]
    mu = torch.stack([(a+b)/math.sqrt(2),signs*(a-b)/math.sqrt(2)],dim=1)
    u = torch.randn(n,2,device=device)
    x = theta.exp()*(mu+var.sqrt()[:,None]*u)
    neg = theta.neg().exp()*u/var.sqrt()[:,None]
    return x,neg,var


def probes(output):
    torch.set_num_threads(1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target = X_shaped(device)
    result = {'source_commit':git_commit(), 'exact_marginal':[], 'target_curvature':[]}
    torch.manual_seed(717)
    with torch.no_grad():
        ref = target.sample(200000)
        fisher = target.score(ref).square().sum(1).mean().item()
    result['target_fisher'] = fisher
    N,h=128,.75
    laplace=[]
    for eig in [(7.6,.4),(4.,4.)]:
        t=1/(h*h)
        M=math.prod(1+2*t*e for e in eig)**(-.5)
        u=[e/(1+2*t*e) for e in eig]
        A,B=sum(u),sum(z*z for z in u)
        laplace.append(M*(4+4*t*A+t*t*(A*A+2*B)))
    result['gradient_asymptotic_coefficient']=2*np.mean(laplace)/N**2
    for v,hetero in [(.2,False),(.11,False),(.05,False),(.02,False),(.11,True),(.002,False)]:
        records=[]
        theta=torch.tensor(0.,device=device,requires_grad=True)
        for repeat in range(256):
            x,b,var=exact_batch(128,v,theta,target,hetero)
            y,c,_=exact_batch(128,v,theta,target,hetero)
            dx=x[:,None]-y[None]
            k=(-dx.square().sum(-1)/(2*.75**2)).exp()
            cond=((target.score(x)+b)@(target.score(y)+c).T*k).mean()
            stein=stein_matrix(x,y,target,.75).mean()
            gc=torch.autograd.grad(cond,theta,retain_graph=True)[0].item()
            gs=torch.autograd.grad(stein,theta)[0].item()
            records.append((cond.item(),stein.item(),gc,gs))
        a=np.array(records)
        row={'v':v,'heterogeneous':hetero,'batch':128,'repeats':256,
             'conditional_loss_mean':float(a[:,0].mean()),'stein_loss_mean':float(a[:,1].mean()),
             'conditional_grad_mean':float(a[:,2].mean()),'stein_grad_mean':float(a[:,3].mean()),
             'conditional_grad_var':float(a[:,2].var(ddof=1)),
             'stein_grad_var':float(a[:,3].var(ddof=1)),
             'conditional_grad_se':float(a[:,2].std(ddof=1)/16),
             'stein_grad_se':float(a[:,3].std(ddof=1)/16),
             'theoretical_residual_second':(1/.02+1/.2 if hetero else 2/v)-fisher,
             'gradient_asymptotic_prediction':None if hetero else result['gradient_asymptotic_coefficient']/v**2,
             'raw_repetitions':a.tolist()}
        result['exact_marginal'].append(row)
        print({k:v for k,v in row.items() if k!='raw_repetitions'},flush=True)
    # Differentiate the empirical median vs an independently fixed bandwidth.
    # Same distribution, same draws, same parameter direction.
    gradients=[]
    for repeat in range(256):
        theta=torch.tensor(0.,device=device,requires_grad=True)
        x,b,_=exact_batch(128,.2,theta,target)
        y,c,_=exact_batch(128,.2,theta,target)
        d2=(x[:,None]-y[None]).square().sum(-1)
        h2=.5*d2.median()/math.log(129)
        f=(target.score(x)+b)@(target.score(y)+c).T
        full=(f*(-d2/(2*h2)).exp()).mean()
        detached=(f*(-d2/(2*h2.detach())).exp()).mean()
        fixed=(f*(-d2/(2*.75**2)).exp()).mean()
        gf=torch.autograd.grad(full,theta,retain_graph=True)[0].item()
        gd=torch.autograd.grad(detached,theta,retain_graph=True)[0].item()
        gi=torch.autograd.grad(fixed,theta)[0].item()
        gradients.append((gf,gd,gi))
    a=np.array(gradients)
    result['median_gradient']={'repeats':256,'mean':a.mean(0).tolist(),
                               'variance':a.var(0,ddof=1).tolist(),
                               'order':['full_median','detached_median','fixed_h075']}
    A,B=2/.76,1.8/.76
    for radius in [0.,1.,2.,4.,8.,16.]:
        result['target_curvature'].append({'axis_radius':radius,'Hxx':-A+B*B*radius*radius})
    result['schedule']={'initial_lr':.001,'lr_at_target':.001*.9**25,
                        'mass_before_target':1000*.001*(1-.9**25)/.1,
                        'mass_after_target':1000*.001*.9**25*(1-.9**25)/.1}
    write_json(output,result)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['evaluate','probes','evaluate-old'])
    p.add_argument('--root',default='/root/ruivi/results/ksivi_variance_investigation_20261006')
    p.add_argument('--output',required=True)
    a=p.parse_args()
    if a.action=='evaluate': evaluate(a.root,a.output)
    elif a.action=='evaluate-old': evaluate_old(a.root,a.output)
    else: probes(a.output)


if __name__=='__main__': main()
