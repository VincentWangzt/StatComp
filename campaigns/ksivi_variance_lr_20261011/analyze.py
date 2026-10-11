"""Independent evaluation, gradient-path probes, and machine-readable evidence."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment import ROOT,REPO,create_model,rng,write_json,revision,sw2,loss_from_samples,loss_at
from models.vi_model import ConditionalGaussianHeads
from models.target_models import X_shaped


def load(path,step=None,device=None):
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    spec=json.loads((path/'specification.json').read_text())['spec']
    step=step or spec['steps']
    m,o,s=create_model(spec,device)
    state=torch.load(path/f'checkpoint_{step}.pt',map_location=device,weights_only=True)
    m.load_state_dict(state['model']);o.load_state_dict(state['optimizer']);s.load_state_dict(state['scheduler'])
    return m,o,s,spec


def stein_matrix(x,y,target,h):
    delta=x[:,None]-y[None]
    distance=delta.square().sum(-1)
    k=(-distance/(2*h*h)).exp()
    sx,sy=target.score(x),target.score(y)
    return k*(sx@sy.T+((sx[:,None]-sy[None])*delta).sum(-1)/(h*h)+2/(h*h)-distance/h**4)


def mixture_log_density(points,mu,variance):
    result=[]
    for batch in points.split(32):
        ll=-.5*(((batch[:,None]-mu[None]).square()/variance[None]).sum(-1)
                 +variance.log().sum(-1)[None]+2*math.log(2*math.pi))
        result.append(torch.logsumexp(ll,dim=1)-math.log(len(mu)))
    return torch.cat(result)


def independent_metrics(model,target):
    with rng(46829),torch.no_grad():
        x,_=model(model.sample_epsilon(20000))
        reference=target.sample(20000)
        result=dict(sw2_20k=sw2(x,reference))
        y,_=model(model.sample_epsilon(10000))
        result['sw2_independent_10k']=sw2(y,target.sample(10000))
        # The target density includes its mixture normalization constant.
        points=target.sample(2048)
        p_log=target.logp(points).flatten()
        eps=model.sample_epsilon(65536)
        mu=model.getmu(eps);v=model.getstd(eps).square()
        if v.ndim==1:v=v.expand_as(mu)
        previous=None
        for n in [8192,16384,65536]:
            ell=p_log-mixture_log_density(points,mu[:n],v[:n])
            result[f'kl_pq_{n}']=ell.mean().item()
            result[f'kl_pq_se_{n}']=ell.std().item()/math.sqrt(len(ell))
            if previous is not None:result['kl_last_change']=abs(result[f'kl_pq_{n}']-previous)
            previous=result[f'kl_pq_{n}']
        result['kl_mixture_size']=65536
        if result['kl_last_change']>.02:
            eps=model.sample_epsilon(262144)
            mu=model.getmu(eps);v=model.getstd(eps).square()
            if v.ndim==1:v=v.expand_as(mu)
            ell=p_log-mixture_log_density(points,mu,v)
            result['kl_pq_262144']=ell.mean().item()
            result['kl_pq_se_262144']=ell.std().item()/math.sqrt(len(ell))
            result['kl_last_change']=abs(result['kl_pq_262144']-previous)
            result['kl_mixture_size']=262144
        result['kl_pq']=result['kl_pq_'+str(result['kl_mixture_size'])]
        for h in [.5,.75,1.5]:
            vals=[]
            for _ in range(16):
                a,_=model(model.sample_epsilon(256));b,_=model(model.sample_epsilon(256))
                vals.append(stein_matrix(a,b,target,h).mean().item())
            result[f'ksd2_h{h}']=float(np.mean(vals))
            result[f'ksd2_se_h{h}']=float(np.std(vals,ddof=1)/4)
        cov=torch.cov(x.T)
        t=(x[:,0]+x[:,1])/math.sqrt(2);u=(x[:,0]-x[:,1])/math.sqrt(2)
        result.update(mean_norm=x.mean(0).norm().item(),var_x=cov[0,0].item(),var_y=cov[1,1].item(),
                      cross_fourth=(t.square()*u.square()).mean().item(),
                      tail_mass=(x.norm(dim=1)>4).float().mean().item())
        if isinstance(model,ConditionalGaussianHeads):
            probe=model.sample_epsilon(10000)
            features=model.variance_features(probe)
            raw=model.raw_parameters(probe)[1]
            result.update(variance_feature_rms=features.square().mean().sqrt().item(),
                          variance_feature_norm=features.norm(dim=1).mean().item(),
                          variance_head_weight_norm=model.variance_head.weight.norm().item(),
                          raw_variance_mean=raw.mean().item(),raw_variance_q01=torch.quantile(raw,.01).item())
    return result


def directional_noise(model,target,repeats=128):
    records=[]
    with rng(72183):
        for _ in range(repeats):
            with torch.no_grad():
                x0,b0=model(model.sample_epsilon(128));y0,c0=model(model.sample_epsilon(128))
            theta=torch.zeros((),device=x0.device,requires_grad=True)
            x=x0*theta.exp();y=y0*theta.exp()
            b=b0*(-theta).exp();c=c0*(-theta).exp()
            original=loss_from_samples(x,b,y,c,target,1.,'original',fixed_h=.75)
            stein=stein_matrix(x,y,target,.75).mean()
            gc=torch.autograd.grad(original,theta,retain_graph=True)[0].item()
            gs=torch.autograd.grad(stein,theta)[0].item()
            records.append([gc,gs])
    values=np.asarray(records)
    return dict(repeats=repeats,original_mean=float(values[:,0].mean()),stein_mean=float(values[:,1].mean()),
                original_variance=float(values[:,0].var(ddof=1)),stein_variance=float(values[:,1].var(ddof=1)),
                mean_difference=float((values[:,0]-values[:,1]).mean()),
                mean_difference_se=float((values[:,0]-values[:,1]).std(ddof=1)/math.sqrt(repeats)))


def flatten(grads,params):
    return torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for g,p in zip(grads,params)])


def path_gradients(model,target,repeats=32):
    if not isinstance(model,ConditionalGaussianHeads):return None
    backbone=list(model.mean_backbone.parameters())
    entries=[];mean_samples=[];variance_samples=[]
    with rng(77171):
        for _ in range(repeats):
            e1,e2=model.sample_epsilon(128),model.sample_epsilon(128)
            mu1,raw1=model.raw_parameters(e1);mu2,raw2=model.raw_parameters(e2)
            x,b=model.reparameterize(mu1,raw1);y,c=model.reparameterize(mu2,raw2)
            loss=loss_from_samples(x,b,y,c,target,1.,'original')
            dmu1,draw1,dmu2,draw2=torch.autograd.grad(loss,[mu1,raw1,mu2,raw2],retain_graph=True)
            mean=flatten(torch.autograd.grad([mu1,mu2],backbone,[dmu1,dmu2],retain_graph=True,allow_unused=True),backbone)
            variance=flatten(torch.autograd.grad([raw1,raw2],backbone,[draw1,draw2],retain_graph=True,allow_unused=True),backbone)
            total=flatten(torch.autograd.grad(loss,backbone),backbone)
            error=(total-mean-variance).norm().item()/max(total.norm().item(),1e-12)
            cosine=(mean@variance/(mean.norm()*variance.norm()).clamp_min(1e-20)).item()
            entries.append(dict(mean_norm=mean.norm().item(),variance_norm=variance.norm().item(),
                                total_norm=total.norm().item(),cosine=cosine,decomposition_relative_error=error))
            mean_samples.append(mean.detach().cpu().numpy());variance_samples.append(variance.detach().cpu().numpy())
    mean_array=np.asarray(mean_samples,dtype=np.float64);variance_array=np.asarray(variance_samples,dtype=np.float64)
    return dict(repeats=repeats,**{k:float(np.mean([r[k] for r in entries])) for k in entries[0]},
                antagonistic_fraction=float(np.mean([r['cosine']<0 for r in entries])),
                mean_path_signal_norm=float(np.linalg.norm(mean_array.mean(0))),
                variance_path_signal_norm=float(np.linalg.norm(variance_array.mean(0))),
                mean_path_noise_trace=float(mean_array.var(0,ddof=1).sum()),
                variance_path_noise_trace=float(variance_array.var(0,ddof=1).sum()))


def functional_drift(model,optimizer,target,spec,freeze=False):
    if not isinstance(model,ConditionalGaussianHeads):return None
    state=copy.deepcopy(model.state_dict());opt_state=copy.deepcopy(optimizer.state_dict())
    try:
        with rng(99117):
            eps=model.sample_epsilon(1024)
            with torch.no_grad():
                h0=model.variance_features(eps).double()
                w0=model.variance_head.weight.detach().double().clone()
                bias0=model.variance_head.bias.detach().double().clone()
                a0=h0@w0.T+bias0
            # A standardized local step avoids float32 cancellation at late StepLR.
            for g in optimizer.param_groups:
                g['lr']=.001*(0. if freeze else spec['ratio']) if g.get('name')=='variance' else .001
            optimizer.zero_grad();loss_at(model,target,spec,50000).backward();optimizer.step()
            with torch.no_grad():
                h1=model.variance_features(eps).double()
                w1=model.variance_head.weight.detach().double().clone()
                bias1=model.variance_head.bias.detach().double().clone()
                head=h0@(w1-w0).T+(bias1-bias0)
                backbone=(h1-h0)@w0.T
                interaction=(h1-h0)@(w1-w0).T
                actual=h1@w1.T+bias1-a0
                rms=lambda a:a.square().mean().sqrt().item()
                return dict(mean_lr=.001,variance_lr=optimizer.param_groups[-1]['lr'],head_rms=rms(head),
                            backbone_rms=rms(backbone),interaction_rms=rms(interaction),actual_rms=rms(actual),
                            identity_max_error=(actual-head-backbone-interaction).abs().max().item(),freeze=freeze)
    finally:
        model.load_state_dict(state);optimizer.load_state_dict(opt_state)


def evaluate(root,phases,output,probe=False):
    torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    result=json.loads(output.read_text()) if output.exists() else dict(rows=[])
    done={(r['path'],r['step']) for r in result['rows']}
    selected=None
    if (root/'selection.json').exists():
        selected=json.loads((root/'selection.json').read_text())['selected']
    probe_names={'shared_r1','separate_r1','global','stein','floor02','stopgrad_r0p1'}
    if selected:probe_names|={s['name'] for s in selected}
    for phase in phases:
        for path in sorted((root/phase).glob('*/result.json')):
            completed=json.loads(path.read_text());spec=completed['spec']
            steps=[50000] if phase!='extension' else [100000,200000]
            for step in steps:
                key=(str(path.parent),step)
                if key in done:continue
                model,opt,_,spec=load(path.parent,step,device)
                target=X_shaped(device)
                metrics=independent_metrics(model,target)
                quick=next(r for r in completed['trajectory'] if r['step']==step)
                row=dict(path=str(path.parent),step=step,spec=spec,quick=quick,metrics=metrics)
                if probe and phase in ('sweep','architecture') and spec['name'] in probe_names:
                    row['directional_noise']=directional_noise(model,target)
                    row['gradient_paths']=path_gradients(model,target)
                    row['drift']=functional_drift(model,opt,target,spec)
                    row['freeze_drift']=functional_drift(model,opt,target,spec,freeze=True)
                result['rows'].append(row);done.add(key)
                result.update(source_revision=revision(),metric_definition=dict(sw2='20000 independent samples,128 directions',
                              kl='2048 exact-target points; adaptive Gaussian mixture approximation to q',
                              ksd='16 independent cross-batch estimates,256 samples per batch,fixed bandwidth'))
                write_json(output,result)
                print(phase,spec['name'],spec['seed'],step,round(metrics['sw2_20k'],4),round(metrics['kl_pq'],4),flush=True)


def export(root,output):
    output.mkdir(parents=True,exist_ok=True)
    rows=[];trajectories=[]
    for phase in ['sweep','architecture','confirmation','ablation','extension']:
        for path in sorted((root/phase).glob('*/result.json')):
            result=json.loads(path.read_text());s=result['spec']
            for t in result['trajectory']:
                trajectories.append(dict(phase=phase,name=s['name'],seed=s['seed'],**t))
            rows.append(dict(spec=s,final=result['final'],trajectory=result['trajectory'],source=result['source']))
    write_json(output/'training_results.json',dict(rows=rows,source_revision=revision()))
    if (root/'selection.json').exists():write_json(output/'selection.json',json.loads((root/'selection.json').read_text()))
    if (root/'validation.json').exists():write_json(output/'validation.json',json.loads((root/'validation.json').read_text()))
    for name,data in [('trajectories.csv',trajectories),('final_training_metrics.csv',
                      [dict(phase=r['spec']['phase'],name=r['spec']['name'],seed=r['spec']['seed'],**r['final']) for r in rows])]:
        if not data:continue
        keys=sorted(set().union(*(d.keys() for d in data)))
        with (output/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(data)
    print('EXPORTED',len(rows),'runs',output)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['evaluate','export'])
    p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--phase',nargs='+',default=['sweep','architecture','confirmation','ablation','extension'])
    p.add_argument('--output',type=Path,default=HERE/'report/evaluation.json')
    p.add_argument('--probes',action='store_true')
    a=p.parse_args()
    if a.action=='evaluate':evaluate(a.root,a.phase,a.output,a.probes)
    else:export(a.root,a.output)


if __name__=='__main__':main()
