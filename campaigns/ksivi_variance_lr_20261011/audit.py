"""Regenerate saved samples and independently check final losses/gradients."""
from pathlib import Path
import argparse
import json
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment import ROOT,rng,write_json,revision,diagnostics,loss_from_samples,specifications
from analyze import load
from models.target_models import X_shaped
from utils.kernels import GaussianKernel


def audit(root,output):
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    records=[];coverage={}
    for phase in ['sweep','architecture','confirmation','ablation','extension']:
        expected=specifications(phase,root)
        paths=list((root/phase).glob('*/result.json'))
        coverage[phase]=dict(expected=len(expected),completed=len(paths))
        assert len(paths)==len(expected),(phase,len(paths),len(expected))
        for path in sorted(paths):
            completed=json.loads(path.read_text());spec=completed['spec']
            assert spec['steps']==(200000 if phase=='extension' else 50000)
            steps=[50000] if phase!='extension' else [100000,200000]
            for step in steps:
                model,_,_,spec=load(path.parent,step,device);target=X_shaped(device)
                metric,samples=diagnostics(model,target)
                saved=np.load(path.parent/f'samples_{step}.npy')
                assert np.array_equal(saved,samples.numpy()),(path,step,'samples')
                quick=next(r for r in completed['trajectory'] if r['step']==step)
                error=max(abs(metric[k]-quick[k]) for k in metric)
                assert error<1e-7,(path,step,error)
                with rng(815731):
                    e1,e2=model.sample_epsilon(12),model.sample_epsilon(12)
                    x,b=model(e1);y,c=model(e2)
                    sx,sy=target.score(x),target.score(y)
                    estimator=spec.get('estimator','original')
                    kernel=GaussianKernel()
                    k=kernel.pair_eval(x,y,fit_h=True,detach_h=spec.get('detach_h',False) or estimator=='stein')
                    if estimator=='original':reference=(((sx+b)@(sy+c).T)*k).mean()
                    else:
                        # Compute the Stein operator by actual spatial autograd.
                        delta=x[:,None]-y[None,:]
                        h=kernel.h
                        spatial_k=(-delta.square().sum(-1)/(2*h*h)).exp()
                        gradient=torch.autograd.grad(spatial_k.sum(),delta,create_graph=True)[0]
                        trace=-sum(torch.autograd.grad(gradient[...,j].sum(),delta,create_graph=True,retain_graph=True)[0][...,j] for j in range(2))
                        reference=(spatial_k*(sx@sy.T)+(-sx[:,None]*gradient+sy[None]*gradient).sum(-1)+trace).mean()
                    actual=loss_from_samples(x,b,y,c,target,1.,estimator,spec.get('detach_h',False))
                    params=list(model.parameters())
                    ga=torch.autograd.grad(actual,params,retain_graph=True,allow_unused=True)
                    gr=torch.autograd.grad(reference,params,allow_unused=True)
                    av=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,ga)])
                    rv=torch.cat([(torch.zeros_like(p) if g is None else g).flatten() for p,g in zip(params,gr)])
                    rel=(av-rv).norm().item()/max(rv.norm().item(),1e-8)
                    absolute=(av-rv).abs().max().item()
                    assert rel<3e-4 or absolute<2e-6,(path,step,rel,absolute)
                    records.append(dict(phase=phase,name=spec['name'],seed=spec['seed'],step=step,samples_exact=True,
                                        metrics_max_error=error,gradient_relative_error=rel,gradient_max_absolute_error=absolute,
                                        loss_absolute_error=abs(actual.item()-reference.item())))
                print('AUDIT',phase,spec['name'],spec['seed'],step,rel,flush=True)
    legacy=[]
    for seed in [42,43,44]:
        for step in [0,1000,5000,10000,25000,50000]:
            a,_,_,_=load(root/'sweep'/f'legacy_s{seed}',step or 0,device)
            b,_,_,_=load(root/'sweep'/f'shared_r1_s{seed}',step or 0,device)
            aa=a.net.state_dict()
            bb=b.mean_backbone.state_dict()
            assert all(torch.equal(aa[k],v) for k,v in bb.items()),(seed,step,'backbone')
            assert torch.equal(a.net[-1].weight,torch.cat([b.mean_head.weight,b.variance_head.weight])),(seed,step,'weight')
            assert torch.equal(a.net[-1].bias,torch.cat([b.mean_head.bias,b.variance_head.bias])),(seed,step,'bias')
            legacy.append(dict(seed=seed,step=step,parameters_exact=True))
    extensions=[]
    for path in sorted((root/'extension').glob('*/result.json')):
        spec=json.loads(path.read_text())['spec']
        child=torch.load(path.parent/'checkpoint_50000.pt',map_location='cpu',weights_only=True)
        parent=root/spec.get('parent_phase','sweep')/f"{spec['parent']}_s{spec['seed']}"/'checkpoint_50000.pt'
        base=torch.load(parent,map_location='cpu',weights_only=True)
        assert all(torch.equal(v,base['model'][k]) for k,v in child['model'].items())
        assert torch.equal(child['cpu_rng'],base['cpu_rng'])
        assert all(torch.equal(x,y) for x,y in zip(child['cuda_rng'],base['cuda_rng']))
        for key,state in child['optimizer']['state'].items():
            assert all(torch.equal(v,base['optimizer']['state'][key][k]) for k,v in state.items())
        extensions.append(dict(name=spec['name'],seed=spec['seed'],model_rng_and_adam_moments_exact=True))
    write_json(output,dict(source_revision=revision(),device=str(device),coverage=coverage,records=records,
                           legacy_shared_full_trajectories=legacy,extension_checkpoint_continuity=extensions,
                           maximum_gradient_relative_error=max(r['gradient_relative_error'] for r in records),passed=True))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=HERE/'report/audit.json')
    a=p.parse_args();audit(a.root,a.output)
