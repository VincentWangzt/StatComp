"""Numerical checks of the report's exact toy formulas, independent of training."""
import argparse
from pathlib import Path
import sys

import numpy as np
import torch

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from experiment import ROOT,write_json,revision,loss_from_samples,rng,create_model,loss_at


class StandardGaussian:
    def score(self,x):return -x


def checks(output,repeats=20000,batch=128):
    torch.set_num_threads(1)
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(109837)
    rows=[]
    # q_v=N(0,1) for every v; constant kernel isolates representation noise.
    for v in [.8,.2,.05,.01,.003,.001]:
        e=torch.randn(repeats,batch,device=device,dtype=torch.float64)
        u=torch.randn_like(e);ep=torch.randn_like(e);up=torch.randn_like(e)
        def pair(e,u):
            x=(1-v)**.5*e+v**.5*u
            f=-x+u/v**.5
            df=v/(2*(1-v)**.5)*e-(1+v)/(2*v**.5)*u
            dx=-v/(2*(1-v)**.5)*e+v**.5/2*u
            return x.mean(1),f.mean(1),df.mean(1),dx.mean(1)
        x,f,df,dx=pair(e,u);y,g,dg,dy=pair(ep,up)
        original=df*g+f*dg
        stein=dx*y+x*dy
        theoretical_original=(2+v-v*v)/(2*batch*batch*v*v)
        theoretical_stein=v/(2*(1-v)*batch*batch)
        ov=original.var(unbiased=True).item();sv=stein.var(unbiased=True).item()
        assert abs(ov/theoretical_original-1)<.08,(v,ov,theoretical_original)
        assert abs(sv/theoretical_stein-1)<.08,(v,sv,theoretical_stein)
        rows.append(dict(v=v,original_variance=ov,stein_variance=sv,
                         original_exact=theoretical_original,stein_exact=theoretical_stein,
                         original_mean=original.mean().item(),stein_mean=stein.mean().item()))
    # Same objective and simultaneous updates: separate blocks need not halve speed.
    initial=np.array([1.,2.]);eta=.1;steps=100
    joint=initial.copy();separate=initial.copy()
    for _ in range(steps):
        joint-=eta*joint
        mean,variance=separate.copy()
        separate=np.array([mean-eta*mean,variance-eta*variance])
    assert np.array_equal(joint,separate)
    primary=sum(.001*.9**(t//1000) for t in range(50000))
    extended=sum(.001*.9**(t//1000) for t in range(200000))
    ts=np.linspace(.0001,12,100000)
    v0=float(torch.nn.functional.softplus(torch.tensor(1.)).item())
    target_characteristic=.5*np.exp(-.1*ts**2)+.5*np.exp(-1.9*ts**2)
    lower=((target_characteristic-np.exp(-v0*ts**2/2))/ts).max()
    rbf=[]
    for v in [.2,.05,.01,.003,.001]:
        samples=[]
        with rng(543216):
            for _ in range(512):
                e=torch.randn(batch,2,device=device,dtype=torch.float64);u=torch.randn_like(e)
                ep=torch.randn_like(e);up=torch.randn_like(e)
                theta=torch.zeros((),device=device,dtype=torch.float64,requires_grad=True)
                value=v*theta.exp()
                x=(1-value).sqrt()*e+value.sqrt()*u
                y=(1-value).sqrt()*ep+value.sqrt()*up
                original=loss_from_samples(x,u/value.sqrt(),y,up/value.sqrt(),StandardGaussian(),1.,'original',fixed_h=.75)
                stein=loss_from_samples(x,u/value.sqrt(),y,up/value.sqrt(),StandardGaussian(),1.,'stein',fixed_h=.75)
                gc=torch.autograd.grad(original,theta,retain_graph=True)[0].item()
                gs=torch.autograd.grad(stein,theta)[0].item()
                samples.append([gc,gs])
        array=np.asarray(samples)
        rbf.append(dict(v=v,repeats=512,batch=batch,dimension=2,h=.75,
                        original_mean=float(array[:,0].mean()),stein_mean=float(array[:,1].mean()),
                        original_variance=float(array[:,0].var(ddof=1)),stein_variance=float(array[:,1].var(ddof=1))))
    # A head frozen after it has learned can still move through shared features.
    from analyze import functional_drift
    from models.target_models import X_shaped
    drift=[]
    for architecture in ['shared','separate','stopgrad']:
        spec=dict(seed=42,architecture=architecture,ratio=.1)
        model,opt,_=create_model(spec,device);target=X_shaped(torch.device(device))
        for _ in range(5):
            opt.zero_grad();loss_at(model,target,spec,25000).backward();opt.step()
        measured=functional_drift(model,opt,target,spec,freeze=True)
        assert measured['identity_max_error']<1e-12
        if architecture=='separate':assert measured['actual_rms']==0
        else:assert measured['actual_rms']>0
        drift.append(dict(architecture=architecture,**measured))
    result=dict(source_revision=revision(),device=device,repeats=repeats,batch=batch,representation_noise=rows,
                rbf_representation_noise=rbf,trained_frozen_head_drift=drift,
                quadratic_block_updates_equal=True,step_lr_50k_sum=primary,step_lr_200k_sum=extended,
                additional_fraction=(extended-primary)/primary,
                frozen_initial_variance=v0,projected_wasserstein_lower_bound=float(lower),
                scope='constant-kernel Gaussian proxy; not a theorem of universal RBF gradient variance ordering')
    write_json(output,result);print(result)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=HERE/'report/theory_checks.json')
    a=p.parse_args();checks(a.output)
