"""Publication figures generated from committed experiment/evaluation records."""
from pathlib import Path
import json
import math
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

COLORS=['#b64038','#e69525','#2b8c75','#6056a5','#2f6c9d','#963a83','#7f8c40','#777777']


def label(name):
    special={'shared_r1':'Shared, equal LR','separate_r1':'Separate, equal LR','global':'Global variance',
             'stein':'Shared, Stein loss','floor02':'Shared, floor 0.2','legacy':'Legacy conditional'}
    if name in special:return special[name]
    return re.sub(r'_r([0-9]+(?:p[0-9]+)?)',lambda m:'; LR ratio '+m[1].replace('p','.'),name).replace('_',' ')


def save(fig,output,name):
    fig.savefig(output/(name+'.pdf'),bbox_inches='tight')
    fig.savefig(output/(name+'.png'),dpi=190,bbox_inches='tight')
    plt.close(fig)


def grouped(rows,phase):
    groups={}
    for r in rows:
        if r['spec']['phase']==phase:groups.setdefault(r['spec']['name'],[]).append(r)
    return groups


def metricbar(ax,groups,names,colors=None):
    for i,name in enumerate(names):
        vals=np.array([r['final']['sw2'] for r in groups[name]])
        color=colors[i%len(colors)] if colors else COLORS[i%len(COLORS)]
        ax.scatter(np.full(len(vals),i)+np.linspace(-.11,.11,len(vals)),vals,s=24,color=color,zorder=3)
        ax.plot([i-.22,i+.22],[vals.mean()]*2,lw=2.4,color=color)
    ax.set_xticks(range(len(names)),[label(n) for n in names],rotation=32,ha='right')
    ax.set_yscale('log');ax.set_ylabel('Final sliced-Wasserstein error')
    ax.grid(axis='y',alpha=.2)


def make(rows,selection,evaluation,theory,root,output):
    output.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'mathtext.fontset':'stix',
                         'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'legend.fontsize':8,
                         'pdf.fonttype':42,'ps.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    sweep=grouped(rows,'sweep');arch=grouped(rows,'architecture');confirm=grouped(rows,'confirmation')
    selected=[s['name'] for s in selection['selected']]
    palette=dict(zip(['shared_r1']+selected+['global','stein','floor02','separate_r1'],COLORS))
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.15),layout='constrained')
    for architecture,color in [('shared',COLORS[0]),('separate',COLORS[2])]:
        configs=sorted([g for g in sweep.values() if g[0]['spec']['architecture']==architecture and
                        g[0]['spec'].get('estimator','original')=='original' and 'floor' not in g[0]['spec'] and g[0]['spec']['ratio']>0],
                       key=lambda g:g[0]['spec']['ratio'])
        xs=[g[0]['spec']['ratio'] for g in configs]
        for j,key in enumerate(['sw2','inverse_v']):
            values=[np.array([r['final'][key] for r in g]) for g in configs]
            mean=np.array([v.mean() for v in values]);sd=np.array([v.std(ddof=1) for v in values])
            axes[j].errorbar(xs,mean,yerr=[np.minimum(sd,.85*mean),sd],color=color,marker='o',lw=1.6,
                             capsize=2,label=architecture.capitalize())
            for x,v in zip(xs,values):axes[j].scatter([x]*len(v),v,s=10,color=color,alpha=.5)
    for name,color in [('global',COLORS[4]),('stein',COLORS[5]),('floor02',COLORS[6])]:
        axes[0].axhline(np.mean([r['final']['sw2'] for r in sweep[name]]),color=color,ls='--',lw=1,label=label(name))
    for ax in axes:ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Variance / mean learning-rate ratio');ax.grid(alpha=.2)
    axes[0].set_ylabel('Final error at 50K updates');axes[1].set_ylabel(r'Mean inverse component variance $E[1/v]$')
    axes[0].legend(fontsize=7);axes[1].legend();save(fig,output,'learning_rate_sweep')

    fig,axes=plt.subplots(2,2,figsize=(7.1,5.2),layout='constrained')
    for ax,architecture,color in zip(axes.flat,['narrow','shallow','linear','stopgrad'],[COLORS[2],COLORS[4],COLORS[3],COLORS[5]]):
        configs=sorted([g for g in arch.values() if g[0]['spec']['architecture']==architecture],key=lambda g:g[0]['spec']['ratio'])
        for base,groups,col,style in [('Separate 128 x 2',sweep,'#999999','--'),(architecture.capitalize(),arch,color,'-')]:
            gg=configs if groups is arch else [g for g in groups.values() if g[0]['spec']['architecture']=='separate' and g[0]['spec']['ratio']>0]
            gg=sorted(gg,key=lambda g:g[0]['spec']['ratio'])
            xx=[g[0]['spec']['ratio'] for g in gg];yy=[np.mean([r['final']['sw2'] for r in g]) for g in gg]
            ax.plot(xx,yy,style,marker='o',color=col,label=base)
            if groups is arch:
                for x,g in zip(xx,gg):ax.scatter([x]*len(g),[r['final']['sw2'] for r in g],s=12,color=col,alpha=.5)
        ax.set_xscale('log');ax.set_yscale('log');ax.set_xlabel('Variance / mean LR ratio');ax.set_ylabel('Final error');ax.legend();ax.grid(alpha=.2)
    save(fig,output,'architecture_ablation')

    names=['shared_r1']+selected+['global','stein','floor02','separate_r1']
    names=list(dict.fromkeys(n for n in names if n in confirm))
    if names:
        fig,ax=plt.subplots(figsize=(7.1,3.8),layout='constrained');metricbar(ax,confirm,names,[palette.get(n,'#555555') for n in names])
        ax.set_title('Held-out seeds 45--49; 50K updates (dots: individual runs; bars: means)')
        save(fig,output,'heldout_comparison')

    names=['shared_r1']+selected[:2]+['global','stein','floor02']
    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for name in names:
        color=palette.get(name,'#555555')
        if name not in sweep:continue
        common=sorted(set.intersection(*[{t['step'] for t in r['trajectory']} for r in sweep[name]]))
        common=[t for t in common if t>=1000]
        for ax,key in zip(axes,['sw2','v_q01']):
            means=[np.mean([next(t[key] for t in r['trajectory'] if t['step']==step) for r in sweep[name]]) for step in common]
            ax.plot(np.array(common)/1000,means,marker='o',ms=3,color=color,label=label(name))
            ax.set_yscale('log');ax.set_xlabel('Optimizer updates (thousands)');ax.axvline(25,ls=':',color='#888888');ax.grid(alpha=.2)
    axes[0].set_ylabel('Mean sliced-Wasserstein error');axes[1].set_ylabel('Mean 1% component-variance quantile')
    axes[0].legend(fontsize=6.5);save(fig,output,'trajectories')

    ablations=grouped(rows,'ablation')
    if ablations:
        fig,axes=plt.subplots(2,2,figsize=(7.1,5.8),layout='constrained')
        tests=[('_noanneal','Disable annealing'),('_init02','Initial variance 0.2'),
               ('_detach','Detach median bandwidth'),('_stein','Stein at selected head LR')]
        for ax,(suffix,title) in zip(axes.flat,tests):
            chosen=[(n,g) for n,g in ablations.items() if n.endswith(suffix)]
            for i,(name,g) in enumerate(chosen):
                parent=g[0]['spec']['parent'];base=sweep.get(parent,{})
                if not base:continue
                old={r['spec']['seed']:r['final']['sw2'] for r in base}
                for r in g:
                    ax.plot([i-.15,i+.15],[old[r['spec']['seed']],r['final']['sw2']],color=COLORS[i%len(COLORS)],marker='o',ms=3,alpha=.75)
            ax.set_xticks(range(len(chosen)),[label(g[0]['spec']['parent']) for _,g in chosen],rotation=25,ha='right',fontsize=7)
            ax.set_title(title+'\n(left: parent; right: changed)');ax.set_yscale('log');ax.set_ylabel('Final error');ax.grid(axis='y',alpha=.2)
        save(fig,output,'atomic_ablations')

    extensions=grouped(rows,'extension')
    if extensions:
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.6),layout='constrained')
        parents=list(dict.fromkeys(r['spec']['parent'] for g in extensions.values() for r in g))
        for parent in parents:
            color=palette.get(parent,'#555555')
            for ax,policy in zip(axes,['decay','restart1e-4']):
                gg=extensions.get(parent+'_'+policy,[])
                if not gg:continue
                times=sorted(set.intersection(*[{t['step'] for t in r['trajectory']} for r in gg]))
                times=[t for t in times if t in (50000,100000,200000)]
                values=[np.mean([next(t['sw2'] for t in r['trajectory'] if t['step']==step) for r in gg]) for step in times]
                ax.plot(np.array(times)/1000,values,'-o',color=color,ms=4,label=label(parent))
                ax.set_yscale('log');ax.set_xlabel('Optimizer updates (thousands)');ax.set_ylabel('Mean error');ax.grid(alpha=.2)
        axes[0].set_title('Continue original decay');axes[1].set_title('Restart at 1e-4 after 50K');axes[0].legend(fontsize=6.5)
        save(fig,output,'extended_budgets')

    fig,axes=plt.subplots(1,2,figsize=(7.1,3.2),layout='constrained')
    for ax,key in zip(axes,['representation_noise','rbf_representation_noise']):
        data=theory[key];x=[r['v'] for r in data]
        for name,color in [('original',COLORS[0]),('stein',COLORS[5])]:
            ax.loglog(x,[r[name+'_variance'] for r in data],'-o',color=color,label=name.capitalize())
            if key=='representation_noise':ax.loglog(x,[r[name+'_exact'] for r in data],'--',color=color,alpha=.6)
        ax.set_xlabel('Component variance v (marginal remains N(0,I))');ax.set_ylabel('Gradient variance');ax.grid(alpha=.2);ax.legend()
    axes[0].set_title('Exact constant-kernel proxy (d=1)');axes[1].set_title('Fixed Gaussian kernel h=0.75 (d=2)')
    save(fig,output,'representation_noise')

    probes=[r for r in evaluation if 'directional_noise' in r]
    chosen=['shared_r1',selected[0],selected[1],'linear_r1','global','stein','floor02','stopgrad_r0p01']
    gg={n:[r for r in probes if r['spec']['name']==n] for n in chosen};gg={n:g for n,g in gg.items() if g}
    if gg:
        fig,axes=plt.subplots(1,2,figsize=(7.1,3.8),layout='constrained')
        for i,(name,g) in enumerate(gg.items()):
            for off,key,color in [(-.16,'original_variance',COLORS[0]),(.16,'stein_variance',COLORS[5])]:
                values=[r['directional_noise'][key] for r in g]
                axes[0].scatter(np.full(len(values),i+off),values,s=17,color=color,label=key.split('_')[0] if i==0 else None)
            drift=[r['freeze_drift']['actual_rms'] for r in g if r.get('freeze_drift')]
            if drift:axes[1].scatter([i]*len(drift),np.maximum(drift,1e-12),color=COLORS[i%len(COLORS)],s=20)
        for ax in axes:ax.set_xticks(range(len(gg)),[label(n) for n in gg],rotation=35,ha='right',fontsize=7);ax.set_yscale('log');ax.grid(axis='y',alpha=.2)
        axes[0].set_ylabel('Dilation-gradient variance, fixed h=0.75');axes[0].legend();axes[1].set_ylabel('Raw variance drift with trained head frozen')
        axes[1].set_title('Standardized local mean LR 0.001; zeros shown at 1e-12')
        save(fig,output,'trained_mechanisms')

    sample_names=['shared_r1',selected[0],selected[1],'linear_r1','global','stein']
    phase='confirmation' if confirm else 'sweep';seeds=[45,46,47] if confirm else [42,43,44]
    if all((root/phase/f'{n}_s{s}'/'samples_50000.npy').exists() for n in sample_names for s in seeds):
        fig,axes=plt.subplots(3,6,figsize=(10.8,5.9),layout='constrained')
        grid=np.linspace(-5,5,180);xx,yy=np.meshgrid(grid,grid)
        density=(np.exp(-(2*xx*xx-3.6*xx*yy+2*yy*yy)/(2*.76))+
                 np.exp(-(2*xx*xx+3.6*xx*yy+2*yy*yy)/(2*.76)))/(4*math.pi*math.sqrt(.76))
        for row,seed in enumerate(seeds):
            for col,name in enumerate(sample_names):
                ax=axes[row,col];x=np.load(root/phase/f'{name}_s{seed}'/'samples_50000.npy')
                ax.hexbin(x[:,0],x[:,1],gridsize=40,extent=(-5,5,-5,5),mincnt=1,cmap='viridis',norm=LogNorm(1,150),rasterized=True)
                ax.contour(xx,yy,density,levels=[.005,.02,.05,.1],colors='black',linewidths=.5)
                ax.set(xlim=(-5,5),ylim=(-5,5),xticks=[-5,0,5],yticks=[-5,0,5]);ax.set_aspect('equal');ax.tick_params(labelsize=6)
                if row==0:ax.set_title(label(name),fontsize=7)
                if col==0:ax.set_ylabel(f'Seed {seed}',fontsize=8)
        save(fig,output,'final_samples')
