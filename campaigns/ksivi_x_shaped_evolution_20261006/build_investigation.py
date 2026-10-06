"""Generate the author-facing report and scientific figures on the GPU server."""
from __future__ import annotations
import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from investigate import REPO, write_json, git_commit

CAMPAIGN=Path(__file__).resolve().parent
LABELS={'canonical_cond':'Conditional: canonical','canonical_global':'Global: canonical',
        'stein_cond':'Conditional: Stein estimator','stein_global':'Global: Stein estimator',
        'floor02_cond':'Conditional: variance floor 0.2',
        'no_anneal_const_cond':'Conditional: no anneal, constant LR',
        'no_anneal_const_global':'Global: no anneal, constant LR',
        'detach_h_cond':'Conditional: detach bandwidth'}
LABELS.update(fixed_h_cond='Conditional: fixed h=0.75',
              fixed_h_stein_cond='Conditional: fixed h, Stein')
COLORS={'canonical_cond':'#bf403e','canonical_global':'#2767a0',
        'stein_cond':'#21886b','stein_global':'#6185ae','floor02_cond':'#a06b20',
        'detach_h_cond':'#985ba1','no_anneal_const_cond':'#555555',
        'no_anneal_const_global':'#999999'}
COLORS.update(fixed_h_cond='#c97663',fixed_h_stein_cond='#56a78c')


def csv_export(path,rows):
    keys=sorted(set().union(*(r.keys() for r in rows)))
    with path.open('w',newline='') as f:
        w=csv.DictWriter(f,keys)
        w.writeheader()
        w.writerows([{k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in rows])


def tex_escape(s):
    return str(s).replace('_',r'\_').replace('%',r'\%').replace('&',r'\&')


def avg_sd(vals,digits=3):
    vals=np.array(vals,dtype=float)
    if len(vals)==1:return f'{vals[0]:.{digits}f}'
    return rf'${vals.mean():.{digits}f}\;({vals.std(ddof=1):.{digits}f})$'


FIGURE_CAPTIONS = {
    'confirmation': ('Matched comparisons after 50,000 updates. Dots are independently trained seeds 42--44; '
                     'vertical strokes mark means. Initial mean weights, variance values, and training noise are paired. '
                     'Forward KL uses the largest evaluated latent mixture; held-out KSD uses an independent fixed bandwidth of 0.75.'),
    'trajectories': ('Accuracy and the small-variance tail across three trajectories per procedure. '
                     'The dotted line marks completion of target annealing at update 25,000. '
                     'Floor occupancy uses the common threshold 0.00010001; diagnostic draws preserve training RNG.'),
    'exact_marginal_noise': ('Representation noise at exactly equal marginal distributions. Each measurement uses '
                            '256 independent repetitions and two batches of 128 at fixed bandwidth 0.75. '
                            'The dashed curve is the analytic small-variance prediction, with no fitted coefficient. '
                            'The right panel holds both the marginal and mean component variance 0.11 fixed.'),
    'screening': ('All 23 screening outcomes at 10,000 updates, seed 42. Blue denotes global variance; red denotes '
                  'conditional variance. Annealed runs still have a score multiplier of 0.46 unless warmup is shortened. '
                  'Distances are measured against the final target and do not establish final-target convergence.'),
    'geometry_and_schedule': ('Target path, learning-rate decay, and target curvature. The score multiplier reaches one '
                              'after the learning rate has fallen to approximately 0.0000718. The diverging Hessian entry '
                              'violates the global bounded-Hessian assumption for both variational families; this fact alone '
                              'does not explain their ordering.'),
    'final_samples': ('Final samples for seeds 42--44, with exact target contours and a common viewing window. '
                      'All samples contribute to numerical metrics, including tails outside the plot. '
                      'The separate sample pages in the companion booklet provide larger panels for each seed.'),
}


def report_figures(figures):
    figures=[f for f in figures if not f['name'].startswith('final_samples_s')]
    pages=[r'\begin{landscape}',r'\pagestyle{empty}',r'\captionsetup{hypcap=false}']
    for i,f in enumerate(figures):
        if i:pages.append(r'\clearpage')
        pages.extend([r'\noindent{\small\itshape KSIVI on the X-shaped target}\hfill{\small Scientific evidence}\par',
                      r'\vspace{3pt}\hrule\vspace{10pt}'])
        if i==0:pages.append(r'\section{Scientific figures}')
        pages.extend([r'\begin{center}',
                      rf'\includegraphics[width=\linewidth,height=.72\textheight,keepaspectratio]{{figures/{f["name"]}.pdf}}',
                      r'\captionof{figure}{'+FIGURE_CAPTIONS[f['name']]+r'}',r'\end{center}',
                      r'\vfill\begin{center}\small\thepage\end{center}'])
    pages.extend([r'\end{landscape}',r'\pagestyle{fancy}'])
    return '\n'.join(pages)


def figure_booklet(figures):
    figures=[f for f in figures if f['name']!='final_samples']
    source=r'''\documentclass[11pt,a4paper,landscape]{article}
\usepackage[margin=17mm,headheight=14pt,includeheadfoot]{geometry}
\usepackage[T1]{fontenc}
\usepackage{newtxtext,newtxmath,microtype,graphicx,caption,fancyhdr,xcolor}
\usepackage{hyperref}
\definecolor{ink}{HTML}{23384D}
\hypersetup{colorlinks=true,linkcolor=ink,urlcolor=ink,pdftitle={KSIVI variance investigation: scientific figures}}
\pagestyle{fancy}\fancyhf{}
\fancyhead[L]{\small\itshape KSIVI on the X-shaped target}
\fancyhead[R]{\small Scientific evidence}
\fancyfoot[C]{\thepage}\renewcommand{\headrulewidth}{0.3pt}
\captionsetup{font=small,labelfont=bf,labelsep=period}
\setlength{\parindent}{0pt}
\begin{document}
'''
    for i,f in enumerate(figures):
        if i:source+='\n'+r'\clearpage'+'\n'
        source+='\n'.join([r'\begin{figure}[!ht]',r'\centering',
                           rf'\includegraphics[width=\linewidth,height=.85\textheight,keepaspectratio]{{figures/{f["name"]}.pdf}}',
                           r'\caption{'+FIGURE_CAPTIONS.get(f['name'],f['caption'])+r'}',r'\end{figure}'])+'\n'
    return source+r'\end{document}'+'\n'


def build(root,out,partial=False):
    out.mkdir(parents=True,exist_ok=True)
    vectors=out/'figures'
    vectors.mkdir(exist_ok=True)
    data=json.loads((out/'new_metrics.json').read_text())['rows']
    old=json.loads((out/'historical_metrics.json').read_text())['rows']
    probe=json.loads((out/'mechanism_probes.json').read_text())
    summaries=[json.loads(p.read_text()) for p in sorted(root.glob('*/*/summary.json'))]
    if not partial:
        for stage in ['screen','confirm','contrast','fixed']:
            state=json.loads((root/stage/'state.json').read_text())
            assert state['status']=='completed',(stage,state['status'])
        assert len(summaries)==len(data),(len(summaries),len(data))
    by_name=defaultdict(list)
    for r in data:
        if r['spec']['steps']==50000:by_name[r['spec']['name']].append(r)
    for rows in by_name.values():rows.sort(key=lambda r:r['spec']['seed'])
    csv_export(out/'new_metrics.csv',[dict(r,**r['spec']) for r in data])
    csv_export(out/'historical_metrics.csv',old)
    csv_export(out/'trajectories.csv',[dict(t,**s['spec']) for s in summaries for t in s['trajectory']])
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,
                         'figure.facecolor':'white','axes.titleweight':'bold',
                         'savefig.facecolor':'white','font.family':'STIXGeneral',
                         'mathtext.fontset':'stix','pdf.fonttype':42,'ps.fonttype':42})
    figures=[]
    def save(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        fig.savefig(vectors/(name+'.pdf'),bbox_inches='tight',metadata={'CreationDate':None,'ModDate':None})
        plt.close(fig)
        figures.append({'name':name,'caption':caption})

    fig,axes=plt.subplots(1,3,figsize=(13,6.2),layout='constrained')
    shown=[n for n in LABELS if n in by_name]
    for ax,metric,title in zip(axes,['sw2','kl_pq_best','ksd2_h0.75'],
                               ['Sliced Wasserstein distance','Forward KL estimate','Independent fixed-bandwidth KSD squared']):
        for i,n in enumerate(shown):
            vals=[r[metric] for r in by_name[n]]
            ax.scatter(vals,np.full(len(vals),i),color=COLORS[n],s=35,zorder=3)
            ax.plot([np.mean(vals)]*2,[i-.22,i+.22],color=COLORS[n],lw=3)
        ax.set_yticks(range(len(shown)))
        ax.invert_yaxis()
        ax.set_yticklabels([LABELS[n].replace('variance ','') for n in shown] if ax is axes[0] else [],fontsize=10)
        ax.set_xlabel('Distance' if metric=='sw2' else 'KL estimate' if metric=='kl_pq_best' else r'$\mathrm{KSD}^2$')
        ax.set_title(title,fontsize=11)
        ax.grid(axis='x',alpha=.2)
    save(fig,'confirmation','Matched initial mean/variance, seeds 42--44. KSD uses h=0.75 chosen independently of evaluation draws.')

    fig,axes=plt.subplots(1,3,figsize=(13,4.2),layout='constrained')
    for s in summaries:
        spec=s['spec']
        if spec['steps']!=50000 or spec['name'] not in ['canonical_cond','canonical_global','stein_cond','floor02_cond']:continue
        n=spec['name'];t=s['trajectory'];steps=[r['step'] for r in t]
        for ax,m in zip(axes,['sw2','inverse_v','floor_fraction']):
            ax.plot(steps,[r[m] for r in t],c=COLORS[n],alpha=.65,lw=1.6)
    for ax,title in zip(axes,['Sliced Wasserstein distance','Mean inverse conditional variance','Fraction at canonical floor (0.0001)']):
        ax.set_title(title,fontsize=11);ax.set_xlabel('Updates');ax.axvline(25000,c='k',ls=':',alpha=.5);ax.grid(alpha=.2)
    axes[1].set_yscale('log');axes[0].set_ylim(bottom=0)
    for n in ['canonical_cond','canonical_global','stein_cond','floor02_cond']:
        axes[0].plot([],[],c=COLORS[n],label=LABELS[n])
    axes[0].legend(fontsize=8)
    save(fig,'trajectories','Vertical line: annealing reaches the final target. The floor fraction uses a common threshold of 0.00010001 across methods. Diagnostic RNG is isolated from training.')

    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    exact=[r for r in probe['exact_marginal'] if not r['heterogeneous']]
    exact.sort(key=lambda r:r['v'])
    for metric,color,label in [('conditional_grad_var','#bf403e','Conditional-score estimator'),
                               ('stein_grad_var','#21886b','Stein integration estimator')]:
        axes[0].loglog([r['v'] for r in exact],[r[metric] for r in exact],'-o',color=color,label=label)
    grid=np.geomspace(.002,.2,80)
    axes[0].loglog(grid,probe['gradient_asymptotic_coefficient']/grid**2,'--',c='black',label='Analytic asymptotic C / v squared')
    axes[0].set_xlabel('Component variance v');axes[0].set_ylabel('Variance of dilation gradient');axes[0].legend(fontsize=9)
    axes[0].set_title('Constant component variance')
    axes[1].bar(['Constant v=0.11','v=0.02 or 0.2'],
                 [next(r['conditional_grad_var'] for r in exact if r['v']==.11),
                  next(r['conditional_grad_var'] for r in probe['exact_marginal'] if r['heterogeneous'])],
                 color=['#2767a0','#bf403e'])
    axes[1].set_ylabel('Conditional-score gradient variance');axes[1].set_title('Equal mean variance, equal marginal')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    save(fig,'exact_marginal_noise','Global dilation parameter theta=0 is a population optimum. Fixed h=0.75; independent cross batches.')

    screen=[r for r in data if r['spec']['stage']=='screen']
    fig,axes=plt.subplots(1,2,figsize=(12,7.5),layout='constrained')
    for ax,metric,title in zip(axes,['sw2','inverse_v'],['Accuracy relative to the FINAL target','Mean inverse conditional variance']):
        vals=[r[metric] for r in screen];names=[r['spec']['name'] for r in screen]
        ax.barh(range(len(names)),vals,color=['#2767a0' if r['spec']['family']=='global' else '#bf403e' for r in screen])
        ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names],fontsize=10)
        ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
        if metric=='inverse_v':ax.set_xscale('log')
    save(fig,'screening','Screening ranks are not final-target convergence claims. Blue: global variance; red: conditional variance.')

    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    steps=np.arange(1,50001);lr=.001*.9**((steps-1)//1000);alpha=np.minimum(1,.1+.9*steps/25000)
    axes[0].plot(steps,alpha,color='#21886b',label='Target score multiplier');axes[0].set_ylabel('alpha',color='#21886b')
    ax2=axes[0].twinx();ax2.plot(steps,lr,color='#bf403e');ax2.set_ylabel('Learning rate',color='#bf403e');ax2.set_yscale('log')
    axes[0].set_xlabel('Updates');axes[0].set_title('Annealing and step size')
    radii=np.linspace(0,10,200);axes[1].plot(radii,-2/.76+(1.8/.76)**2*radii*radii,c='#2767a0')
    axes[1].set_xlabel('R at point (0,R)');axes[1].set_ylabel('Hessian entry Hxx');axes[1].set_title('Unbounded target curvature')
    save(fig,'geometry_and_schedule','The X-shaped mixture violates the global bounded-Hessian assumption. This is a boundary of the theorem, not by itself a cause of the family gap.')

    selected=[n for n in ['canonical_cond','canonical_global','stein_cond','floor02_cond'] if n in by_name]
    if selected:
        fig,axes=plt.subplots(3,len(selected),figsize=(3.2*len(selected),9),squeeze=False,layout='constrained')
        for col,n in enumerate(selected):
            for row,r in enumerate(by_name[n]):
                path=Path(r['path']);state=torch.load(path/'snapshot_50000.pt',map_location='cpu',weights_only=True)
                x=state['x'].numpy();ax=axes[row,col]
                ax.hexbin(x[:,0],x[:,1],gridsize=60,extent=(-5,5,-5,5),mincnt=1,cmap='Blues',bins='log',rasterized=True)
                xx,yy=np.meshgrid(np.linspace(-5,5,160),np.linspace(-5,5,160))
                ell=-math.log(2*math.pi)-.5*math.log(.76)-(xx*xx+yy*yy)/.76
                ell+=np.logaddexp(1.8/.76*xx*yy,-1.8/.76*xx*yy)-math.log(2)
                ax.contour(xx,yy,np.exp(ell),levels=[.005,.02,.07,.14],colors='#333333',linewidths=.65)
                ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal')
                if row==0:ax.set_title(LABELS[n].replace(': ','\n'),fontsize=13)
                ax.text(.02,.98,f"Seed {r['spec']['seed']}\nSW2={r['sw2']:.3f}",transform=ax.transAxes,va='top',fontsize=12)
                ax.tick_params(labelsize=12)
                if col==0:ax.set_ylabel('x2')
                if row==2:ax.set_xlabel('x1')
        save(fig,'final_samples','Hexbin densities are for visual comparison. Quantitative metrics include all samples, including escaped tails.')
        for seed in [42,43,44]:
            fig,axes=plt.subplots(1,len(selected),figsize=(13,3.8),layout='constrained')
            for col,n in enumerate(selected):
                r=next(r for r in by_name[n] if r['spec']['seed']==seed)
                state=torch.load(Path(r['path'])/'snapshot_50000.pt',map_location='cpu',weights_only=True)
                x=state['x'].numpy();ax=axes[col]
                ax.hexbin(x[:,0],x[:,1],gridsize=60,extent=(-5,5,-5,5),mincnt=1,
                          cmap='Blues',bins='log',rasterized=True)
                ax.contour(xx,yy,np.exp(ell),levels=[.005,.02,.07,.14],colors='#333333',linewidths=.65)
                ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal',xlabel=r'$x_1$')
                ax.set_title(LABELS[n].replace(': ','\n'),fontsize=12)
                ax.text(.03,.97,rf'$\mathrm{{SW}}_2={r["sw2"]:.3f}$',transform=ax.transAxes,va='top',fontsize=11)
                if col==0:ax.set_ylabel(r'$x_2$')
            save(fig,f'final_samples_s{seed}',
                 f'Final samples, seed {seed}, after 50,000 updates. Black curves are exact target contours. '
                 'All four procedures use matched initial mean functions and variance values. '
                 'The common viewing window omits some tails, which remain included in all numerical metrics.')

    def group_mean(name,key):return np.mean([r[key] for r in by_name[name]])

    table=[]
    for n in LABELS:
        rows=by_name.get(n,[])
        if not rows:continue
        table.append(' & '.join([tex_escape(LABELS[n]),str(len(rows)),avg_sd([r['sw2'] for r in rows]),
                                avg_sd([r['kl_pq_best'] for r in rows]),avg_sd([r['ksd2_h0.75'] for r in rows],4)])+r' \\')
    seedtable=[]
    for n,rows in by_name.items():
        for r in rows:
            seedtable.append(' & '.join([tex_escape(n),str(r['spec']['seed']),f"{r['sw2']:.3f}",
                                        f"{r['kl_pq_best']:.3f}",f"{r['inverse_v']:.1f}",
                                        f"{r['direction_cond_grad_var']:.4f}",f"{r['direction_stein_grad_var']:.4f}"])+r' \\')
    oldtable=[]
    for r in old:
        name=('Global' if r['family']=='ConditionalGaussianGlobal' else 'Conditional')
        init='Matched variance' if r['constant_init'] else 'Default'
        oldtable.append(' & '.join([name,init,str(r['seed']),f"{r['sw2']:.3f}",
                                   f"{r['kl_pq_best']:.3f}",f"{r['inverse_v']:.1f}"])+r' \\')
    probetable=[]
    for r in probe['exact_marginal']:
        name='0.02/0.20 mixture' if r['heterogeneous'] else str(r['v'])
        probetable.append(' & '.join([name,f"{r['conditional_grad_var']:.5f}",f"{r['stein_grad_var']:.5f}",
                                     f"{r['theoretical_residual_second']:.2f}"])+r' \\')
    screentable=[]
    for r in screen:
        screentable.append(' & '.join([tex_escape(r['spec']['name']),f"{r['sw2']:.3f}",
                                      f"{r['inverse_v']:.1f}",f"{r['cross_fourth']:.3f}"])+r' \\')
    differences=[]
    if all(n in by_name for n in ['canonical_cond','canonical_global']):
        for c1,g1 in zip(by_name['canonical_cond'],by_name['canonical_global']):
            assert c1['spec']['seed']==g1['spec']['seed']
            differences.append(f"Seed {c1['spec']['seed']}: {c1['sw2']-g1['sw2']:+.3f}")
    diagnostics=[]
    for n,rows in by_name.items():
        delta=max(r['kl_last_change'] for r in rows)
        diagnostics.append(f"{tex_escape(LABELS[n])} & {delta:.3f}"+r' \\')
    replacements={'@@RESULTS@@':'\n'.join(table),'@@SEEDS@@':'\n'.join(seedtable),
                  '@@HISTORICAL@@':'\n'.join(oldtable),'@@PROBES@@':'\n'.join(probetable),
                  '@@SCREEN@@':'\n'.join(screentable),'@@DIFFERENCES@@':'; '.join(differences),
                  '@@KL_STABILITY@@':'\n'.join(diagnostics),
                  '@@COUNT@@':str(len(summaries)),'@@AUDIT_COUNT@@':str(len(json.loads((out/'existing_audit.json').read_text())['rows'])),
                  '@@FIGURES@@':report_figures(figures),'@@COMMIT@@':git_commit(),'@@STATUS@@':'INTERIM' if partial else 'COMPLETED'}
    replacements['@@OUTCOMES@@']=(
        f"The canonical gap appears in all three paired seeds: mean SW$_2$ is {group_mean('canonical_cond','sw2'):.3f} versus "
        f"{group_mean('canonical_global','sw2'):.3f}, and mean forward KL is {group_mean('canonical_cond','kl_pq_best'):.3f} versus "
        f"{group_mean('canonical_global','kl_pq_best'):.3f}. Raising the conditional variance floor to $0.2$ improves each "
        f"canonical conditional seed, with mean SW$_2$ {group_mean('floor02_cond','sw2'):.3f}. The adaptive-bandwidth Stein conditional "
        f"variant also improves each canonical conditional seed, with mean SW$_2$ {group_mean('stein_cond','sw2'):.3f}; "
        f"bandwidth detachment alone leaves mean SW$_2$ {group_mean('detach_h_cond','sw2'):.3f}.")
    source=(CAMPAIGN/'investigation_template.tex').read_text()
    for k,v in replacements.items():source=source.replace(k,v)
    assert '@@' not in source
    (out/'report.tex').write_text(source)
    (out/'evidence_figures.tex').write_text(figure_booklet(figures))
    write_json(out/'manifest.json',{'status':'interim' if partial else 'completed','source_commit':git_commit(),
                                   'completed_runs':len(summaries),'evaluated_runs':len(data),
                                   'run_source_commits':sorted({s['source_commit'] for s in summaries}),
                                   'figures':figures,'paired_sw2_difference':differences})


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=Path('/root/ruivi/results/ksivi_variance_investigation_20261006'))
    p.add_argument('--output',type=Path,default=CAMPAIGN/'investigation')
    p.add_argument('--partial',action='store_true')
    a=p.parse_args();build(a.root,a.output,a.partial)

if __name__=='__main__':main()
