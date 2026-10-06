"""Generate the author-facing report and scientific figures on the GPU server."""
from __future__ import annotations
import argparse
import csv
import json
import math
import io
import textwrap
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


def build(root,out,partial=False):
    out.mkdir(parents=True,exist_ok=True)
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
                         'savefig.facecolor':'white'})
    figures=[]
    def save(fig,name,caption):
        fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight')
        plt.close(fig)
        figures.append({'name':name,'caption':caption})

    fig,axes=plt.subplots(1,3,figsize=(14,4.8),layout='constrained')
    shown=[n for n in LABELS if n in by_name]
    for ax,metric,title in zip(axes,['sw2','kl_pq_best','ksd2_h0.75'],
                               ['Sliced Wasserstein distance','Forward KL estimate','Independent fixed-bandwidth KSD squared']):
        for i,n in enumerate(shown):
            vals=[r[metric] for r in by_name[n]]
            ax.scatter(np.full(len(vals),i),vals,color=COLORS[n],s=35,zorder=3)
            ax.plot([i-.22,i+.22],[np.mean(vals)]*2,color=COLORS[n],lw=3)
        ax.set_xticks(range(len(shown)),[LABELS[n].replace(': ','\n') for n in shown],rotation=65,ha='right',fontsize=8)
        ax.set_title(title,fontsize=11)
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('50,000 updates: each dot is one seed; horizontal strokes are means',fontsize=14)
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
    fig.suptitle('Training path and the small-variance tail (three independent trajectories per method)',fontsize=13)
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
    axes[0].set_title('Exactly the same marginal p in every experiment')
    axes[1].bar(['Constant v=0.11','v=0.02 or 0.2'],
                 [next(r['conditional_grad_var'] for r in exact if r['v']==.11),
                  next(r['conditional_grad_var'] for r in probe['exact_marginal'] if r['heterogeneous'])],
                 color=['#2767a0','#bf403e'])
    axes[1].set_ylabel('Variance of conditional-score dilation gradient');axes[1].set_title('Same mean variance 0.11, same marginal p')
    for ax in axes:ax.grid(axis='y',alpha=.2)
    fig.suptitle('Representation noise separated from approximation error: 256 independent batches, N=128',fontsize=12)
    save(fig,'exact_marginal_noise','Global dilation parameter theta=0 is a population optimum. Fixed h=0.75; independent cross batches.')

    screen=[r for r in data if r['spec']['stage']=='screen']
    fig,axes=plt.subplots(1,2,figsize=(12,7.5),layout='constrained')
    for ax,metric,title in zip(axes,['sw2','inverse_v'],['Accuracy relative to the FINAL target','Mean inverse conditional variance']):
        vals=[r[metric] for r in screen];names=[r['spec']['name'] for r in screen]
        ax.barh(range(len(names)),vals,color=['#2767a0' if r['spec']['family']=='global' else '#bf403e' for r in screen])
        ax.set_yticks(range(len(names)),[n.replace('_',' ') for n in names],fontsize=8)
        ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2)
        if metric=='inverse_v':ax.set_xscale('log')
    fig.suptitle('Broad screening: seed 42, 10,000 updates (annealed runs have alpha=0.46)',fontsize=13)
    save(fig,'screening','Screening ranks are not final-target convergence claims. Blue: global variance; red: conditional variance.')

    fig,axes=plt.subplots(1,2,figsize=(10,4.3),layout='constrained')
    steps=np.arange(1,50001);lr=.001*.9**((steps-1)//1000);alpha=np.minimum(1,.1+.9*steps/25000)
    axes[0].plot(steps,alpha,color='#21886b',label='Target score multiplier');axes[0].set_ylabel('alpha',color='#21886b')
    ax2=axes[0].twinx();ax2.plot(steps,lr,color='#bf403e');ax2.set_ylabel('Learning rate',color='#bf403e');ax2.set_yscale('log')
    axes[0].set_xlabel('Updates');axes[0].set_title('Moving target and shrinking adaptation budget')
    radii=np.linspace(0,10,200);axes[1].plot(radii,-2/.76+(1.8/.76)**2*radii*radii,c='#2767a0')
    axes[1].set_xlabel('R at point (0,R)');axes[1].set_ylabel('Hessian entry Hxx');axes[1].set_title('Target curvature is globally unbounded')
    save(fig,'geometry_and_schedule','The X-shaped mixture violates the global bounded-Hessian assumption. This is a boundary of the theorem, not by itself a cause of the family gap.')

    selected=[n for n in ['canonical_cond','canonical_global','stein_cond','floor02_cond'] if n in by_name]
    if selected:
        fig,axes=plt.subplots(3,len(selected),figsize=(3.2*len(selected),9),squeeze=False,layout='constrained')
        for col,n in enumerate(selected):
            for row,r in enumerate(by_name[n]):
                path=Path(r['path']);state=torch.load(path/'snapshot_50000.pt',map_location='cpu',weights_only=True)
                x=state['x'].numpy();ax=axes[row,col]
                ax.hexbin(x[:,0],x[:,1],gridsize=60,extent=(-5,5,-5,5),mincnt=1,cmap='Blues',bins='log')
                xx,yy=np.meshgrid(np.linspace(-5,5,160),np.linspace(-5,5,160))
                ell=-math.log(2*math.pi)-.5*math.log(.76)-(xx*xx+yy*yy)/.76
                ell+=np.logaddexp(1.8/.76*xx*yy,-1.8/.76*xx*yy)-math.log(2)
                ax.contour(xx,yy,np.exp(ell),levels=[.005,.02,.07,.14],colors='#333333',linewidths=.65)
                ax.set(xlim=(-5,5),ylim=(-5,5),aspect='equal')
                if row==0:ax.set_title(LABELS[n].replace(': ','\n'),fontsize=10)
                ax.text(.02,.98,f"Seed {r['spec']['seed']}\nSW2={r['sw2']:.3f}",transform=ax.transAxes,va='top',fontsize=9)
                if col==0:ax.set_ylabel('x2')
                if row==2:ax.set_xlabel('x1')
        fig.suptitle('Final samples, common viewing window; numerical tail metrics report mass outside this window',fontsize=13)
        save(fig,'final_samples','Hexbin densities are for visual comparison. Quantitative metrics include all samples, including escaped tails.')

    # Combine figure pages through reportlab; keep the mathematical report
    # standalone so the Codex built-in LaTeX compiler requires no project files.
    from reportlab.platypus import (BaseDocTemplate,PageTemplate,Frame,Paragraph,
                                   Spacer,Image,Table,TableStyle,PageBreak,NextPageTemplate)
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib import colors
    from PIL import Image as PILImage
    styles=getSampleStyleSheet()
    styles['BodyText'].fontSize=10.5;styles['BodyText'].leading=14.5;styles['BodyText'].spaceAfter=7
    styles['Heading1'].fontSize=15;styles['Heading1'].leading=18
    styles['Title'].fontSize=23;styles['Title'].leading=28
    doc=BaseDocTemplate(str(out/'evidence_figures.pdf'),pagesize=(595,842),
                        title='KSIVI on the X-shaped target: theory and evidence',
                        author='StatComp variance investigation')
    def footer(c,d):
        c.saveState();c.setFont('Helvetica',8)
        c.drawString(36,22,'KSIVI variance investigation | campaign 20261006')
        c.drawRightString(c._pagesize[0]-36,22,str(d.page));c.restoreState()
    doc.addPageTemplates([
        PageTemplate(id='Report',pagesize=(595,842),frames=[Frame(36,40,523,750,id='r')],onPage=footer),
        PageTemplate(id='Figures',pagesize=(1008,720),frames=[Frame(30,40,948,640,id='f')],onPage=footer)])
    story=[]
    def para(s,style='BodyText'):story.append(Paragraph(s,styles[style]))
    def equation(s):
        fig=plt.figure(figsize=(7,.55))
        fig.text(.5,.5,'$'+s+'$',ha='center',va='center',fontsize=15)
        buf=io.BytesIO();fig.savefig(buf,format='png',dpi=220,bbox_inches='tight',pad_inches=.05);plt.close(fig)
        buf.seek(0);im=PILImage.open(buf);w,h=im.size;scale=min(1,505/(w/220*72))
        story.append(Image(buf,width=w/220*72*scale,height=h/220*72*scale));story.append(Spacer(1,6))
    def result_table(rows,headers,widths):
        cells=[[Paragraph(str(c),styles['BodyText']) for c in row] for row in [headers]+rows]
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LINEBELOW',(0,0),(-1,0),1,colors.HexColor('#2767a0')),
                               ('LINEBELOW',(0,-1),(-1,-1),.7,colors.grey),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f2f5f7')]),
                               ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5)]))
        story.append(t);story.append(Spacer(1,12))
    para('When conditional variance harms KSIVI on the X-shaped target','Title')
    para('Geometry, representation noise, and optimization boundaries','Heading2')
    para(('INTERIM REVIEW COPY. ' if partial else '')+
         f'{len(summaries)} new remote runs; three-seed long comparisons; {len(json.loads((out/"existing_audit.json").read_text())["rows"])} historical checkpoints audited.')
    para('The main conclusion','Heading1')
    para('The gap is a failure mode of finite stochastic optimization, not an intrinsic inferiority of the conditional variational family. '
         'The larger family contains the global family, and a fixed-kernel population KSD depends only on the marginal distribution. '
         'For this target, a global variance of 0.2 already supplies exactly the needed transverse width. Conditional scale directions can '
         'instead develop small-variance tails that sharply increase conditional-score gradient noise. The optimizer and moving target determine whether that noise leads to failure.')
    para('The evidence is deliberately qualified: not every conditional run fails, not every global run succeeds, and the intervention results do not imply a universal preference for the Stein estimator. '
         'The full standalone manuscript, report.tex, supplies additional derivations, per-seed diagnostics, and evaluation details.')
    para('The exact comparison','Heading1')
    para('Canonical policy: two width-128 SiLU hidden layers, Gaussian latent dimension 2, softplus conditional variances with floor 0.0001; '
         'Adam (0.9, 0.999), learning rate 0.001, StepLR(1000, 0.9), two independent batches of 128, and 50,000 updates. '
         'The target-score multiplier moves linearly from 0.1 to 1 over 25,000 updates. Gaussian spatial kernels and the empirical median bandwidth are differentiated. '
         'New comparisons match mean weights and initial variance to float32 roundoff and pair the training noise. Diagnostics preserve training RNG.')
    para('Long comparisons: means (sample standard deviations across trained seeds). SW2 is sliced Wasserstein distance; KL is forward KL estimated from exact target samples. '
         'Independent KSD uses a fixed h=0.75. Each trained model, not each evaluation sample, is one statistical replicate.')
    rows=[]
    for n in LABELS:
        rr=by_name.get(n,[])
        if not rr:continue
        def plain(k,prec=3):
            a=np.array([r[k] for r in rr]);return f'{a.mean():.{prec}f} ({a.std(ddof=1):.{prec}f})' if len(a)>1 else f'{a[0]:.{prec}f}'
        rows.append([LABELS[n],len(rr),plain('sw2'),plain('kl_pq_best'),plain('ksd2_h0.75',4)])
    result_table(rows,['Procedure','n','SW2','KL(p || q)','KSD squared'],[186,25,92,100,105])
    def group_mean(name,key):return np.mean([r[key] for r in by_name[name]])
    para(f"The canonical gap occurs in all three paired seeds: mean SW2 is {group_mean('canonical_cond','sw2'):.3f} versus "
         f"{group_mean('canonical_global','sw2'):.3f}, and forward KL is {group_mean('canonical_cond','kl_pq_best'):.3f} versus "
         f"{group_mean('canonical_global','kl_pq_best'):.3f}. Raising the conditional variance floor to 0.2 improves every canonical conditional seed, "
         f"with mean SW2 {group_mean('floor02_cond','sw2'):.3f}. The adaptive-bandwidth Stein conditional variant also improves every canonical conditional seed "
         f"and has mean SW2 {group_mean('stein_cond','sw2'):.3f}; detaching bandwidth alone leaves mean SW2 {group_mean('detach_h_cond','sw2'):.3f}.")
    para('Theory 1: population representation invariance','Heading1')
    equation(r'f=s_p(X)-s_c(X,\epsilon)=s_p(X)+U/\sigma(\epsilon)')
    equation(r'\mathbb{E}[s_c(X,\epsilon)\mid X]=s_q(X)')
    para('For two independent hierarchical draws, conditional expectation converts E[k(X,X\') f(X,epsilon)<super>T</super> f(X\',epsilon\')] '
         'into the marginal KSD squared. The same marginal q therefore has the same population loss under different Gaussian-mixture representations. '
         'Constant variance is available inside the conditional network by zeroing variance-output weights, so its globally optimized fixed-kernel loss cannot be worse. '
         'This says nothing about finite Adam optimization or a data-adaptive kernel.')
    para('Theory 2: the exact global-variance boundary','Heading1')
    para('The target is an equal mixture of centered Gaussians with diagonal entries 2 and off-diagonal entries +1.8 or -1.8. Its component eigenvalues are 3.8 and 0.2. '
         'Let S be an equiprobable sign and T and U be independent standard normal draws:')
    equation(r'X=\sqrt{1.8}\,T(1,S)^T+\sqrt{0.2}\,U')
    para('This has exactly the target distribution: mixing means supply the long arms and global noise supplies transverse width. '
         'A finite continuous neural mean approximates the branch switch; the construction is not a claim of exact representation by a finite width-128 network.')
    equation(r'D=\mathrm{diag}(d_1,d_2):\quad (2-d_1)(2-d_2)\geq3.24,\quad d_1,d_2\leq2')
    para('<b>Proof of the boundary.</b> These inequalities are equivalent to both residual covariance matrices being positive semidefinite, which suffices by Gaussian convolution. '
         'If a residual covariance is negative in a direction, dividing the target characteristic function by the noise characteristic function gives a positive exponential term growing without bound. '
         'A characteristic function has absolute value at most one, so exact deconvolution is impossible. For isotropic D=vI the exact boundary is 0 &lt;= v &lt;= 0.2. '
         'For off-diagonals +/-2 rho it becomes v &lt;= 2(1-|rho|).')
    para('Theory 3: latent representation noise at an exact optimum','Heading1')
    equation(r'q=p:\quad\mathbb{E}\|s_p-s_c\|^2=\mathbb{E}\,\mathrm{tr}(D^{-1})-\mathbb{E}_p\|s_p\|^2')
    para('<b>Proof.</b> Expand the squared residual and use E[s_c|X]=s_p(X). A Gaussian conditional score has second moment tr(D<super>-1</super>). '
         'Thus the residual can be noisy even when the marginal is exactly correct. Jensen gives E[1/v] &gt;= 1/E[v]. A moderate average variance can conceal substantial noise from a small-variance tail. '
         'This identity concerns score noise; it does not alone prove an increase in every gradient coordinate.')
    para('Theory 4: an analytic gradient-noise explosion','Heading1')
    para('For any 0 &lt; v &lt;= 0.2, use mixing means from the residual covariance mixture and add independent N(0,vI) noise. '
         'The marginal remains exactly p for every v. Differentiate a global dilation X(theta)=exp(theta)X, sigma(theta)=exp(theta)sigma at theta=0, using an independently fixed Gaussian bandwidth h:')
    equation(r'\lim_{v\to0}v^2\mathrm{Var}(g_v)=\frac{2}{N^2}\mathbb{E}_{p\otimes p}\left[k_h^2\left(2+\frac{\|X-X^{\prime}\|^2}{h^2}\right)^2\right]>0')
    para('<b>Proof.</b> The leading pair-gradient term is -k(X,X\')[2+||X-X\'||<super>2</super>/h<super>2</super>] U<super>T</super>U\'/v. '
         'Couple the residual Gaussian means so they converge to independent target draws as v approaches zero. Polynomial growth of the target score derivatives and Gaussian moments give L2 convergence of v times the gradient. '
         'Different pair summands have zero covariance even if they share one index, because the other Gaussian noise is centered. E[(U<super>T</super>U\')<super>2</super>]=2, giving the positive limit. '
         'The actual variance floor is positive; this is a mechanism for large finite variance constants, not a claim of infinite runtime variance.')
    para('The Stein integration-by-parts gradient is a deterministic function of the independent marginal samples X and X\'. Its full dilation-gradient distribution is therefore exactly representation invariant in this construction. '
         'Dilating only mean outputs still gives inverse-square conditional-gradient variance growth, with the squared bracket replaced by ||X-X\'||<super>4</super>/h<super>4</super>; '
         'the Stein mean-direction second moment stays bounded. Scaling final-layer mean rows is an available network parameter direction, so this mechanism can contaminate mean updates too.')
    para('The variance constant is computed from Gaussian quadratic-form Laplace transforms. X-X\' has covariance eigenvalues (7.6,0.4) or (4,4), equally weighted. '
         'At h=0.75 and N=128, the predicted asymptotic variance is 0.000099824/v<super>2</super>. No parameter is fitted to the gradient experiment.')
    rr=probe['exact_marginal']
    result_table([[('0.02 / 0.20 mixture' if r['heterogeneous'] else str(r['v'])),f"{r['conditional_grad_var']:.5f}",f"{r['stein_grad_var']:.5f}",
                   ('-' if r['heterogeneous'] else f"{r['gradient_asymptotic_prediction']:.5f}")] for r in rr],
                 ['Component variance','Measured conditional variance','Measured Stein variance','Analytic asymptotic'],[125,135,125,125])
    para('Each row uses 256 independent repetitions, two batches of 128, and exactly the same marginal p. At v=0.002 the prediction is 24.96 and measurement 24.46, while the Stein variance is about 0.002. '
         'Randomizing variance between 0.02 and 0.20 keeps both marginal p and mean variance 0.11, but raises the measured conditional gradient variance about sevenfold relative to constant 0.11. '
         'This is a distributional counterfactual; architecture-specific causal evidence comes from training controls.')
    para('What the training controls do and do not establish','Heading1')
    para('The Stein training variant removes explicit U/sigma from the integrand. The bandwidth-only control also detaches the median but retains spatial kernel gradients. '
         'Their comparison separates bandwidth detachment from the estimator change. A second three-seed contrast fixes h=0.75 in both estimators, so it also removes adaptive-bandwidth bias. '
         'The population-equivalence claim is restricted to fixed independent kernels: jointly data-dependent bandwidths introduce additional integration-by-parts terms.')
    para('The variance-floor intervention bounds conditional variance below by 0.2; the exact target construction remains available in the flexible-mean closure. '
         'It also changes gradient geometry and introduces clamped zero gradients, so it is not a pure noise intervention. '
         'Removing annealing and sustaining learning rate does not guarantee rescue: outcomes differ across seeds, and at least one conditional run under that policy beats its global counterpart.')
    para('The independently fixed-bandwidth training contrast sets an important limit: Stein improves SW2 for seeds 42 and 43 but worsens it for seed 44. '
         'Its mean forward KL is nearly unchanged and mean held-out KSD is worse. Thus eliminating explicit inverse-variance terms does not ensure better optimization; '
         'the large adaptive-bandwidth improvement cannot be transferred to every kernel policy. Conversely, a successful Stein conditional run still has a large inverse-variance tail. '
         'Small variance can represent the target well; the stochastic estimator determines whether that representation is difficult to optimize.')
    para('Broad screening varied batch size, floors, shared versus separate networks, annealing duration, decay, estimator, bandwidth differentiation, and correlation. '
         'These one-seed runs lasted 10,000 updates; annealed runs still target a tempered density at that horizon. The isotropic Gaussian control shows no comparable conditional disadvantage. '
         'Shared-trunk removal or larger batches alone do not establish a full explanation. An exact empirical critical correlation is not located by these tests.')
    para('Why the convergence theorem does not settle the observation','Heading1')
    para('Annealing reaches the final target after learning rate has fallen to 0.0000718. Only about 6.7% of total nominal step-size mass is spent on that final target. '
         'This is not an Adam-displacement bound, but explains limited late adaptation. The target also has unbounded Hessian:')
    equation(r'\log p(x,y)=\mathrm{const}-\frac{a}{2}(x^2+y^2)+\log\cosh(bxy),\quad a=2/0.76,\ b=1.8/0.76')
    equation(r'\partial_{xx}\log p(0,R)=-a+b^2R^2\to\infty')
    para('The global bounded-Hessian assumption of the published optimization theorem is therefore not literally met by this toy target. This applies to both families and is not itself a cause of their ordering. '
         'The theorem also concerns stationarity under stated assumptions, not global minimization or small KL. The observation does not contradict a guarantee of family expressiveness or successful global optimization.')
    para('Evidence, reproducibility, and limits','Heading1')
    para('All runs used the specified server after reloading AGENTS.md (port 37874), an RTX 4090, PyTorch 2.9.0+cu126, and the ruivi environment. '
         'Jobs ran in tmux; code and generated reports were synchronized through Git. Raw logs, samples, checkpoints, and state are under /root/ruivi/results/ksivi_variance_investigation_20261006. '
         'Each run records its own code revision. The fast research objective gradient was verified against the production update on CPU and GPU.')
    para('Held-out SW2 uses 10,000 samples and 128 directions. KL uses 4,096 exact target draws and nested explicit Gaussian mixtures; integration increases to 262,144 latent draws when needed. '
         'Finite latent integration remains a source of KL uncertainty. Independent KSD uses 16 pairs of 256 samples and fixed bandwidths 0.5, 0.75, and 1.5. '
         'Tables and JSON retain every seed, evaluation standard errors, latent integration checks, and failed interventions. Covariance alone is insufficient: target cross-fourth moment in rotated coordinates is 0.76, compared with 4 for a covariance-matched isotropic Gaussian.')
    para('The analytical identities, exact representation boundary, and gradient asymptotic are proved. The training comparisons and unchanged-marginal simulation are measured. '
         'The combined account of stochastic noise, geometry, and basin selection is a supported interpretation, not a proof that one scalar diagnostic determines every outcome. '
         'Three seeds do not establish universal optimizer, kernel, or target boundaries.')
    para('References','Heading1')
    for s in ['Cheng et al. (2024), Kernel Semi-Implicit Variational Inference, ICML / PMLR 235. https://proceedings.mlr.press/v235/cheng24l.html',
              'Yu et al. (2026), A Kernel Approach for Semi-implicit Variational Inference. https://arxiv.org/abs/2601.12023. Appendix A.4 records global variance in their experiments.',
              'Korba et al. (2021), Kernel Stein Discrepancy Descent, ICML / PMLR 139. https://proceedings.mlr.press/v139/korba21a.html',
              'Original code: https://github.com/longinYu/KSIVI']:para(s)
    story.append(NextPageTemplate('Figures'));story.append(PageBreak())
    for i,f in enumerate(figures):
        para(f"Figure {i+1}: {f['name'].replace('_',' ')}",'Heading1')
        im=PILImage.open(out/(f['name']+'.png'));w,h=im.size;scale=min(936/w,545/h)
        story.append(Image(str(out/(f['name']+'.png')),width=w*scale,height=h*scale))
        para(f['caption'].replace('--','-'))
        if i!=len(figures)-1:story.append(PageBreak())
    doc.build(story)

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
        diagnostics.append(f"{tex_escape(n)}: {delta:.3f}")
    replacements={'@@RESULTS@@':'\n'.join(table),'@@SEEDS@@':'\n'.join(seedtable),
                  '@@HISTORICAL@@':'\n'.join(oldtable),'@@PROBES@@':'\n'.join(probetable),
                  '@@SCREEN@@':'\n'.join(screentable),'@@DIFFERENCES@@':'; '.join(differences),
                  '@@KL_STABILITY@@':'; '.join(diagnostics),
                  '@@COUNT@@':str(len(summaries)),'@@AUDIT_COUNT@@':str(len(json.loads((out/'existing_audit.json').read_text())['rows'])),
                  '@@COMMIT@@':git_commit(),'@@STATUS@@':'INTERIM' if partial else 'COMPLETED'}
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
