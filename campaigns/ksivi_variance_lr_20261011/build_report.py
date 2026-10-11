"""Build the evidence report from completed runs; figures are regenerated on server."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

import numpy as np

from figures import make, label

HERE=Path(__file__).resolve().parent
ROOT=Path('/root/ruivi/results/ksivi_variance_lr_20261011') if os.name!='nt' else HERE.parents[1]/'results/ksivi_variance_lr_20261011'


def read(path):return json.loads(path.read_text())


def tex(s):
    return str(s).replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#')


def number(x,places=3):return f'{float(x):.{places}f}'


def mean(xs):return float(np.mean(xs))


def ms(xs):
    return f'${mean(xs):.3f}\\pm{np.std(xs,ddof=1):.3f}$' if len(xs)>1 else number(xs[0])


def groups(rows,phase):
    result={}
    for r in rows:
        if r['spec']['phase']==phase:result.setdefault(r['spec']['name'],[]).append(r)
    for g in result.values():g.sort(key=lambda r:r['spec']['seed'])
    return result


def values(g,key='sw2'):return [r['final'][key] for r in g]


def short(name):
    translations={'shared':'Shared','separate':'Separate 128x2','linear':'Linear variance',
                  'narrow':'Separate 32x2','shallow':'Separate 128x1','stopgrad':'Shared stop-gradient'}
    text=label(name)
    for old,new in translations.items():
        if text.startswith(old):return new+text[len(old):]
    return text


def table(headers,rows,caption=None,label_name=None,width='4.8cm',long=False):
    columns='@{}p{'+width+'}'+'r'*(len(headers)-1)+'@{}'
    body='\n'.join(' & '.join(map(str,row))+r' \\' for row in rows)
    if long:
        title=(r'\caption{'+caption+'}'+(r'\label{'+label_name+'}' if label_name else '')+r'\\'+'\n') if caption else ''
        header=' & '.join(headers)+r' \\ \midrule'+'\n'
        return r'{\small'+'\n'+r'\begin{longtable}{'+columns+'}\n'+title+r'\toprule'+'\n'+header+r'\endfirsthead'+'\n'+r'\toprule'+'\n'+header+r'\endhead'+'\n'+body+'\n'+r'\bottomrule\end{longtable}}'+'\n'
    return r'\begin{tabular}{'+columns+'}\n'+r'\toprule'+'\n'+' & '.join(headers)+r' \\ \midrule'+'\n'+body+'\n'+r'\bottomrule\end{tabular}'


def figure(name,caption,wide=False):
    return (r'\begin{figure}[htbp]\centering'+'\n'+r'\includegraphics[width=\linewidth]{figures/'+name+'.pdf}\n'+
            r'\caption{'+caption+r'}\label{fig:'+name+'}\n'+r'\end{figure}'+'\n')


def paired(a,b):
    aa={r['spec']['seed']:r['final']['sw2'] for r in a};bb={r['spec']['seed']:r['final']['sw2'] for r in b}
    keys=sorted(set(aa)&set(bb));d=np.array([aa[s]-bb[s] for s in keys])
    return dict(n=len(keys),wins=int((d<0).sum()),difference_mean=float(d.mean()),difference_sd=float(d.std(ddof=1)),
                differences=d.tolist(),seeds=keys)


def compile_pdf(output,runtime,engine,compiler=None):
    runtime.mkdir(parents=True,exist_ok=True)
    executable=compiler or shutil.which(engine)
    if not executable:raise RuntimeError('Compiler unavailable: '+engine)
    env=os.environ.copy()
    if engine=='tectonic':
        env['TECTONIC_CACHE_DIR']=str(Path('/root/ruivi/results/ksivi_variance_investigation_20261006/runtime/typesetting/cache'))
        command=[str(executable),'--keep-logs','--keep-intermediates','--outdir',str(runtime.resolve()),'report.tex']
    else:command=[str(executable),'-pdf','-interaction=nonstopmode','-halt-on-error','-file-line-error','-outdir='+str(runtime.resolve()),'report.tex']
    with (runtime/'compiler.console.log').open('w',encoding='utf-8') as stream:
        subprocess.run(command,cwd=output,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    shutil.copy2(runtime/'report.pdf',output/'report.pdf')
    version=subprocess.check_output([str(executable),'--version'],env=env,text=True).strip()
    digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    (output/'typesetting.json').write_text(json.dumps(dict(engine=engine,version=version,
        source_sha256=digest(output/'report.tex'),pdf_sha256=digest(output/'report.pdf'),
        pdf_bytes=(output/'report.pdf').stat().st_size),indent=2)+'\n')


def build(root,output,draft=False):
    data=read(output/'training_results.json');rows=data['rows']
    selection=read(output/'selection.json');theory=read(output/'theory_checks.json')
    evaluation=read(output/'evaluation.json')['rows'] if (output/'evaluation.json').exists() else []
    audit=read(output/'audit.json') if (output/'audit.json').exists() else {}
    counts=Counter(r['spec']['phase'] for r in rows)
    if not draft:
        assert dict(counts)==dict(sweep=66,architecture=42,confirmation=40,ablation=42,extension=36),counts
        assert len(evaluation)==262,len(evaluation)
        assert audit.get('passed'),audit.get('coverage')
        assert len({(r['path'],r['step']) for r in evaluation})==262
    make(rows,selection,evaluation,theory,root,output/'figures')
    sweep,arch,confirm,ablation,extensions=[groups(rows,p) for p in ['sweep','architecture','confirmation','ablation','extension']]
    selected=[s['name'] for s in selection['selected']]
    slow,sep,alternative=selected
    main_names=['shared_r1',slow,sep,alternative,'global','stein','floor02','separate_r1']
    main_names=list(dict.fromkeys(n for n in main_names if n in confirm))
    summary=table(['50K configuration','Discovery (3)','Held out (5)'],[
        [tex(short(n)),ms(values((sweep|arch)[n])),ms(values(confirm[n]))] for n in main_names],width='7cm')
    pair={n:paired(confirm[slow],confirm[n]) for n in main_names if n!=slow}
    if confirm:
        reduction=100*(1-mean(values(confirm[slow]))/mean(values(confirm['shared_r1'])))
        abstract=(rf'We test variance-specific Adam learning rates and variance architectures in the original StatComp KSIVI implementation on the X-shaped target. '
          rf'The campaign contains 190 primary runs at 50K updates and 36 checkpoint continuations to 200K. '
          rf'A shared variance-head LR of $10^{{-5}}$, with mean/backbone LR $10^{{-3}}$, reduces held-out sliced-Wasserstein error from '
          rf'{ms(values(confirm["shared_r1"]))} to {ms(values(confirm[slow]))}, improving all five paired seeds. '
          rf'Its mean error is comparable to global variance ({mean(values(confirm["global"])):.3f}); five seeds do not establish superiority. '
          rf'An independent linear variance map also performs well, while a large separate network is less reliable. '
          r'Exact gradient-path identities, a finite-budget Adam reachability bound, and unchanged-marginal gradient-noise results for constant and Gaussian kernels explain why slower variance updates can help, why excessive slowing can fail, and why head LR does not control the entire shared variance function. '
          r'Estimator, initialization, annealing, and continuation ablations delimit the result; there is no universal recommendation to decrease the variance LR.')
        findings=(r'\begin{enumerate}'+'\n'+
          rf'\item \textbf{{The proposed slower head helps this conditional setting.}} Reducing only the shared variance-head LR by $100\times$ improves all five held-out seeds and reduces their mean error by {reduction:.1f}\%. The backbone retains both mean and variance gradient paths.'+'\n'+
          rf'\item \textbf{{The useful rate depends on the variance architecture.}} Shared ratio 0.01 and separate ratio 0.03 were selected on discovery seeds. A separate linear variance model selects ratio 1; ratios 0.001--0.1 under-adapt in that model. Separation alone does not establish an advantage over global variance.'+'\n'+
          rf'\item \textbf{{Comparable marginal accuracy does not imply the same mechanism.}} On held-out seeds the slow original estimator is competitive with global variance and Stein loss. A smaller head LR worsens the tested Stein treatment, so estimator changes and step-size changes are not interchangeable.'+'\n'+
          r'\item \textbf{Two distinct theoretical effects are necessary.} Smaller block steps reduce update noise at a fixed state, but must still permit the component variance to adapt from its initial value 1.3133. Shared features provide an additional full-rate route, and cumulative LR constrains what a slow independent network can reach.'+'\n'+r'\end{enumerate}')
    else:abstract='DRAFT: final held-out results and audits are pending.';findings=abstract

    results=r'\subsection{Learning-rate sweep: improvement has a finite useful range}'+'\n'
    results+=(rf'On discovery seeds, the shared equal-LR mean error is {mean(values(sweep["shared_r1"])):.3f}, versus {mean(values(sweep[slow])):.3f} at ratio 0.01. '
       rf'Shared ratios 0.003, 0.01, and 0.03 yield means {mean(values(sweep["shared_r0p003"])):.3f}, {mean(values(sweep[slow])):.3f}, and {mean(values(sweep["shared_r0p03"])):.3f}. '
       r'This is a useful region rather than evidence that 0.01 is an exact optimum. Ratio 0.001 is more variable, and ratio zero fails. '
       rf'The independent width-128 two-layer variance network instead has mean errors {mean(values(sweep["separate_r0p001"])):.3f}, {mean(values(sweep["separate_r0p003"])):.3f}, and {mean(values(sweep[sep])):.3f} at ratios 0.001, 0.003, and 0.03. '
       r'High ratios can also fail; the separate ratio-3 mean is '+number(mean(values(sweep['separate_r3'])))+'. Thus neither a monotone LR rule nor separation alone describes the observations.\n\n')
    results+=figure('learning_rate_sweep',r'Three paired discovery seeds at 50K updates. Lines are arithmetic means; dots are runs. Error bars are sample SD, with lower ends clipped for logarithmic display only. The right panel measures component-score second moments through $\E[1/v]$, not marginal error. Zero-LR controls are tabulated in the appendix.')
    results+=r'\FloatBarrier\subsection{Prospective held-out comparison}'+'\n'
    if confirm:
        results+=(rf'Selection was committed at \texttt{{808b97f}} before seeds 45--49 were launched. The shared slow head improves all five paired equal-LR controls. '
                  rf'It beats global variance on {pair["global"]["wins"]}/5 seeds and Stein loss on {pair["stein"]["wins"]}/5. '
                  rf'The separate selected network improves its own equal-LR control on {paired(confirm[sep],confirm["separate_r1"])["wins"]}/5 seeds, '
                  rf'but has held-out mean {mean(values(confirm[sep])):.3f}, compared with {mean(values(confirm[slow])):.3f} for the slow shared head. '
                  rf'The selected linear variance map has mean {mean(values(confirm[alternative])):.3f}, with all five errors between {min(values(confirm[alternative])):.3f} and {max(values(confirm[alternative])):.3f}. '
                  r'These observations favor a slow shared output layer or restricted variance capacity as practical candidates. They do not show either candidate is statistically superior to global variance.\n\n')
        results+=figure('heldout_comparison','Five held-out training seeds, all at 50K optimizer updates. Each dot is one trained model. Horizontal bars are means. Hyperparameter selection uses none of these seeds.')
        results+=table(['Slow shared minus comparator','Mean difference','SD of differences','Wins'],[
            [tex(short(n)),number(v['difference_mean']),number(v['difference_sd']),f"{v['wins']}/{v['n']}"] for n,v in pair.items()],
            'Paired held-out differences in training-diagnostic SW$_2$; negative favors the slow shared head.',width='7cm',long=True)
    results+=r'\FloatBarrier\subsection{Changing the variance backbone or its gradient route}'+'\n'
    results+=(rf'The selected alternative is an independent affine map $a(Z)=WZ+c$ with only six variance parameters, versus two for global variance. '
              rf'It is a genuine conditional family, but the learned conditional variation is measured separately below. Its good result therefore need not imply that a rich conditional variance network was necessary. '
              rf'At ratio 0.01 the stop-gradient shared variant has mean error {mean(values(arch["stopgrad_r0p01"])):.3f}, compared with {mean(values(sweep[slow])):.3f} with the full shared gradient. '
              rf'At ratio 0.1 it recovers to {mean(values(arch["stopgrad_r0p1"])):.3f}. '
              r'Removing the variance-to-backbone gradient is therefore not a harmless way to implement a slower variance function: here it removes useful adaptation under the finite budget. The comparison is a causal routing intervention with identical forward architecture, but it also changes the update rule as described in Section 3.\n\n')
    results+=figure('architecture_ablation','Variance architecture ablations, discovery seeds at 50K. The mean network remains width 128 with two hidden layers. Width 32, one hidden layer, and no hidden layer change only the variance branch relative to the separate network; stop-gradient changes only routing relative to shared features. Gray curves show the separate width-128 two-layer reference.')
    results+=table(['Variance architecture','Total parameters','Variance parameters'],[
       [tex(short(n)),str((sweep|arch)[n][0]['source']['parameters']),str((sweep|arch)[n][0]['source']['parameters']-17154)]
       for n in ['global','shared_r1','separate_r1','narrow_r1','shallow_r1','linear_r1']],width='8cm',long=True,
       caption='Parameter counts. The common mean-only network has 17,154 parameters. For shared features, the last column counts additional variance output parameters; the backbone still receives both paths.')
    results+=r'\FloatBarrier\subsection{Atomic initialization, annealing, kernel, and estimator controls}'+'\n'
    if ablation:
        results+=(rf'\textbf{{Initialization and freezing.}} Frozen variance 1.3133 gives mean error {mean(values(sweep["shared_r0"])):.3f}; '
           rf'frozen variance 0.2 gives {mean(values(ablation["shared_r0_init02"])):.3f}. '
           rf'Initializing the selected separate network at 0.2 changes its mean from {mean(values(sweep[sep])):.3f} to {mean(values(ablation[sep+"_init02"])):.3f}. '
           r'This supports the need to adapt the initially broad component noise. The effect is not uniform: initialization at 0.2 does not improve every selected shared or global run. A floor and a frozen initial scale are different interventions.\n\n')
        results+=(rf'\textbf{{Annealing.}} Removing score annealing changes the shared equal-LR mean from {mean(values(sweep["shared_r1"])):.3f} to {mean(values(ablation["shared_r1_noanneal"])):.3f}, '
           rf'but changes global variance from {mean(values(sweep["global"])):.3f} to {mean(values(ablation["global_noanneal"])):.3f}, '
           rf'the selected separate network from {mean(values(sweep[sep])):.3f} to {mean(values(ablation[sep+"_noanneal"])):.3f}, '
           rf'and Stein from {mean(values(sweep["stein"])):.3f} to {mean(values(ablation["stein_noanneal"])):.3f}. '
           r'Annealing interacts with the optimizer state and model; disabling it is not a general cure.\n\n')
        results+=(rf'\textbf{{Estimator and bandwidth.}} The slow-head original-loss mean is {mean(values(sweep[slow])):.3f}. '
           rf'Its bandwidth-detached control has mean {mean(values(ablation[slow+"_detach"])):.3f}, while Stein with that same slow head has mean {mean(values(ablation[slow+"_stein"])):.3f}. '
           r'The latter two share the bandwidth-detachment policy, providing the cleaner estimator contrast. The tested slow head does not improve Stein. '
           rf'Adding a 0.2 floor to the slow original head gives mean {mean(values(ablation[slow+"_floor02"])):.3f}, versus {mean(values(sweep["floor02"])):.3f} for a floor at equal LR. '
           r'This combination was evaluated only on discovery seeds; it is not a held-out recommendation.\n\n')
        results+=figure('atomic_ablations','Paired atomic changes at 50K updates. Each short segment connects the same seed in the parent (left) and changed (right) configuration. A decrease is an improvement. The appendix includes all 14 contrasts, including the floor and frozen-variance controls.')
    results+=r'\FloatBarrier\subsection{50K, 100K, and 200K: continued decay versus restart}'+'\n'
    if extensions:
        results+=('All continuations start from the exact 50K model, Adam moments, and training RNG state. Each branch adds 150K updates. '
          'The table and Figure~\\ref{fig:extended_budgets} compare retained decay with a constant $10^{-4}$ mean LR restart; variance LR ratios remain fixed. '
          'Continuation evidence uses discovery seeds and should not be read as a new held-out ranking.\n\n')
        erows=[]
        for name,g in extensions.items():
            extract=lambda step:[next(t['sw2'] for t in r['trajectory'] if t['step']==step) for r in g]
            erows.append([tex(short(g[0]['spec']['parent'])),tex(g[0]['spec']['extension_policy']),ms(extract(50000)),ms(extract(100000)),ms(extract(200000))])
        results+=table(['Parent','Policy','50K','100K','200K'],erows,
            r'Update-matched continuation results; means $\pm$ sample SD over three seeds. Restart changes cumulative LR as well as running longer.',width='4.2cm',long=True)
        results+=figure('extended_budgets','Matched checkpoint continuations. The original decay has almost exhausted its nominal LR budget at 50K. Restart supplies a materially different budget, while keeping the estimator and variance LR ratio unchanged. The linear ratio-0.001 control tests the finite-budget obstruction.')
        finalmean=lambda name:mean(values(extensions[name]))
        results+=(rf'At 200K with restart, mean errors are {finalmean(slow+"_restart1e-4"):.3f} for the selected shared head, '
          rf'{finalmean(sep+"_restart1e-4"):.3f} for the selected separate network, '
          rf'{finalmean("global_restart1e-4"):.3f} for global variance, and {finalmean("stein_restart1e-4"):.3f} for Stein. '
          rf'The shared equal-LR parent remains much worse at {finalmean("shared_r1_restart1e-4"):.3f}. '
          rf'The very slow linear ratio-0.001 model changes from {mean(values(arch["linear_r0p001"])):.3f} at 50K to {finalmean("linear_r0p001_restart1e-4"):.3f}; more optimization does not repair its frozen-scale-like constraint. '
          r'The reachability theorem rules out an exact fit in this control but does not predict the large magnitude of its observed escape.\n\n')

    mechanisms=r'\FloatBarrier\subsection{Marginal fit and inverse-variance moments are distinct}'+'\n'
    mechanisms+=(rf'The shared equal-LR discovery mean $\E[1/v]$ is {mean(values(sweep["shared_r1"],"inverse_v")):.1f}, '
        rf'compared with {mean(values(sweep[slow],"inverse_v")):.2f} for the slow head. '
        rf'Stein training nevertheless reaches {mean(values(sweep["stein"],"inverse_v")):.1f}, with much better marginal error than the equal-LR original estimator. '
        r'Conversely, the frozen wide-variance model has small inverse variance and poor fit. These controls reject an explanation based solely on the size of component variance: the score representation, update dynamics, and necessary variance adaptation all matter.\n\n')
    mechanisms+=figure('representation_noise',r'Unchanged-marginal controls: all representations have the same standard Gaussian marginal. Left: 20,000 replicated gradients and the exact constant-kernel formulas (dashed), two batches of 128. Right: 512 replications with fixed Gaussian bandwidth 0.75. The derivative follows $\log v$ while preserving the marginal; these are mechanism controls, not additional trained X-shaped models.')
    mechanisms+=(r'Increasing batch size can reduce this noise, but cannot give a variance-uniform guarantee at a fixed batch size. In the exact proxy, original variance behaves as $(Nv)^{-2}$ when $v$ is small; keeping it bounded requires $N$ to grow at least on the order of $1/v$. '
       r'The Stein proxy instead has variance of order $v/N^2$ in that limit. This is an exact counterexample to a uniform fixed-batch cure, not a claim that every neural-network gradient has the same batch-size law. No new batch-size sweep is used to infer an X-shaped threshold.\n\n')
    if 'rbf_small_variance_asymptotic' in theory:
        coefficient=theory['rbf_small_variance_asymptotic']['limit_v_squared_original_gradient_variance']
        small=[r for r in theory['rbf_representation_noise'] if r['v']<=.003]
        mechanisms+=(rf'The Gaussian-kernel theorem predicts $v^2\Var(\widehat g_c)\to{coefficient:.6g}$. '
          + 'The observed ratios to that limit are '+', '.join(f"{r['scaled_original_ratio_to_limit']:.4f} at $v={r['v']}$" for r in small)+
          r'. Thus the fixed-RBF simulation checks a proved coefficient, with the same Gaussian marginal throughout. A slower LR acts on the resulting update; it does not remove this estimator-level divergence.\n\n')
    if confirm and evaluation:
        eg=groups(evaluation,'confirmation')
        complete=[n for n in main_names if n in eg]
        mechanisms+=r'\subsection{Independent marginal metrics and density-approximation checks}'+'\n'
        mechanisms+=table(['Held-out configuration','SW$_2$ (20K)','Forward KL','KSD$^2$, $h=.75$'],[
            [tex(short(n)),ms([r['metrics']['sw2_20k'] for r in eg[n]]),ms([r['metrics']['kl_pq'] for r in eg[n]]),
             f"{mean([r['metrics']['ksd2_h0.75'] for r in eg[n]]):.3g}"] for n in complete],
             'Independent evaluation of the same five held-out trained models. SD is across training seeds; KSD values are means of unbiased cross-batch estimates.',width='6.1cm',long=True)
        deltas=[r['metrics']['kl_last_change'] for r in evaluation]
        unresolved=sum(d>.02 for d in deltas)
        mechanisms+=(rf'The independent samples preserve the main comparisons without recycling the selection metric sample. Across all {len(evaluation)} evaluated checkpoints, '
           rf'{unresolved} retain a last KL mixture-refinement change above 0.02; the maximum is {max(deltas):.3g}. '
           r'Those density estimates are flagged in the CSV and must not be treated as exact KL. Independent fixed-bandwidth KSD values and their Monte Carlo standard errors are retained at all three widths, including negative estimates. '
           r'KSD, forward KL, and SW$_2$ measure different aspects of fit and need not order all bad runs identically. For example, if $q_R=(1-\epsilon)p+\epsilon r_R$, then $D_{KL}(p\Vert q_R)\le-\log(1-\epsilon)$ even when $r_R$ moves arbitrarily far away. '
           r'Transport error can then grow without bound. This elementary example explains why a modest forward KL alone cannot rule out escaped mass; it does not assert that a particular checkpoint has this exact mixture form.\n\n')
        if alternative in eg and 'variance_conditional_cv_mean' in eg[alternative][0]['metrics']:
            cv=[r['metrics']['variance_conditional_cv_mean'] for r in eg[alternative]]
            mechanisms+=(rf'The linear variance model has held-out mean conditional coefficient of variation {mean(cv):.4f} (averaged over dimensions and seeds; range {min(cv):.4f}--{max(cv):.4f}). '
              r'These small coefficients mean the learned variance is close to global on typical latent inputs. A good score for this low-capacity map is therefore not evidence that rich conditional variance was needed.\n\n')
    probes=[r for r in evaluation if 'directional_noise' in r]
    if probes:
        mechanisms+=r'\subsection{Gradient paths and functional drift at trained checkpoints}'+'\n'
        pg={n:[r for r in probes if r['spec']['name']==n] for n in ['shared_r1',slow,sep,'stopgrad_r0p01','linear_r1']}
        pg={n:g for n,g in pg.items() if g}
        mechanisms+=table(['Checkpoint','Mean path norm','Variance path norm','Mean cosine','Freeze drift RMS'],[
          [tex(short(n)),f"{mean([r['gradient_paths']['mean_norm'] for r in g]):.3g}",
           f"{mean([r['gradient_paths']['variance_norm'] for r in g]):.3g}",
           number(mean([r['gradient_paths']['cosine'] for r in g])),
           f"{mean([r['freeze_drift']['actual_rms'] for r in g]):.3g}"] for n,g in pg.items()],
          'Discovery checkpoints at 50K. Norms/cosines average 32 stochastic gradients per model and then training seeds. Freeze drift uses a standardized local mean LR 0.001, with the variance block frozen; it is an intervention, not an additional training run.',width='4.4cm',long=True)
        maxerror=max(r['gradient_paths']['decomposition_relative_error'] for r in probes if r.get('gradient_paths'))
        maxdrift=max(r['freeze_drift']['identity_max_error'] for r in probes if r.get('freeze_drift'))
        mechanisms+=(rf'The measured gradient decomposition has maximum reported mean relative residual {maxerror:.3g}; the raw-output drift identity has maximum absolute residual {maxdrift:.3g}. '
           r'A trained shared variance head can keep changing its output when its own LR is zero. The independent variance network has exactly zero drift when its entire block is frozen. Stop-gradient can still drift through mean-driven features. '
           r'These interventions directly establish the remaining route. Gradient-path cosine alone would not prove it, and its sign is not a universal explanation of the training outcome.\n\n')
        mechanisms+=table(['Checkpoint',r'\shortstack{Head drift\\RMS}',r'\shortstack{Feature drift\\RMS}',r'\shortstack{Mean-path\\noise trace}',r'\shortstack{Var-path\\noise trace}'],[
          [tex(short(n)),f"{mean([r['drift']['head_rms'] for r in g]):.3g}",
           f"{mean([r['drift']['backbone_rms'] for r in g]):.3g}",
           f"{mean([r['gradient_paths']['mean_path_noise_trace'] for r in g]):.3g}",
           f"{mean([r['gradient_paths']['variance_path_noise_trace'] for r in g]):.3g}"] for n,g in pg.items()],
          r'Unfrozen standardized local step and stochastic backbone gradients at the same checkpoints. Feature drift denotes the $W\Delta h$ term, including independent variance features for the separate network. Noise traces concern the original loss and only the mean backbone; a zero variance-path trace in separate/routed models is structural, not zero noise in their variance branch.',width='4.2cm',long=True)
        mechanisms+=(r'The slow shared head has feature-driven raw-variance drift larger than head-driven drift in this local intervention, confirming that its variance function is not simply frozen. '
          r'Equal-LR shared checkpoints have much larger measured backbone noise traces than the selected slow-head checkpoints. These are comparisons between different learned states: they support the noise mechanism but do not isolate a same-state causal effect of the LR.\n\n')
        mechanisms+=figure('trained_mechanisms','Trained-checkpoint probes on discovery seeds. Left: gradient variance for a common marginal-dilation direction, using independently fixed bandwidth 0.75 and 128 repeated pairs of batches. Right: raw variance output drift after freezing the trained variance block, with mean LR 0.001; exact zeros are displayed at $10^{-12}$. Parameter-gradient and semantic-dilation diagnostics have different scopes.')

    claims=r'\FloatBarrier'+'\n'
    claim_rows=[
      (r'Slow shared head improves the tested original loss.',r'LR-only paired intervention; prospective five-seed confirmation.',r'This target, transform, mean architecture, batch size, and 50K policy.'),
      (r'Head LR cannot control all shared variance motion.',r'Exact identity \eqref{eq:drift}, gradient decomposition, and trained-head freeze intervention.',r'A route exists; no universal nonzero lower bound at every state.'),
      (r'Very slow variance learning can be insufficient.',r'Frozen-scale characteristic-function obstruction; finite-budget Adam bound; linear tiny-LR and initialization controls.',r'The quantitative reachability theorem is for the linear variance architecture.'),
      (r'Small component variance can increase original gradient noise without worse marginal fit.',r'Exact constant-kernel formula, Gaussian-kernel asymptotic theorem, and unchanged-marginal replications.',r'Specified marginal-preserving direction; not a universal covariance ordering or every parameter-gradient variance.'),
      (r'Separate backbones do not necessarily halve convergence speed.',r'Exact simultaneous-quadratic counterexample; update counts matched in all comparisons.',r'Compute and conditioning can change; no universal wall-time speed guarantee.'),
      (r'Different block LRs do not themselves bias the estimator.',r'Block-preconditioning identity and local smoothness inequality.',r'SGD local statement; no Adam global convergence claim.'),
      (r'Slowing the head and using Stein are different interventions.',r'Original-detached versus Stein at the same slow LR; fixed-width noise probes.',r'Stein default comparison also changes bandwidth differentiation.'),
      (r'Extra decayed steps provide little nominal LR budget.',r'Geometric-series calculation and checkpoint continuation/restart controls.',r'Nominal LR sums are not actual Adam path lengths.')]
    claims+=r'{\small\begin{longtable}{@{}p{4.3cm}p{6.2cm}p{5cm}@{}}\toprule Claim & Evidence & Boundary \\ \midrule\endhead'+'\n'
    claims+='\n'.join(' & '.join(row)+r' \\' for row in claim_rows)+r'\bottomrule\end{longtable}}'+'\n'
    claims+=(r'\subsection{Practical interpretation}'+'\n'+
       r'For the original conditional-variance loss and this training policy, start with a shared variance output LR $10^{-5}$ and mean/backbone LR $10^{-3}$. '
       r'This is an empirically supported candidate, not a proven universal optimum. Keep the backbone gradient intact; a smaller output LR already changes its learned trajectory. '
       r'If using an independent variance branch, tune its rate again rather than transferring the shared ratio unchanged. A very small branch may be sufficient; the linear branch at ratio 1 is competitive here. '
       r'Compare at matched optimizer steps and report whether LR was restarted. Use inverse-variance moments alongside independent marginal metrics; neither diagnostic alone identifies the whole mechanism.\n\n')

    appendix=''
    for phase,title in [('sweep','Full shared/separate LR sweep'),('architecture','Variance architecture and routing sweep'),('confirmation','Prospective held-out training seeds'),('ablation','Every atomic ablation')]:
        gg=groups(rows,phase)
        if not gg:continue
        seeds=sorted({r['spec']['seed'] for g in gg.values() for r in g})
        arows=[]
        for n,g in gg.items():
            byseed={r['spec']['seed']:r for r in g}
            arows.append([tex(short(n))]+[number(byseed[s]['final']['sw2']) if s in byseed else '--' for s in seeds]+[ms(values(g))])
        appendix+=r'\subsection{'+title+'}\n'+table(['Configuration']+[str(s) for s in seeds]+[r'Mean $\pm$ SD'],arows,
           'Final training-diagnostic SW$_2$, 10K samples and 128 fixed projection directions, at 50K updates. All seeds are shown.',width='5cm',long=True)
    if extensions:
        arows=[]
        for name,g in extensions.items():
            arows.append([tex(short(g[0]['spec']['parent']))+'; '+tex(g[0]['spec']['extension_policy'])]+[number(r['final']['sw2']) for r in g]+[ms(values(g))])
        appendix+=r'\subsection{All continuation endpoints}'+'\n'+table(['Parent and policy','42','43','44',r'Mean $\pm$ SD'],arows,'Every 200K endpoint; the initial 50K updates are shared with the corresponding parent.',width='7cm',long=True)
    appendix+=r'\subsection{Component variance diagnostics on discovery runs}'+'\n'
    appendix+=table(['Configuration',r'$\E[1/v]$',r'1\% quantile','Median','Mean'],[
        [tex(short(n)),f'{mean(values(g,"inverse_v")):.3g}',f'{mean(values(g,"v_q01")):.3g}',
         f'{mean(values(g,"v_median")):.3g}',f'{mean(values(g,"v_mean")):.3g}'] for n,g in (sweep|arch).items()],
         'Arithmetic means of per-model diagnostics over three discovery seeds. Quantiles pool latent samples and coordinates within a model; they are not quantiles across training seeds.',width='6.5cm',long=True)
    appendix+=r'\subsection{Finite-budget theoretical numerical checks}'+'\n'
    appendix+=table(['Ratio and policy','Coordinate bound',r'$\ell$ on $B_3$','Projected $W_2$ bound'],[
       [str(r['ratio'])+'; '+tex(r['policy']),f"{r['parameter_displacement_bound']:.5f}",f"{r['minimum_variance_on_latent_region']:.5f}",
        f"{r['projected_wasserstein_lower_bound']:.5f}"] for r in theory['linear_head_reachability']],
        'Numerical maximization of the proved projection bound, $P(B_3)=0.933359$. A zero bound is inconclusive, not a guarantee of good optimization.',width='6.2cm',long=True)

    samples=figure('final_samples','Held-out seeds 45--47, selected before viewing these results. Each panel shows all 10K final samples within the common viewing window, with exact target contours in black. Metrics include samples outside the window. Columns: equal-LR conditional, slow shared, selected separate, linear variance, global variance, Stein loss.') if (output/'figures/final_samples.pdf').exists() else 'Sample grids are pending.'

    updates=sum(150000 if r['spec']['phase']=='extension' else 50000 for r in rows)
    repro=(rf'The final campaign contains {sum(counts.values())} completed training trajectories: {counts["sweep"]} sweep, {counts["architecture"]} architecture, '
       rf'{counts["confirmation"]} held-out, {counts["ablation"]} ablation, and {counts["extension"]} continuation branches. '
       rf'Counting shared parent updates once gives {updates:,} optimizer updates. The 36 continuations each add 150K updates; no trajectory exceeds 200K. '
       r'Experiments run on an RTX 4090 using PyTorch 2.9.0+cu126 in the server ruivi environment. Each source manifest records the commit, relevant SHA-256 hashes, complete model configuration, parameter count, and training specification.\n\n')
    if audit:
        repro+=(rf'\textbf{{Final audit.}} {len(audit["records"])} endpoint/checkpoint sample arrays and diagnostics were regenerated. '
           rf'Loss gradients were compared against an independent kernel expression, with the Stein operator computed by spatial autograd; maximum relative difference {audit["maximum_gradient_relative_error"]:.3g}. '
           rf'The split-head equal-LR and legacy conditional models have bitwise identical parameters at all {len(audit["legacy_shared_full_trajectories"])} checked seed/checkpoint pairs, '
           rf'and all {len(audit["extension_checkpoint_continuity"])} continuation branches preserve their parent model, RNG, and Adam moments at 50K. '
           r'Production-runner update equality and parameter coverage are recorded separately in \texttt{validation.json}.\n\n')
    repro+=(r'\textbf{Saved artifacts.} \texttt{training\_results.json} and the metric/trajectory CSVs contain every run; \texttt{selection.json} records the discovery selection; '
       r'\texttt{evaluation.json} and \texttt{independent\_metrics.csv} contain all independent SW$_2$, KL refinement, and fixed-bandwidth KSD estimates and standard errors; '
       r'\texttt{theory\_checks.json} and \texttt{audit.json} retain numerical checks and endpoint verification. '
       r'Figures are supplied as vector PDF and PNG. Raw model/optimizer/RNG checkpoints and samples remain under \texttt{/root/ruivi/results/ksivi\_variance\_lr\_20261011}.\n\n'+
       r'\textbf{Production configuration.} Use \texttt{ConditionalGaussianHeads} to enable real parameter groups. The legacy \texttt{ConditionalGaussian} retains its original combined optimizer behavior. '
       r'The new shared model can be selected from the existing KSIVI config with these overrides (the first line is the model type, followed by optimizer settings):'+'\n'+
       r'\begin{verbatim}'+'\nvi_model_type=ConditionalGaussianHeads\ntrain.vi.lr=0.001 train.vi.var_lr=0.00001\nvi_model.variance_backbone=shared\n'+r'\end{verbatim}'+'\n'+
       r'For the selected separate network use \texttt{variance\_backbone=separate} and \texttt{var\_lr=0.00003}. For the linear variance branch set \texttt{variance\_num\_layers=0} and \texttt{var\_lr=0.001}. '
       r'Full commands, phase order, evaluation, report generation, and source changes are documented in the campaign README. Existing canonical configs are not overwritten.\n')

    mapping=dict(ABSTRACT=abstract,FINDINGS=findings,SUMMARY=summary,RESULTS=results,MECHANISMS=mechanisms,
                 CLAIMS=claims,APPENDIX_TABLES=appendix,SAMPLE_FIGURES=samples,REPRODUCIBILITY=repro)
    template=(HERE/'report_template.tex').read_text()
    for key,text in mapping.items():template=template.replace('@'+key+'@',text)
    # Narrative blocks use an explicit paragraph sentinel to keep LaTeX readable.
    template=template.replace('\\n\\n','\n\n')
    template=re.sub(r'\\n(?=\n|$)','\n',template)
    # Only textual paragraph sentinels are used; mathematical commands are unaffected.
    assert '@ABSTRACT@' not in template
    (output/'report.tex').write_text(template,encoding='utf-8')
    summary_data=dict(counts=dict(counts),unique_optimizer_updates=updates,selection=selected,
        heldout={n:dict(mean=mean(values(g)),sd=float(np.std(values(g),ddof=1)),seeds=[r['spec']['seed'] for r in g],values=values(g)) for n,g in confirm.items()},
        paired_slow_shared=pair,independent_checkpoint_count=len(evaluation),audit_passed=audit.get('passed',False))
    (output/'evidence_summary.json').write_text(json.dumps(summary_data,indent=2)+'\n')
    flat=[]
    for r in evaluation:
        flat.append(dict(phase=r['spec']['phase'],name=r['spec']['name'],seed=r['spec']['seed'],step=r['step'],
                         kl_refinement_flag=r['metrics']['kl_last_change']>.02,**r['metrics']))
    if flat:
        keys=sorted(set().union(*(r.keys() for r in flat)))
        with (output/'independent_metrics.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,keys);writer.writeheader();writer.writerows(flat)
    print('BUILT',len(rows),'runs',len(evaluation),'independent evaluations',output)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,default=ROOT);p.add_argument('--output',type=Path,default=HERE/'report')
    p.add_argument('--draft',action='store_true');p.add_argument('--compile',choices=['latexmk','tectonic'])
    p.add_argument('--compiler');p.add_argument('--runtime',type=Path)
    a=p.parse_args();build(a.root,a.output,a.draft)
    if a.compile:compile_pdf(a.output,a.runtime or a.root/'runtime/typesetting',a.compile,a.compiler)
