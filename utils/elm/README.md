# Expected Log Marginal (ELM)

This package contains helpers for expected-log-marginal style diagnostics.
The training-time `metric.expected_log_marginal` hook uses the paper-style
coordinate-marginal KDE estimator:

```text
KDE-ELM = E_{z ~ r}[sum_j log q_hat_phi,j(z_j)].
```

The package also keeps legacy post-hoc helpers for estimating the full
semi-implicit marginal density

```text
ELM = E_{z ~ r}[log q_phi(z)],
```

where `r` is an empirical reference distribution, usually stored baseline or
MCMC samples.

## Workflow

1. Load a trained VI checkpoint through the normal runner/config machinery.
2. Draw fixed reference points `z_i` from the baseline sample store.
3. Fit a reverse proposal `q_psi(epsilon | z)` from VI joint samples
   `(epsilon, z)`.
4. Estimate `log q_phi(z_i)` for several IS budgets.
5. Inspect ELM, standard error, ESS, and runtime as the IS budget grows.

The standalone entry point for the legacy reverse-IS workflow is
`scripts/evaluate_expected_log_marginal.py`.

For paper-style coordinate-marginal KDE evaluation, use
`scripts/evaluate_kde_expected_log_marginal.py`. This estimates

```text
KDE-ELM = (1 / N) sum_i sum_j log q_hat_phi,j(z_ij),
```

where each `q_hat_phi,j` is a one-dimensional Gaussian KDE fit from generated
VI samples in coordinate `j`.

## Math To Code Map

The marginal variational density integrates out the auxiliary variable:

```text
q_phi(z) = int q_phi(z | epsilon) p(epsilon) d epsilon
         = E_{epsilon ~ p(epsilon)}[q_phi(z | epsilon)].
```

For fixed reference samples `z_1, ..., z_N`, ELM is approximated by

```text
ELM ~= (1 / N) sum_i log q_phi(z_i).
```

The direct prior-MC estimator in `estimate_log_q_prior` draws
`epsilon_k ~ p(epsilon)` and computes

```text
hat q_K(z_i) = (1 / K) sum_k q_phi(z_i | epsilon_k).
```

This is simple, but it can be inefficient when `q_phi(z | epsilon)` is sharp:
most prior epsilon samples may contribute almost nothing for a particular
reference point.

The reverse-IS estimator in `estimate_log_q_reverse_is` instead samples from a
proposal fitted to the reverse conditional:

```text
epsilon_ik ~ q_psi(epsilon | z_i)
```

and estimates

```text
hat q_K(z_i) =
    (1 / K) sum_k
        q_phi(z_i | epsilon_ik) p(epsilon_ik)
        / q_psi(epsilon_ik | z_i).
```

The code stores this as a log weight:

```text
log_weight =
    log q_phi(z_i | epsilon_ik)
    + log p(epsilon_ik)
    - log q_psi(epsilon_ik | z_i).
```

For each reference point, the effective sample size diagnostic is

```text
ESS_i = (sum_k w_ik)^2 / sum_k w_ik^2.
```

Low ESS means the estimate is dominated by a few large weights. Increasing
`num_is_samples` or improving the proposal can reduce this concentration.

Because the logarithm is applied after the inner Monte Carlo estimate,

```text
log((1 / K) sum_k q_phi(z_i | epsilon_k)),
```

finite-budget estimates are Jensen-biased downward. In sharp conditional
density regimes, ELM can rise as `K` increases because larger budgets are more
likely to capture rare high-density contributions.

## Code Map

- `estimators.py`
  - `sample_reference_samples`: selects fixed baseline samples for the outer
    expectation over `z`.
  - `kde_expected_log_marginal`: estimates paper-style ELM from coordinate-wise
    Gaussian KDE marginals of generated VI samples.
  - `estimate_log_q_prior`: estimates `log q_phi(z_i)` by direct prior Monte
    Carlo over epsilon.
  - `estimate_log_q_reverse_is`: estimates `log q_phi(z_i)` using a fitted
    reverse proposal and importance weights.
  - `summarize_elm`: averages per-reference log marginal estimates into the
    scalar ELM.
- `proposal.py`
  - Resolves proposal aliases/configs such as `gaussian`, `mog`, and `realnvp`.
  - Fits `q_psi(epsilon | z)` from VI joint samples `(epsilon, z)`.
  - Supports direct-fit proposals, such as Gaussian and MoG, and optimizer-fit
    proposals, such as RealNVP.
  - Builds the Gaussian fast-path cache used during reverse-IS evaluation.
- `types.py`
  - Defines the dataclasses returned by proposal fitting and ELM estimation:
    `ReverseProposalFit`, `LogQEstimate`, and `ELMEstimate`.

## Minimal Reverse-IS Sweep

Use generic paths so the command works for any trained run with baseline
samples:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_expected_log_marginal.py `
  --config <FULL_CONFIG> `
  --checkpoint-dir <CHECKPOINT_DIR> `
  --output-dir analysis\elm_reverse_is_sweep `
  --repeat-budget 1:5000 `
  --repeat-budget 1:10000 `
  --repeat-budget 1:20000 `
  --repeat-budget 1:40000 `
  --num-ref-samples 1000 `
  --proposal-type gaussian `
  --proposal-fit-samples 32768 `
  --is-batch-size 1024 `
  --device auto `
  --overwrite
```

## Minimal KDE Sweep

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_kde_expected_log_marginal.py `
  --config <FULL_CONFIG> `
  --checkpoint-dir <CHECKPOINT_DIR> `
  --output-dir analysis\kde_expected_log_marginal `
  --sample-budgets 10000 60000 100000 200000 `
  --num-ref-samples 1000 `
  --kde-device cuda `
  --dim-chunk 25 `
  --ref-chunk 500 `
  --model-chunk 20000 `
  --overwrite
```

The evaluator generates samples once at the largest requested budget and reuses
prefixes for smaller budgets. Model-sample chunks are aggregated exactly in log
space, so chunk sizes control memory rather than the statistical estimate.

Where:

- `<FULL_CONFIG>` is the `full_config.yaml` from a completed run, or another
  config that can rebuild the same runner/model structure.
- `<CHECKPOINT_DIR>` is a checkpoint directory containing `vi_model.pt`.
- `--repeat-budget REPEATS:IS_SAMPLES` adds one budget row. For example,
  `--repeat-budget 5:20000` runs five independent estimates at `K = 20000`.
- `--is-batch-size` controls memory/runtime chunking only. It does not change
  the statistical budget `K`.

## Outputs

The evaluator writes these files under `--output-dir`:

- `raw.jsonl`: one record per repeat and IS budget.
- `summary.csv`: machine-readable summary grouped by IS sample budget.
- `summary.md`: human-readable table with ELM, standard error, median ESS, and
  runtime.
- `proposal_fit.json`: proposal family, fit mode, fit budgets, fit NLL, runtime,
  and resolved proposal config.
- `proposal_state.pt`: optional, written only with `--save-fitted-proposal`.

## Reading A Sweep

- If ELM rises with `num_is_samples`, the smaller budgets are likely missing
  important high-weight contributions.
- If median ESS is close to 1, most reference-point estimates are dominated by a
  single or very small number of weights.
- If runtime is high but ESS remains low, try a better proposal family or a
  larger proposal fit budget before only increasing `num_is_samples`.
- Repeated budgets are useful for separating systematic finite-budget drift from
  run-to-run Monte Carlo variability.
