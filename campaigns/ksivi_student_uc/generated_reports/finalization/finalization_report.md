# Canonical KSIVI Student-t evaluation

Source manifest: `campaigns/toy_scatter_ksivi_detached_annealing/manifest.json`.
Seeds: 42, 43, 44, 45, 46. Each checkpoint is at 50,000 iterations.
Annealing: linear over 25,000 iterations. Warmup log-density coefficient: 0.05.
Riesz median bandwidth is detached; sample-input gradients are enabled.

Evaluation follows the existing paper workflow: 5,000 VI samples with 20 batches
of 2,048 auxiliary samples for the ELBO; 10,000 accepted samples and 5,000
projections for truncated W2 with coordinate threshold 8.

| Seed | KL-style (-ELBO) | Truncated W2 |
| --- | ---: | ---: |
| 42 | 2.729987 | 0.082846 |
| 43 | 11.657788 | 3.260211 |
| 44 | 2.724492 | 0.103731 |
| 45 | 2.734844 | 0.148944 |
| 46 | 2.710746 | 0.146953 |

Five-seed KL-style mean ± SE: 4.511571 ± 1.786559.
Five-seed truncated W2 mean ± SE: 0.748537 ± 0.628046.

All evaluation metrics are finite, with no errors or constrained-sampling fallbacks.
The paper metric rows for KSIVI on Student-t are replaced by this five-seed set.
The canonical scatter grid uses its seed 42 checkpoint.
