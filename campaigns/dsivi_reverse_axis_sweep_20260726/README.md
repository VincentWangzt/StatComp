# DSIVI Reverse-Optimization Axis Sweep

This campaign starts from `configs/dsivi_banana.yaml` at seeds 42–46 and
changes only one reverse-optimization axis at a time.

The shared baseline is:

- reverse optimizer steps (`train.reverse.epochs`): 5
- reverse batch size (`train.reverse.batch_size`): 2048

There are eight variants per seed, for 40 total runs:

| Variant | Reverse steps | Reverse batch size |
| --- | ---: | ---: |
| `baseline` | 5 | 2048 |
| `reverse_steps_1` | 1 | 2048 |
| `reverse_steps_2` | 2 | 2048 |
| `reverse_steps_10` | 10 | 2048 |
| `reverse_batch_512` | 5 | 512 |
| `reverse_batch_1024` | 5 | 1024 |
| `reverse_batch_4096` | 5 | 4096 |
| `reverse_batch_8192` | 5 | 8192 |

Launch from the repository root:

```bash
python scripts/run_default_config_grid_sweep.py \
  --campaign-slug dsivi_reverse_axis_sweep_20260726 \
  --results-dir results/dsivi_reverse_axis_sweep_20260726 \
  --tb-dir tb_logs/dsivi_reverse_axis_sweep_20260726 \
  --methods dsivi \
  --targets banana \
  --seeds 42 43 44 45 46 \
  --variant baseline \
  --variant reverse_steps_1 train.reverse.epochs=1 \
  --variant reverse_steps_2 train.reverse.epochs=2 \
  --variant reverse_steps_10 train.reverse.epochs=10 \
  --variant reverse_batch_512 train.reverse.batch_size=512 \
  --variant reverse_batch_1024 train.reverse.batch_size=1024 \
  --variant reverse_batch_4096 train.reverse.batch_size=4096 \
  --variant reverse_batch_8192 train.reverse.batch_size=8192 \
  --finalize-mode async \
  --finalize-workers 1
```
