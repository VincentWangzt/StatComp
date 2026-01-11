## Environment Setup

First create a python virtual environment with `python==3.14`, here we are using `conda`.

```bash
conda create -n stat_comp python=3.14
conda activate stat_comp
```

Then install the required packages via `pip`.

```bash
pip install -r requirements.txt
```

This will install the necessary dependencies for the project. The default CUDA version is set to `12.6`.

## Running Experiments
To run an experiment, use the `src.py` script with a configuration file. For example, to run the SIVI banana experiment, execute:

```bash
python src.py --config configs/sivi_banana.yaml
```

The example configurations are located in the `configs/` directory. The results and checkpoints will be saved in the `results/` directory by default. The tensorboard logs can be found under the `tb_logs/` directory.

Additional command-line arguments can be added via `key=value` pairs. For example, to change the number of training iterations and the learning rate, you can run:

```bash
python src.py --config configs/sivi_banana.yaml train.epochs=20000
```

## HMC baselines
To run HMC baselines, use the `mcmc_baselines.py` script.
```bash
python mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000
```

Results will be saved in the `results/` directory by default. 