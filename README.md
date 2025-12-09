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