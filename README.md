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

## Viewing TensorBoard Logs | 查看 TensorBoard 日志
During training, metrics and visualizations are automatically logged to TensorBoard. To view these logs:

在训练过程中，指标和可视化结果会自动记录到 TensorBoard。查看这些日志：

1. Launch TensorBoard by running | 启动 TensorBoard：
   ```bash
   tensorboard --logdir tb_logs/
   ```

2. Open your web browser and navigate to | 在浏览器中打开：
   ```
   http://localhost:6006
   ```

3. You can view various training metrics including | 您可以查看各种训练指标，包括：
   - Training loss | 训练损失
   - ELBO (Evidence Lower Bound) | ELBO（证据下界）
   - KL divergence | KL 散度
   - Wasserstein-2 distance | Wasserstein-2 距离
   - Learning rate schedules | 学习率调度
   - Training time statistics | 训练时间统计

**Note | 注意**: If you're running experiments on a remote server, you may need to set up SSH port forwarding to view TensorBoard locally | 如果在远程服务器上运行实验，需要设置 SSH 端口转发以在本地查看 TensorBoard：
```bash
ssh -L 6006:localhost:6006 user@remote-server
```

Then run `tensorboard --logdir tb_logs/` on the remote server and access it via `http://localhost:6006` on your local machine.

然后在远程服务器上运行 `tensorboard --logdir tb_logs/`，并通过本地机器的 `http://localhost:6006` 访问。

## HMC baselines
To run HMC baselines, use the `mcmc_baselines.py` script.
```bash
python mcmc_baseline.py --target banana --num-samples 10000 --burn-in 5000
```

Results will be saved in the `results/` directory by default. 