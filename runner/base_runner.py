import torch
from models.target_models import target_distribution
from models.networks import VIModel
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from utils.logging import get_logger, set_file_handler
import ite
import time
import numpy as np
from omegaconf import OmegaConf, DictConfig
from collections import defaultdict
from utils.annealing import annealing
from utils.metrics import compute_sliced_wasserstein

logger = get_logger()


class BaseSIVIRunner():
    '''
    The base Reverse SIVI class that encapsulates the model, training, and evaluation.
    
    Key components:
    - Target model: provides `logp` and plotting utilities.
    - VI model: parameterizes q_phi(z|epsilon)
    - Reverse model [Optional]: parameterizes q_psi(epsilon|z) via normalizing flow
    - Logging and artifact paths: under `results/` and `tb_logs/`.
    
    Args:
        config (DictConfig): Configuration for the experiment.
        name(str): Name of the Runner.
    '''

    def __init__(
        self,
        config: DictConfig,
        name: str = "BaseSIVIRunner",
    ):
        assert name != "BaseSIVIRunner", "Please use a subclass of BaseSIVIRunner."

        self.name: str = name
        self.config: DictConfig = config
        self.config_path: str = config.config_path
        self.device: torch.device = config.device

        # target type
        self.target_type: str = self.config.target_type
        logger.info(f"Target type: {self.target_type}")

        # target config
        default_target_config_path = f'configs/targets/{self.target_type}.yaml'
        if 'target_config_path' not in self.config:
            logger.warning(
                f"'target_config_path' not found in main_config; using default: {default_target_config_path}"
            )
            self.config.target_config_path = default_target_config_path
        target_config_path: str = self.config.target_config_path
        logger.info(f"Using target config path: {target_config_path}")
        _target_config = {'target': OmegaConf.load(target_config_path)}
        self.config = OmegaConf.merge(
            _target_config,
            self.config,
        )  # type: ignore

        # save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join("results", self.name, self.target_type,
                                      timestamp)
        os.makedirs(self.save_path, exist_ok=True)

        # attach file logger under save path
        set_file_handler(self.save_path, filename="run.log")
        logger.info(f"Artifacts will be saved to: {self.save_path}")

        # Determine resume behaviors
        self.resume_config: DictConfig = self.config.get(
            'resume', {'enabled': False})
        self.resume: bool = self.resume_config['enabled']

        # target
        self.target_model = target_distribution[self.target_type](
            device=self.device)

        # baseline sample
        self.baseline_samples = self._load_baseline_samples()

        # kl ite samples
        self.n_ite_samples = self.config['metric']['kl_ite']['num_samples']
        # w2 samples
        self.n_w2_samples = self.config['metric']['w2']['num_samples']
        self.n_w2_projections = self.config['metric']['w2']['num_projections']

        # elbo samples
        self.n_elbo_z_samples = self.config['metric']['elbo']['num_z_samples']
        self.n_elbo_batches = self.config['metric']['elbo']['num_batches']
        self.n_elbo_batch_size = self.config['metric']['elbo']['batch_size']

        # vi model config
        self.vi_model_type: str = self.config.vi_model_type
        logger.info(f"VI model type: {self.vi_model_type}")

        if 'vi_model_config_path' not in self.config:
            default_vi_model_config_path = f'configs/vi_models/{self.vi_model_type}.yaml'
            logger.warning(
                f"'vi_model_config_path' not found in main_config; using default: {default_vi_model_config_path}"
            )
            self.config.vi_model_config_path = default_vi_model_config_path
        vi_model_config_path: str = self.config.vi_model_config_path
        logger.info(f"Using VI model config path: {vi_model_config_path}")
        _vi_model_config = {'vi_model': OmegaConf.load(vi_model_config_path)}
        self.config = OmegaConf.merge(
            _vi_model_config,
            self.config,
        )  # type: ignore

        self.epsilon_dim = self.config.vi_model['epsilon_dim']
        self.z_dim = self.config.vi_model['z_dim']

        self.vi_model = VIModel[self.vi_model_type](
            config=self.config.vi_model)
        self.vi_model.to(self.device)

        # Default no reverse model training, altered in subclasses
        self.reverse_train = False

        # TensorBoard writer
        self.tb_path = self.save_path.replace("results", "tb_logs")
        os.makedirs(self.tb_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tb_path)

        # --------- Training/Experiment configuration ---------
        self.training_cfg: DictConfig = self.config['train']
        # epochs and batch sizes
        self.training_num_epochs = self.training_cfg['epochs']
        self.training_batch_size = self.training_cfg['batch_size']

        # Annealing config
        self.use_annealing: bool = self.training_cfg['annealing']['enabled']
        self.anneal_steps: int = self.training_cfg['annealing']['steps']
        self.anneal_scheme: str = self.training_cfg['annealing']['scheme']

        # VI optimizer/scheduler config
        self.vi_opt_cfg = self.training_cfg['vi']
        self.vi_lr = self.vi_opt_cfg['lr']
        self.vi_scheduler_cfg = self.vi_opt_cfg['scheduler']
        assert self.vi_scheduler_cfg['type'] == 'StepLR', \
            "Only StepLR scheduler is supported for VI optimizer."

        # Create VI optimizer and scheduler
        self.optimizer_vi = torch.optim.Adam(
            self.vi_model.parameters(),
            lr=self.vi_lr,
        )
        self.scheduler_vi = torch.optim.lr_scheduler.StepLR(
            self.optimizer_vi,
            step_size=self.vi_scheduler_cfg['step_size'],
            gamma=self.vi_scheduler_cfg['gamma'],
        )

        # Sampling config
        self.training_sample_cfg = self.training_cfg['sample']
        self.training_sample_freq = self.training_sample_cfg['freq']
        self.training_sample_num = self.training_sample_cfg['num']
        self.training_sample_save_path = os.path.join(
            self.save_path,
            "samples",
        )
        os.makedirs(self.training_sample_save_path, exist_ok=True)

        # Logging config
        self.training_log_cfg = self.training_cfg['log']
        self.training_metric_log_freq = self.training_log_cfg[
            'metric_log_freq']
        self.training_loss_log_freq = self.training_log_cfg['loss_log_freq']

        # running accumulators
        self.training_sample_loss = 0.0
        self.training_steps = 0

        # Timing
        self.train_start_time: float = 0.0
        self.train_time_avg_window: int = self.training_log_cfg.get(
            'time_avg_window',
            100,
        )
        self.time_history = defaultdict(list[tuple[float, int]])

        # Starting epoch
        self.train_start_epoch: int = 1
        self.curr_epoch: int = 1

        # Checkpoint config
        self.ckpt_cfg = self.training_cfg['checkpoint']
        self.ckpt_enabled = self.ckpt_cfg['enabled']
        self.ckpt_freq = self.ckpt_cfg['freq']
        self.ckpt_base_path = os.path.join(self.save_path, "checkpoints")
        if self.ckpt_enabled:
            os.makedirs(self.ckpt_base_path, exist_ok=True)

        # Plotting config
        self.plot_cfg = self.training_cfg['plot']
        self.plot_freq = self.plot_cfg['freq']
        self.plot_num = self.plot_cfg['num']
        self.plot_save_path = os.path.join(self.save_path, "plots")
        os.makedirs(self.plot_save_path, exist_ok=True)

    def log_config(self):
        '''
        Log the full configuration to TensorBoard and save as YAML file.
        '''
        # Log to TensorBoard as text
        config_str = OmegaConf.to_yaml(self.config, resolve=True)
        self.writer.add_text(
            "config/full_config",
            f"```yaml\n{config_str}\n```",
            0,
        )

        # Save to YAML file
        config_save_path = os.path.join(self.save_path, "full_config.yaml")
        with open(config_save_path, 'w') as f:
            f.write(config_str)
        logger.info(f"Saved full configuration to {config_save_path}.")

    def _load_baseline_samples(self) -> np.ndarray:
        """
        Load baseline MCMC samples from a configured path (`self.config.target.baseline_path`) for the current target. If not available, use a default path `baselines/hmc/{target_dist}.pt`.

        Returns:
            samples (np.ndarray): Loaded baseline samples on cpu.
        
        """
        baseline_path = self.config.target.get('baseline_path', None)

        if not baseline_path:
            baseline_path = f'baselines/hmc/{self.target_type}.pt'
            logger.warning(
                f"baseline_path not found, using default: {baseline_path}")
        try:
            samples = torch.load(baseline_path, map_location='cpu')
            if isinstance(samples, dict):
                samples = samples['samples']
            samples = torch.as_tensor(samples, dtype=torch.float32)
            logger.info(
                f"Loaded baseline samples from {baseline_path}, shape: {samples.shape}"
            )
            return samples.cpu().numpy()
        except Exception as e:
            logger.error(
                f"Failed to load baseline samples from {baseline_path}: {e}")
            raise e

    def evaluate_vi_to_baseline_kl(self) -> float:
        """
        Estimate KL divergence KL(q_phi(z) || q_baseline(z)) using `ite.cost.BDKL_KnnK`.
        Returns:
            kl_div (float): Estimated KL divergence value.
        """
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute KL divergence.")

        _, z = self.vi_model.sampling(num=self.n_ite_samples)
        z_np = z.cpu().numpy()

        cost_obj = ite.cost.BDKL_KnnK()
        try:
            kl_div = cost_obj.estimation(z_np, self.baseline_samples)
            return float(kl_div)
        except Exception as e:
            logger.error(f"KL estimation failed: {e}")
            raise e

    def evaluate_vi_to_baseline_w2(self) -> float:
        """
        Estimate Sliced Wasserstein-2 distance W2(q_phi(z), q_baseline(z)).
        Returns:
            w2 (float): Estimated W2 distance.
        """
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute W2 distance.")

        _, z = self.vi_model.sampling(num=self.n_w2_samples)

        try:
            # baseline_samples is numpy on cpu. z is torch on device usually.
            # compute_sliced_wasserstein expects torch tensors.
            # We can run W2 on CPU or GPU. Let's send baseline to device to run on GPU if available for speed.

            baseline_tensor = torch.as_tensor(self.baseline_samples,
                                              device=self.device)
            # Use self.n_w2_projections
            w2 = compute_sliced_wasserstein(
                z,
                baseline_tensor,
                num_projections=self.n_w2_projections,
                device=self.device,
                p=2)
            return float(w2)
        except Exception as e:
            logger.error(f"W2 estimation failed: {e}")
            raise e

    def eval_kl_ite(self, epoch: int):
        '''
        Evaluate KL divergence between VI and baseline using ITE and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        kl_div = self.evaluate_vi_to_baseline_kl()
        self.writer.add_scalar("train/vi_kl_to_baseline", kl_div, epoch)
        logger.debug(f"Epoch {epoch}, VI KL to baseline: {kl_div:.4f}")

    def eval_w2(self, epoch: int):
        '''
        Evaluate W2 distance between VI and baseline and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        '''
        w2_dist = self.evaluate_vi_to_baseline_w2()
        self.writer.add_scalar("train/vi_w2_to_baseline", w2_dist, epoch)
        logger.debug(f"Epoch {epoch}, VI W2 to baseline: {w2_dist:.4f}")

    def evaluate_elbo(self) -> tuple[float, float, float]:
        """
        Estimate ELBO using importance sampling for q_phi(z).
        ELBO = E_{z ~ q_phi} [log p(z) - log q_phi(z)]
        
        To estimate log q_phi(z), we use:
        q_phi(z) = E_{epsilon' ~ p(epsilon)} [q_phi(z|epsilon')]
        approximated by Monte Carlo integration over epsilon'.
        
        We use multiple batches to also estimate the standard error of q_phi(z) estimation.
        
        Returns:
            (elbo_mean, elbo_std_total, elbo_std_q) (float, float, float): Estimated ELBO mean, total std, and std from q(z) estimation.
        """
        # 1. Sample z from q_phi(z)
        # We just need z samples, epsilon is implicitly integrated out in generation process.
        _, z_samples = self.vi_model.sampling(num=self.n_elbo_z_samples)
        # z_samples: [N_z, Dz]

        # 2. Estimate log q_phi(z) for each z sample
        # q_phi(z) \approx (1/K) \sum_{k=1}^K q_phi(z|epsilon'_k)
        # log q_phi(z) \approx logsumexp(log q_phi(z|epsilon'_k)) - log K

        # We perform this for multiple batches of epsilon' to get variance estimate
        # Batches: B batches of size S

        # Accumulate q(z) estimate (sum of probs)
        # We work in log space for stability.
        # Stores log(\sum q(z|e_k)) for each batch
        batch_log_q_z_sums = []

        with torch.no_grad():
            for _ in range(self.n_elbo_batches):
                # Sample epsilon' batch
                # [S, De]
                epsilon_prime = self.vi_model.sample_epsilon(
                    num=self.n_elbo_batch_size)

                # Expand to match shapes explicitly as requested using repeat
                # z: [N_z, 1, Dz] -> [N_z, S, Dz]
                # epsilon': [1, S, De] -> [N_z, S, De]
                z_expanded = z_samples.unsqueeze(1).expand(
                    -1,
                    self.n_elbo_batch_size,
                    -1,
                )
                eps_expanded = epsilon_prime.unsqueeze(0).expand(
                    self.n_elbo_z_samples,
                    -1,
                    -1,
                )

                # [N_z, S]
                log_q_z_given_eps = self.vi_model.logp(
                    z_expanded,
                    eps_expanded,
                )

                # Sum over S (in log domain) for this batch
                batch_log_sum = torch.logsumexp(
                    log_q_z_given_eps,
                    dim=1,
                )  # [N_z]
                batch_log_q_z_sums.append(batch_log_sum)

        # Stack: [N_z, B]
        log_sums_tensor = torch.stack(batch_log_q_z_sums, dim=1)

        # --- Total Estimate (using all B*S samples) ---
        log_total_sum = torch.logsumexp(log_sums_tensor, dim=1)  # [N_z]
        total_samples = self.n_elbo_batches * self.n_elbo_batch_size
        log_q_z_mean = log_total_sum - torch.log(
            torch.tensor(total_samples, device=self.device))

        # --- Variance Estimation ---
        # Estimator_b = (1/S) * exp(batch_log_sum_b)
        # We want variance of the mean estimator.
        # Var(Mean) = Var(Estimator_b) / B

        log_estimators_b = log_sums_tensor - torch.log(
            torch.tensor(self.n_elbo_batch_size, device=self.device))

        # Using Delta method for variance of log q(z): Var(log X) \approx Var(X) / E[X]^2

        estimators_b = torch.exp(log_estimators_b)  # [N_z, B]
        var_estimators = torch.var(estimators_b, dim=1)  # [N_z]
        mean_estimators = torch.exp(log_q_z_mean)  # [N_z]

        # Squared standard error of mean estimator (of q(z))
        sq_se_mean_q = var_estimators / self.n_elbo_batches

        # Squared standard error of log q(z)
        sq_se_log_q = sq_se_mean_q / (mean_estimators**2 + 1e-10)

        # 3. Compute log p(z)
        log_p_z = self.target_model.logp(z_samples)  # [N_z]

        # 4. Compute ELBO per sample
        # elbo_i = log p(z_i) - log q(z_i)
        elbo_per_sample = log_p_z - log_q_z_mean

        # Mean ELBO
        elbo_mean = torch.mean(elbo_per_sample)

        # Total ELBO Std (direct std of calculated ELBO)
        elbo_std_total = torch.std(elbo_per_sample)

        # Std arising from estimating q_phi(z) (Average std of the log q estimator)
        elbo_std_q = torch.sqrt(torch.mean(sq_se_log_q))

        return elbo_mean.item(), elbo_std_total.item(), elbo_std_q.item()

    def eval_elbo(self, epoch: int):
        '''
        Evaluate ELBO metric and log to TensorBoard.
        '''
        elbo_val, elbo_std_total, elbo_std_q = self.evaluate_elbo()
        self.writer.add_scalar("train/vi_elbo", elbo_val, epoch)
        self.writer.add_scalar("train/vi_elbo_std_total", elbo_std_total,
                               epoch)
        self.writer.add_scalar("train/vi_elbo_std_q", elbo_std_q, epoch)

        logger.debug(
            f"Epoch {epoch}, ELBO: {elbo_val:.4f}, Std Total: {elbo_std_total:.4f}, Std Q: {elbo_std_q:.4f}"
        )

    def save_samples(self, epoch: int):
        '''
        Save samples from the VI model at the given epoch.
        Args:
            epoch (int): Current epoch number.
        '''
        current_sample_time = time.perf_counter()
        epsilon_sample, z_sample = self.vi_model.sampling(
            num=self.training_sample_num)

        sample_dict = {
            'z': z_sample,
            'epsilon': epsilon_sample,
            'epoch': epoch,
            'time': current_sample_time - self.train_start_time,
            'exp_name': self.name,
            'target_type': self.target_type,
            'vi_model_type': self.vi_model_type,
        }

        torch.save(
            sample_dict,
            os.path.join(
                self.training_sample_save_path,
                f"samples_epoch_{epoch}.pt",
            ))

        logger.debug(
            f"Saved {self.training_sample_num} samples at epoch {epoch}.")

    def save_checkpoint(self, epoch: int):
        '''
        Save the state dict of model and optimizer to checkpoints at the given epoch.
        Args:
            epoch (int): Current epoch number.
        '''
        epoch_ckpt_dir = os.path.join(self.ckpt_base_path, f"epoch_{epoch}")
        os.makedirs(epoch_ckpt_dir, exist_ok=True)
        # Save VI model
        vi_ckpt_path = os.path.join(epoch_ckpt_dir, "vi_model.pt")
        torch.save(self.vi_model.state_dict(), vi_ckpt_path)
        # Save VI optimizer and scheduler
        vi_opt_path = os.path.join(epoch_ckpt_dir, "vi_optim.pt")
        vi_sched_path = os.path.join(epoch_ckpt_dir, "vi_sched.pt")
        torch.save(self.optimizer_vi.state_dict(), vi_opt_path)
        torch.save(self.scheduler_vi.state_dict(), vi_sched_path)
        logger.debug(
            f"Saved VIModel checkpoints at epoch {epoch} to {epoch_ckpt_dir}.")

    def load_checkpoints(self):
        '''
        Load model state dicts from checkpoint directory when resuming training. Use default initialization if checkpoint files are missing. Will try to load optimizer and scheduler states if available.
        '''
        ckpt_dir = self.config.resume.ckpt_dir
        if not os.path.isdir(ckpt_dir) or not os.listdir(ckpt_dir):
            raise RuntimeError(
                f"Checkpoint directory {ckpt_dir} does not exist or is empty.")
        logger.info(f"Resume requested. Checkpoint dir: {ckpt_dir}")
        try:
            # VI model checkpoint
            vi_ckpt_path = os.path.join(
                ckpt_dir,
                'vi_model.pt',
            )
            if os.path.isfile(vi_ckpt_path):
                state = torch.load(
                    vi_ckpt_path,
                    map_location=self.device,
                )
                self.vi_model.load_state_dict(state)
                logger.info(f"Loaded VI model checkpoint from {vi_ckpt_path}")
            else:
                logger.warning(
                    f"VI checkpoint not found at {vi_ckpt_path}; using default initialization."
                )
        except Exception as e:
            logger.error(f"Failed to load checkpoints from {ckpt_dir}: {e}.")
            raise e

        if not self.config.resume.get('load_optimizer', False):
            return

        logger.debug("Trying to load optimizer and scheduler states...")

        try:
            vi_opt_path = os.path.join(ckpt_dir, 'vi_optim.pt')
            vi_sched_path = os.path.join(
                ckpt_dir,
                'vi_sched.pt',
            )
            if os.path.isfile(vi_opt_path):
                opt_state = torch.load(
                    vi_opt_path,
                    map_location=self.device,
                )
                self.optimizer_vi.load_state_dict(opt_state)
                logger.info(f"Loaded VI optimizer from {vi_opt_path}")
            else:
                logger.warning(
                    f"VI optimizer checkpoint not found at {vi_opt_path}; using fresh optimizer."
                )
            if os.path.isfile(vi_sched_path):
                sched_state = torch.load(
                    vi_sched_path,
                    map_location=self.device,
                )
                self.scheduler_vi.load_state_dict(sched_state)
                logger.info(f"Loaded VI scheduler from {vi_sched_path}")
            else:
                logger.warning(
                    f"VI scheduler checkpoint not found at {vi_sched_path}; using fresh scheduler."
                )
        except Exception as e:
            logger.error(f"Failed to load VI optimizer/scheduler: {e}.")
            raise e

        if self.config.resume.get("no_override_epoch", False):
            return

        # Set starting epoch
        try:
            base = os.path.basename(ckpt_dir.rstrip('/'))
            if base.startswith('epoch_'):
                parsed = int(base.split('_')[1])
                self.train_start_epoch = parsed + 1
                logger.info(
                    f"Starting training from epoch {self.train_start_epoch} due to resume."
                )
            else:
                logger.warning(
                    f"Resume dir '{ckpt_dir}' does not end with 'epoch_<n>'; starting from epoch 1."
                )
        except Exception as e:
            logger.warning(
                f"Failed to parse starting epoch from resume dir '{ckpt_dir}': {e}. Starting from epoch 1."
            )

    def train_reverse_model(self, epoch_outer: int):
        '''
        Train the reverse model for several inner epochs using samples from the current VI.
        Args:
            epoch_outer (int): Current outer epoch number.
        '''
        raise NotImplementedError(
            "train_reverse_model must be implemented in subclasses.")

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Estimate log q_phi(z), especially the gradient.

        Args:
            z (torch.Tensor): Samples from q_phi(z|epsilon), shape (batch_size, z_dim).
            epsilon (torch.Tensor): Corresponding epsilon samples, shape (batch_size, epsilon_dim).
        
        Returns:
            log_q_phi_z (torch.Tensor): Estimated log q_phi(z), shape (batch_size,).
        '''
        raise NotImplementedError(
            "calc_log_q_phi_z must be implemented in subclasses.")

    def learn(self):
        '''
        Run the full training procedure for UIVI.

        Algorithm overview:
        1. Warm up the reverse model [Optional].
        2. For each VI step:
            - Sample z from VI q_phi(z|epsilon).
            - Compute target log-density `log p(z)` from the target model.
            - Estimate `nabla_z log q_phi(z)`
            - loss = - E_q[ log p(z) - nabla_z log q_phi(z) ]
        3. Periodically update the reverse model for several inner epochs using
            samples from the current VI [Optional].
        
        Returns:
            None
        '''

        # If resuming, optionally load optimizer & scheduler states
        if self.resume:
            self.load_checkpoints()

        # Main training loop
        self.vi_model.train()
        # Timing accumulators
        last_time = time.perf_counter()
        self.train_start_time = time.perf_counter()
        time_scalars = {}

        for epoch in tqdm(
                range(self.train_start_epoch, self.training_num_epochs + 1),
                desc="Main Training",
                initial=self.train_start_epoch - 1,
                total=self.training_num_epochs,
        ):
            epoch_start_time = time.perf_counter()
            self.curr_epoch = epoch
            time_scalars.clear()

            # Sample epsilon
            t_vi0 = time.perf_counter()
            epsilon = self.vi_model.sample_epsilon(
                num=self.training_batch_size)

            # Sample z from variational distribution
            z, neg_score_implicit = self.vi_model.forward(epsilon)

            # Compute log prob under target distribution
            # log_prob_target = self.target_model.logp(z)
            log_prob_target: torch.Tensor = self.target_model.score(
                z.clone().detach()) * z
            log_prob_target = log_prob_target.sum(dim=-1)
            # log_prob_target: shape (batch_size,)

            # Apply annealing if enabled
            anneal_factor = annealing(
                t=epoch,
                warm_up_interval=self.anneal_steps,
                anneal=self.use_annealing,
                scheme=self.anneal_scheme,
            )
            log_prob_target = log_prob_target * anneal_factor

            t_vi1 = time.perf_counter()
            time_vi_sample_step = t_vi1 - t_vi0
            time_scalars['vi_sample'] = time_vi_sample_step

            t_ns0 = time.perf_counter()

            # Give an estimate to log q_phi(z), specifically its gradient
            log_q_phi_z = self.calc_log_q_phi_z(z, epsilon)

            t_ns1 = time.perf_counter()
            time_neg_score_step = t_ns1 - t_ns0
            time_scalars['neg_score'] = time_neg_score_step

            t_bw0 = time.perf_counter()

            # Compute loss
            loss = -torch.mean(log_prob_target - log_q_phi_z)
            grad_norm = torch.nn.utils.get_total_norm(
                self.vi_model.parameters(), )

            if torch.isfinite(loss):
                self.optimizer_vi.zero_grad()
                loss.backward()
                self.optimizer_vi.step()
                self.scheduler_vi.step()
            else:
                logger.warning(
                    f"NaN or Inf detected in VI loss at epoch {epoch}. Skipping update."
                )
                logger.debug(
                    f"Detected {~torch.isfinite(log_prob_target).sum()} non-finite values in log_prob_target."
                )
                logger.debug(
                    f"Detected {~torch.isfinite(log_q_phi_z).sum()} non-finite values in log_q_phi_z."
                )

            t_bw1 = time.perf_counter()
            time_backward_step = t_bw1 - t_bw0
            time_scalars['backward'] = time_backward_step

            # TensorBoard scalars
            self.writer.add_scalar("train/loss", loss.item(), epoch)
            self.writer.add_scalar(
                "norm/grad_train_step",
                grad_norm.item(),
                epoch,
            )
            z_norm = torch.norm(z, dim=1)
            self.writer.add_scalar(
                "norm/z_norm_avg",
                z_norm.mean().item(),
                epoch,
            )
            self.writer.add_scalar(
                "norm/z_norm_std",
                z_norm.std().item(),
                epoch,
            )
            epsilon_norm = torch.norm(epsilon, dim=1)
            self.writer.add_scalar(
                "norm/epsilon_norm_avg",
                epsilon_norm.mean().item(),
                epoch,
            )
            self.writer.add_scalar(
                "norm/epsilon_norm_std",
                epsilon_norm.std().item(),
                epoch,
            )

            self.training_sample_loss += loss.item()
            if epoch % self.training_loss_log_freq == 0:
                avg_loss = self.training_sample_loss / self.training_loss_log_freq
                current_time = time.perf_counter()
                avg_epoch_time = (current_time -
                                  last_time) / self.training_loss_log_freq

                logger.debug(
                    f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Avg Epoch Time: {avg_epoch_time:.4f}s"
                )
                # Reset accumulators
                self.training_sample_loss = 0.0
                last_time = current_time

            # Train reverse model
            if self.reverse_train:
                time_rev0 = time.perf_counter()
                self.train_reverse_model(epoch)
                time_rev1 = time.perf_counter()
                time_reverse_step = time_rev1 - time_rev0
                time_scalars['reverse_train'] = time_reverse_step

            # Generate and save samples
            if epoch % self.training_sample_freq == 0:
                time_sample0 = time.perf_counter()
                self.save_samples(epoch)
                time_sample1 = time.perf_counter()
                time_sample_step = time_sample1 - time_sample0
                time_scalars['sampling'] = time_sample_step

            # Save checkpoints
            if self.ckpt_enabled and (epoch % self.ckpt_freq == 0):
                time_ckpt0 = time.perf_counter()
                self.save_checkpoint(epoch)
                time_ckpt1 = time.perf_counter()
                time_ckpt_step = time_ckpt1 - time_ckpt0
                time_scalars['checkpoint'] = time_ckpt_step

            # Log metrics
            if self.training_metric_log_freq > 0 and (
                    epoch % self.training_metric_log_freq == 0):

                t_kl0 = time.perf_counter()
                self.eval_kl_ite(epoch)
                t_kl1 = time.perf_counter()

                time_kl_step = t_kl1 - t_kl0
                time_scalars['kl_estimation'] = time_kl_step

                t_w2_0 = time.perf_counter()
                self.eval_w2(epoch)
                t_w2_1 = time.perf_counter()

                time_w2_step = t_w2_1 - t_w2_0
                time_scalars['w2_estimation'] = time_w2_step

                # Evaluate ELBO
                t_elbo0 = time.perf_counter()
                self.eval_elbo(epoch)
                t_elbo1 = time.perf_counter()

                time_scalars['elbo_estimation'] = t_elbo1 - t_elbo0

            # Generate and save contour plots
            if epoch % self.plot_freq == 0:
                t_plot0 = time.perf_counter()
                _, z_plot = self.vi_model.sampling(num=self.plot_num)

                self.target_model.contour_plot(
                    self.config.target.bbox,
                    fnet=None,
                    samples=z_plot.cpu().numpy(),
                    save_to_path=os.path.join(
                        self.plot_save_path,
                        f"contour_epoch_{epoch}.png",
                    ),
                    quiver=False,
                    t=epoch,
                )
                logger.debug(f"Saved contour plot at epoch {epoch}.")
                t_plot1 = time.perf_counter()
                time_plot_step = t_plot1 - t_plot0
                time_scalars['plot'] = time_plot_step
            epoch_end_time = time.perf_counter()
            epoch_time = epoch_end_time - epoch_start_time
            time_scalars['epoch'] = epoch_time

            # Update time history and log
            for key, value in time_scalars.items():
                self.time_history[key].append((value, epoch))
                self.writer.add_scalar(f"time/{key}", value, epoch)

                while self.time_history[key][0][
                        1] <= epoch - self.train_time_avg_window:
                    self.time_history[key].pop(0)

                avg_val = sum(val for val, _ in self.time_history[key]) / min(
                    self.train_time_avg_window, epoch)
                self.writer.add_scalar(f"time_avg/{key}", avg_val, epoch)

        # Close writer at end
        total_time = time.perf_counter() - self.train_start_time
        avg_epoch_time = total_time / max(1, self.training_num_epochs)
        logger.info(
            f"Training completed. Total time: {total_time:.3f}s, Avg epoch time: {avg_epoch_time:.6f}s"
        )
        self.writer.add_scalar("summary/total_training_time", total_time, 0)
        self.writer.add_scalar("summary/avg_epoch_time", avg_epoch_time, 0)
        self.writer.close()
