import torch
from models.target_models import target_distribution, DEFAULT_BBOX
from models.networks import SIMINet, ConditionalRealNVP, ConditionalGaussian
import os
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset
import yaml
import argparse
from utils.logging import get_logger, set_file_handler
from typing import Any, Callable
import ite
import math
import time
import numpy as np

# 1208 TODO:
# 1. Add KL to baseline as a metric [Test]
# 2. Restucture the learn() to be more modular, and add more things from local variable to self
# 3. Add readme and scripts and documentation of scripts
# 4. Perhaps move resume to a parameter when initializing? or just pass in the args. Anyway, the parsing of args should be moved inside the __main__ section.
# 5. Fix env.yml
# 6. Inspect mcmc_baseline.py


def load_config(path: str) -> dict[str, Any]:
    '''
    load configuration from a YAML file
    
    Args:
        path (str): path to the YAML config file
    Returns:
        dict: configuration dictionary
    '''
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def parse_args():
    parser = argparse.ArgumentParser(description="UIVI / Reverse-UIVI Runner")
    default_cfg = os.path.join(os.path.dirname(__file__), 'configs',
                               'reverse_uivi.yaml')
    parser.add_argument('--config',
                        type=str,
                        default=default_cfg,
                        help='Path to YAML config file')
    parser.add_argument(
        '--resume_ckpt_dir',
        type=str,
        default=None,
        help=
        'Path to a checkpoint directory (e.g., results/<exp>/<target>/<timestamp>/checkpoints/epoch_<epoch>) to resume models from',
    )
    parser.add_argument(
        '--load_optimizer',
        action='store_true',
        default=False,
        help=
        'When resuming, also load optimizer and scheduler states if available. This will also override the starting epoch (and warmup) based on the checkpoint directory.',
    )
    parser.add_argument(
        '--no_override_start_epoch',
        action='store_true',
        default=False,
        help=
        'Disable overriding starting epoch when resuming; do not skip warmup',
    )
    return parser.parse_args()


# Parse CLI and load configuration
args = parse_args()
CONFIG_PATH = args.config
cfg = load_config(CONFIG_PATH)
# Setup logging (console + optional file under save path)
logger = get_logger("uivi_runner")

# Device and seeds
use_cuda = cfg.get('use_cuda_if_available', True) and torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.get('cuda_visible_devices', "0")
    torch.cuda.manual_seed_all(cfg.get('seed', 42))
else:
    device = torch.device("cpu")

torch.manual_seed(cfg.get('seed', 42))

# Log configs and environment at start
logger.info(f"Using config: {CONFIG_PATH}")
logger.info("Starting run with configuration:")
logger.info(yaml.dump(cfg, sort_keys=False))
logger.info(f"Device: {device}")


class ReverseUIVI():
    '''
    The main Reverse UIVI class that encapsulates the model, training, and evaluation.
    
    Key components:
    - Target model: provides `logp` and plotting utilities.
    - VI model: parameterizes q_phi(z|epsilon)
    - Reverse model: parameterizes q_psi(epsilon|z) via normalizing flow
    - Logging and artifact paths: under `results/` and `tb_logs/`.
    
    Args:
        device (torch.device): The device to run the computations on.
        cfg (dict): Configuration dictionary containing all hyperparameters and settings.
    '''

    def __init__(
        self,
        device: torch.device,
        cfg: dict[str, Any],
        resume_ckpt_dir: str | None = None,
    ):
        self.exp_name = cfg['experiment']['name']
        self.target_dist = cfg['experiment']['target_dist']
        self.device = device
        self.cfg = cfg
        self.resume_ckpt_dir = resume_ckpt_dir

        # target
        self.target_model = target_distribution[self.target_dist](
            device=device)

        # baseline sample
        self.baseline_samples = self._load_baseline_samples()

        # kl ite samples
        self.n_ite_samples = cfg['kl']['n_ite_samples']

        # vi model from config
        model_config = cfg['models']
        assert model_config['vi_model_type'] == 'ConditionalGaussian', \
            "Only ConditionalGaussian VI model is supported."
        self.epsilon_dim = model_config['epsilon_dim']
        self.z_dim = model_config['z_dim']

        self.vi_model = ConditionalGaussian(
            epsilon_dim=model_config['epsilon_dim'],
            hidden_dim=model_config['vi_hidden_dim'],
            z_dim=model_config['z_dim'],
            device=self.device,
        ).to(self.device)

        # reverse model only needed for reverse_uivi
        self.reverse_model = None
        if self.exp_name == 'reverse_uivi':
            assert model_config['reverse_model_type'] == 'ConditionalRealNVP', \
                "Only ConditionalRealNVP reverse model is supported in Reverse UIVI."
            self.reverse_model = ConditionalRealNVP(
                z_dim=model_config['z_dim'],
                epsilon_dim=model_config['epsilon_dim'],
                hidden_dim=model_config['reverse_hidden_dim'],
                num_layers=model_config['reverse_num_layers'],
                device=self.device,
            ).to(self.device)

        # Optionally resume models from checkpoint directory
        if self.resume_ckpt_dir is not None:
            logger.info(
                f"Resume requested. Checkpoint dir: {self.resume_ckpt_dir}")
            try:
                # VI model checkpoint
                vi_ckpt_path = os.path.join(
                    self.resume_ckpt_dir,
                    'vi_model.pt',
                )
                if os.path.isfile(vi_ckpt_path):
                    state = torch.load(
                        vi_ckpt_path,
                        map_location=self.device,
                    )
                    self.vi_model.load_state_dict(state)
                    logger.info(
                        f"Loaded VI model checkpoint from {vi_ckpt_path}")
                else:
                    logger.warning(
                        f"VI checkpoint not found at {vi_ckpt_path}; using default initialization."
                    )
                # Reverse model checkpoint (if applicable)
                if self.reverse_model is not None:
                    rev_ckpt_path = os.path.join(
                        self.resume_ckpt_dir,
                        'reverse_model.pt',
                    )
                    if os.path.isfile(rev_ckpt_path):
                        state = torch.load(
                            rev_ckpt_path,
                            map_location=self.device,
                        )
                        self.reverse_model.load_state_dict(state)
                        logger.info(
                            f"Loaded reverse model checkpoint from {rev_ckpt_path}"
                        )
                    else:
                        logger.warning(
                            f"Reverse model checkpoint not found at {rev_ckpt_path}; using default initialization."
                        )
            except Exception as e:
                logger.error(
                    f"Failed to load checkpoints from {self.resume_ckpt_dir}: {e}."
                )
                raise e

        # save path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.save_path = os.path.join("results", self.exp_name,
                                      self.target_dist, timestamp)
        os.makedirs(self.save_path, exist_ok=True)

        # attach file logger under save path
        set_file_handler(self.save_path, filename="run.log")
        logger.info(f"Artifacts will be saved to: {self.save_path}")

        # TensorBoard writer
        self.tb_path = self.save_path.replace("results", "tb_logs")
        os.makedirs(self.tb_path, exist_ok=True)

        self.writer = SummaryWriter(log_dir=self.tb_path)

    def _load_baseline_samples(self) -> np.ndarray:
        """
        Load baseline MCMC samples from a configured path (`self.cfg['baseline']['baseline_path']`) for the current target. If not available, use a default path `baselines/hmc/{target_dist}.pt`.

        Returns:
            samples (np.ndarray): Loaded baseline samples on cpu.
        
        """
        baseline_cfg = self.cfg.get('baseline', {})
        baseline_path = baseline_cfg.get('baseline_path', None)

        if not baseline_path:
            baseline_path = f'baselines/hmc/{self.target_dist}.pt'
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

    def calculate_KL(self) -> float:
        '''
        Calculate the KL divergence between the true joint distribution and the
        joint distribution induced by the reverse model using the ITE package.
        Returns:
            kl_div (float): Estimated KL divergence value.
        '''
        if self.reverse_model is None:
            raise RuntimeError(
                "KL calculation is only available for reverse_uivi where reverse_model exists."
            )

        # Sample from true joint
        true_eps, true_z = self.vi_model.sampling(num=self.n_ite_samples)
        true_joint = torch.cat([true_eps, true_z], dim=1).cpu().numpy()

        # Sample epsilon from reverse model given true z
        with torch.no_grad():
            generated_eps = self.reverse_model.sample(
                true_z, num_samples=1)[1].squeeze(1)
        generated_joint = torch.cat([generated_eps, true_z],
                                    dim=1).cpu().numpy()

        # Estimate KL divergence using ITE
        cost_obj = ite.cost.BDKL_KnnK()
        kl_div = cost_obj.estimation(generated_joint, true_joint)

        return kl_div

    def train_reverse_model(
        self,
        optimizer: torch.optim.Optimizer,
        epochs: int,
        batch_size: int,
        progress_bar: bool = True,
        log_func: Callable[[float, int], None] = None,
    ) -> None:
        '''
        Train the reverse model. Generate samples from the VI model and then optimize the reverse model to maximize the log-reverse-probablity of these samples.
        
        Args:
            optimizer (torch.optim.Optimizer): Optimizer for the reverse model.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            progress_bar (bool, optional): Whether to display a progress bar. Defaults to True.
            log_func (Callable[[float, int], None], optional): Function to log training progress. Defaults to None. The first argument is the loss, the second is the epoch.
        Returns:
            None
        '''
        if self.reverse_model is None:
            return
        self.reverse_model.train()
        iterator = range(epochs)
        if progress_bar:
            iterator = tqdm(iterator, desc="Reverse Model Training")
        for epoch in iterator:
            epsilon_samples, z_samples = self.vi_model.sampling(
                num=batch_size, )
            optimizer.zero_grad()
            u, log_prob = self.reverse_model.forward(epsilon_samples,
                                                     z_samples)
            loss = -torch.mean(log_prob)
            loss.backward()
            optimizer.step()
            if log_func is not None:
                log_func(loss.item(), epoch)

    def warmup(self) -> None:
        '''
        Warm up the reverse model by training it for a specified number of epochs. Logs the KL divergence and reverse model loss during warmup. Configured via the 'warmup' section in the configuration dictionary.
        
        Returns:
            None
        '''
        if self.reverse_model is None:
            return
        wcfg = self.cfg.get('warmup', {'enabled': False})
        if not wcfg['enabled']:
            return
        lr = wcfg['lr']
        batch_size = wcfg['batch_size']
        epochs = wcfg['epochs']
        kl_freq = wcfg['kl_log_freq']
        loss_log_freq = wcfg['loss_log_freq']
        sample_loss = 0

        optimizer = torch.optim.Adam(self.reverse_model.parameters(), lr=lr)
        warmup_start_time = time.perf_counter()
        last_time = time.perf_counter()

        def log_func(loss, epoch):
            self.writer.add_scalar("warmup/reverse_model_loss", loss, epoch)
            nonlocal sample_loss
            sample_loss += loss
            if epoch % kl_freq == 0:
                kl_div = self.calculate_KL()
                self.writer.add_scalar("warmup/kl_div", kl_div, epoch)
                logger.debug(
                    f"Warmup Epoch {epoch}, KL Divergence: {kl_div:.4f}")
            if epoch % loss_log_freq == 0:
                avg_loss = sample_loss / loss_log_freq
                current_time = time.perf_counter()
                nonlocal last_time
                avg_step_time = (current_time - last_time) / loss_log_freq
                last_time = current_time
                sample_loss = 0
                logger.debug(
                    f"Warmup Epoch {epoch}, Average Reverse Model Loss: {avg_loss:.4f}, Avg Step Time: {avg_step_time:.4f}s"
                )

        self.train_reverse_model(
            optimizer,
            epochs,
            batch_size,
            progress_bar=True,
            log_func=log_func,
        )
        warmup_end_time = time.perf_counter()
        warmup_time = warmup_end_time - warmup_start_time
        logger.info(
            f"Warmup completed for {epochs} epochs. Total time: {warmup_time:.3f}s, Avg epoch time: {warmup_time/epochs:.6f}s"
        )

    # ---------- UIVI (HMC-based epsilon|z sampling) helpers ----------
    def _log_q_z_given_epsilon(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q_phi(z | epsilon) for diagonal Gaussian parameterization.
        Supports broadcasting over leading dimensions.

        Args:
            z (torch.Tensor): shape [..., Dz]
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        mu, log_var = self.vi_model.net(epsilon).chunk(2, dim=-1)
        log_var = log_var.clamp(min=self.vi_model.log_var_min)
        var = torch.exp(log_var)
        # Gaussian log-likelihood per sample
        const = -0.5 * z.shape[-1] * math.log(2 * math.pi)
        ll = const - 0.5 * (log_var.sum(dim=-1) +
                            ((z - mu)**2 / var).sum(dim=-1))
        return ll

    def _log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        """
        Compute log q(epsilon) under standard normal prior.
        Supports broadcasting over leading dimensions.
        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        const = -0.5 * epsilon.shape[-1] * math.log(2 * math.pi)
        return const - 0.5 * (epsilon**2).sum(dim=-1)

    def _log_q_phi_eps_given_z(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q_phi(epsilon | z) = log q(epsilon) + log q_phi(z | epsilon) - const, where the const = log q_phi(z) is ignored.
        Supports broadcasting over leading dimensions.
        Args:
            epsilon (torch.Tensor): shape [..., De]
            z (torch.Tensor): shape [..., Dz]
        Returns:
            log_q_phi (torch.Tensor): shape [...]
        """
        return self._log_q_epsilon(epsilon) + self._log_q_z_given_epsilon(
            z, epsilon)

    def _grad_log_q_phi(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the gradient of log q_phi(epsilon | z) w.r.t. epsilon.
        Args:
            epsilon (torch.Tensor): shape [B, De]
            z (torch.Tensor): shape [B, Dz]
        Returns:
            grad (torch.Tensor): shape [B, De]
        """
        epsilon = epsilon.clone().detach().requires_grad_(True)
        logp = self._log_q_phi_eps_given_z(epsilon, z)
        logp_sum = logp.sum()
        grad = torch.autograd.grad(logp_sum,
                                   epsilon,
                                   retain_graph=False,
                                   create_graph=False)[0]
        return grad.detach()

    def sample_epsilon_hmc(
        self,
        z: torch.Tensor,
        eps_init: torch.Tensor,
        num_samples: int,
        burn_in_steps: int,
        step_size: float,
        leapfrog_steps: int,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """
        Sequential HMC per (z, epsilon) starting from eps_init.
        Runs `burn_in_steps` transitions, then collects `num_samples` samples.
        Returns tiled z, sampled epsilon, and average acceptance rate.
        
        Args:
            z (torch.Tensor): shape [B, Dz]
            eps_init (torch.Tensor): shape [B, De]
            num_samples (int): number of samples to collect after burn-in
            burn_in_steps (int): number of burn-in HMC steps
            step_size (float): leapfrog step size
            leapfrog_steps (int): number of leapfrog steps per HMC transition
        Returns:
            (z_aux, eps_aux, acc_rate) (torch.Tensor, torch.Tensor, float):
            z_aux has shape [B, S, Dz] (tiled z),
            eps_aux has shape [B, S, De] (sampled epsilons),
            acc_rate is the average acceptance rate (float).
        """
        B, Dz = z.shape
        De = self.epsilon_dim
        S = num_samples
        device = self.device

        z_aux = z.unsqueeze(1).expand(B, S, Dz).clone().detach()
        eps_current = eps_init.clone().detach().to(device)  # [B, De]

        samples = []
        accepts = []

        total_steps = burn_in_steps + S
        for step in range(total_steps):
            # Resample momentum
            p0 = torch.randn(B, De, device=device)

            # Current energies
            logp0 = self._log_q_phi_eps_given_z(eps_current, z)
            K0 = 0.5 * (p0**2).sum(dim=-1)

            # Leapfrog from current state (batched over B)
            p = p0 + 0.5 * step_size * self._grad_log_q_phi(eps_current, z)
            eps_prop = eps_current
            for t in range(leapfrog_steps):
                eps_prop = eps_prop + step_size * p
                grad = self._grad_log_q_phi(eps_prop, z)
                if t != leapfrog_steps - 1:
                    p = p + step_size * grad
            p = p + 0.5 * step_size * grad

            # Proposed energies
            logp_prop = self._log_q_phi_eps_given_z(eps_prop, z)
            K_prop = 0.5 * (p**2).sum(dim=-1)

            dH = (K_prop - logp_prop) - (K0 - logp0)
            accept_prob = torch.exp((-dH).clamp(max=0))
            u = torch.rand_like(accept_prob)
            accept_mask = (u < accept_prob)

            # Update current
            eps_current = torch.where(accept_mask.unsqueeze(-1), eps_prop,
                                      eps_current)

            accepts.append(accept_mask.float().clone().detach())
            # After burn-in, record sample
            if step >= burn_in_steps:
                samples.append(eps_current.clone().detach())

        # Stack samples: list of S tensors [B, De] -> [S, B, De] -> [B, S, De]
        eps_out = torch.stack(samples, dim=0).transpose(0, 1)
        acc_rate = torch.stack(accepts, dim=0).mean().item()
        return z_aux.detach(), eps_out.detach(), acc_rate

    def learn(self):
        '''
        Run the full training procedure for Reverse UIVI.

        Algorithm overview:
        1. Warm up the reverse flow.
        2. For each VI step:
            - Sample z from VI q_phi(z|epsilon).
            - Compute target log-density `log p(z)` from the target model.
            - Estimate the score function term using samples from the reverse model q_psi(epsilon|z).
            - Loss = -E_z[log p(z) + StopGrad(E_epsilon'~q_psi[score_phi(z|epsilon')])^T * z]
        3. Periodically update the reverse flow for several inner epochs using
            samples from the current VI.
        
        In the case of uivi, the reverse model is None, and we use HMC to sample
        epsilon ~ q(epsilon|z) instead.
        
        Returns:
            None
        '''
        # Determine resume behaviors from CLI flags
        load_opt_sched = args.load_optimizer
        override_start_epoch = not args.no_override_start_epoch
        # initial warmup (only for reverse_uivi) unless resuming with epoch override
        if self.reverse_model is not None:
            if not (self.resume_ckpt_dir is not None and override_start_epoch):
                self.warmup()

        # training configs
        tcfg = self.cfg['train']
        num_epochs = tcfg['epochs']
        batch_size = tcfg['batch_size']
        reverse_sample_num = tcfg['reverse_sample_num']
        vi_opt_cfg = tcfg['vi']
        vi_lr = vi_opt_cfg['lr']
        vi_scheduler_cfg = vi_opt_cfg['scheduler']
        assert vi_scheduler_cfg['type'] == 'StepLR', \
            "Only StepLR scheduler is supported for VI optimizer."

        # VI optimizer
        optimizer_vi = torch.optim.Adam(self.vi_model.parameters(), lr=vi_lr)
        scheduler_vi = torch.optim.lr_scheduler.StepLR(
            optimizer_vi,
            step_size=vi_scheduler_cfg['step_size'],
            gamma=vi_scheduler_cfg['gamma'],
        )
        # If resuming, optionally load optimizer & scheduler states
        if self.resume_ckpt_dir is not None and load_opt_sched:
            try:
                vi_opt_path = os.path.join(self.resume_ckpt_dir, 'vi_optim.pt')
                vi_sched_path = os.path.join(
                    self.resume_ckpt_dir,
                    'vi_sched.pt',
                )
                if os.path.isfile(vi_opt_path):
                    opt_state = torch.load(
                        vi_opt_path,
                        map_location=self.device,
                    )
                    optimizer_vi.load_state_dict(opt_state)
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
                    scheduler_vi.load_state_dict(sched_state)
                    logger.info(f"Loaded VI scheduler from {vi_sched_path}")
                else:
                    logger.warning(
                        f"VI scheduler checkpoint not found at {vi_sched_path}; using fresh scheduler."
                    )
            except Exception as e:
                logger.error(f"Failed to load VI optimizer/scheduler: {e}.")
                raise e

        # Reverse training cfg (if applicable)
        if self.reverse_model is not None:
            rev_train_cfg = tcfg['reverse']
            reverse_lr = rev_train_cfg['lr']
            rev_batch_size = rev_train_cfg['batch_size']
            rev_epochs = rev_train_cfg['epochs']
            rev_update_freq = rev_train_cfg['update_freq']
            rev_reuse_optimizer = rev_train_cfg['reuse_optimizer']

            if rev_reuse_optimizer:
                reverse_optimizer = torch.optim.Adam(
                    self.reverse_model.parameters(), lr=reverse_lr)
            else:
                reverse_optimizer = None
            # If resuming and allowed, load reverse optimizer only when reusing optimizer
            if (self.resume_ckpt_dir is not None and load_opt_sched
                    and rev_reuse_optimizer and reverse_optimizer is not None):
                try:
                    rev_opt_path = os.path.join(
                        self.resume_ckpt_dir,
                        'reverse_optim.pt',
                    )
                    if os.path.isfile(rev_opt_path):
                        ro_state = torch.load(
                            rev_opt_path,
                            map_location=self.device,
                        )
                        reverse_optimizer.load_state_dict(ro_state)
                        logger.info(
                            f"Loaded reverse optimizer from {rev_opt_path}")
                    else:
                        logger.warning(
                            f"Reverse optimizer checkpoint not found at {rev_opt_path}; using fresh optimizer."
                        )
                except Exception as e:
                    logger.error(f"Failed to load reverse optimizer: {e}.")
                    raise e
        else:
            reverse_optimizer = None

        # HMC cfg (for uivi)
        hmc_cfg = self.cfg.get('hmc', {
            'step_size': 0.20,
            'leapfrog_steps': 5,
            'burn_in_steps': 5,
        })
        hmc_step_size = hmc_cfg['step_size']
        hmc_leapfrog_steps = hmc_cfg['leapfrog_steps']
        hmc_burn_in_steps = hmc_cfg['burn_in_steps']

        # Sampling cfg
        sample_cfg = tcfg['sample']
        sample_freq = sample_cfg['freq']
        sample_num = sample_cfg['num']
        sample_save_path = os.path.join(self.save_path, "samples")
        if not os.path.exists(sample_save_path):
            os.makedirs(sample_save_path)

        # Logging cfg
        log_cfg = tcfg['log']
        kl_log_freq = log_cfg['kl_log_freq']
        loss_log_freq = log_cfg['loss_log_freq']
        reverse_log_freq = log_cfg.get('reverse_log_freq', 0)
        sample_loss = 0
        sample_reverse_loss = 0

        # Checkpoint cfg
        ckpt_cfg = tcfg['checkpoint']
        ckpt_enabled = ckpt_cfg['enabled']
        ckpt_freq = ckpt_cfg['freq']
        ckpt_base_path = os.path.join(self.save_path, "checkpoints")
        if ckpt_enabled:
            os.makedirs(ckpt_base_path, exist_ok=True)

        def rev_log_func(loss, epoch_inner, epoch_outer):
            epoch = (epoch_outer - 1) * rev_epochs + epoch_inner
            nonlocal sample_reverse_loss
            sample_reverse_loss += loss
            if epoch_inner % reverse_log_freq == 0:
                avg_loss = sample_reverse_loss / reverse_log_freq
                sample_reverse_loss = 0
                logger.debug(
                    f"Epoch {epoch_outer}, Inner epoch {epoch_inner}, Reverse Model Loss: {avg_loss:.4f}"
                )
            self.writer.add_scalar("train/reverse_model_loss", loss, epoch)

        # Plotting cfg
        plot_cfg = tcfg['plot']
        plot_freq = plot_cfg['freq']
        plot_num = plot_cfg['num']
        plot_save_path = os.path.join(self.save_path, "plots")
        if not os.path.exists(plot_save_path):
            os.makedirs(plot_save_path)

        # Main training loop
        self.vi_model.train()
        # Timing accumulators
        last_time = time.perf_counter()
        train_start_time = time.perf_counter()

        # Determine starting epoch (override when resuming)
        start_epoch = 1
        if self.resume_ckpt_dir is not None and override_start_epoch:
            try:
                base = os.path.basename(self.resume_ckpt_dir.rstrip('/'))
                if base.startswith('epoch_'):
                    parsed = int(base.split('_')[1])
                    start_epoch = parsed + 1
                    logger.info(
                        f"Starting training from epoch {start_epoch} due to resume."
                    )
                else:
                    logger.warning(
                        f"Resume dir '{self.resume_ckpt_dir}' does not end with 'epoch_<n>'; starting from epoch 1."
                    )
            except Exception as e:
                logger.error(
                    f"Failed to parse starting epoch from resume dir '{self.resume_ckpt_dir}': {e}. Starting from epoch 1."
                )
                raise e

        for epoch in tqdm(
                range(start_epoch, num_epochs + 1),
                desc="Main Training",
                initial=start_epoch - 1,
                total=num_epochs,
        ):
            epoch_start_time = time.perf_counter()

            # Sample epsilon (used both for VI forward and as HMC init for uivi)
            t_vi0 = time.perf_counter()
            epsilon = torch.randn(batch_size,
                                  self.vi_model.epsilon_dim).to(self.device)

            # Sample z from variational distribution
            z, neg_score_implicit = self.vi_model.forward(epsilon)

            # Compute log prob under target distribution
            log_prob_target = self.target_model.logp(z)

            t_vi1 = time.perf_counter()
            time_vi_sample_step = t_vi1 - t_vi0

            t_ns0 = time.perf_counter()

            if self.reverse_model is not None:
                with torch.no_grad():
                    self.reverse_model.eval()
                    z_aux, epsilon_aux = self.reverse_model.sample(
                        z,
                        num_samples=reverse_sample_num,
                    )
            else:
                # UIVI: use HMC to sample epsilon ~ q(epsilon|z)
                z_aux, epsilon_aux, acc_rate = self.sample_epsilon_hmc(
                    z,
                    eps_init=epsilon,
                    num_samples=reverse_sample_num,
                    burn_in_steps=hmc_burn_in_steps,
                    step_size=hmc_step_size,
                    leapfrog_steps=hmc_leapfrog_steps,
                )
                self.writer.add_scalar(
                    "train/hmc_accept_rate",
                    acc_rate,
                    epoch,
                )

            with torch.no_grad():
                neg_score = self.vi_model.neg_score(z_aux, epsilon_aux)
                neg_score = neg_score.mean(dim=1)
                neg_score = neg_score.clone().detach()

            t_ns1 = time.perf_counter()
            time_neg_score_step = t_ns1 - t_ns0

            t_bw0 = time.perf_counter()

            # Compute loss
            loss = -torch.mean(log_prob_target + torch.sum(
                z * neg_score,
                dim=-1,
            ))

            optimizer_vi.zero_grad()
            loss.backward()
            optimizer_vi.step()
            scheduler_vi.step()

            t_bw1 = time.perf_counter()
            time_backward_step = t_bw1 - t_bw0

            # TensorBoard scalars
            self.writer.add_scalar("train/loss", loss.item(), epoch)
            self.writer.add_scalar(
                "time/vi_sample_step",
                time_vi_sample_step,
                epoch,
            )
            self.writer.add_scalar(
                "time/neg_score_step",
                time_neg_score_step,
                epoch,
            )
            self.writer.add_scalar(
                "time/backward_step",
                time_backward_step,
                epoch,
            )

            # Log the average distance from epsilon_aux to original epsilon
            avg_eps_distance = torch.mean(
                torch.norm(
                    epsilon_aux - epsilon.unsqueeze(1),
                    dim=-1,
                )).item()
            self.writer.add_scalar("train/avg_epsilon_distance",
                                   avg_eps_distance, epoch)

            sample_loss += loss.item()
            if epoch % loss_log_freq == 0:
                avg_loss = sample_loss / loss_log_freq
                current_time = time.perf_counter()
                avg_epoch_time = (current_time - last_time) / loss_log_freq

                logger.debug(
                    f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}, Avg Epoch Time: {avg_epoch_time:.4f}s"
                )
                # Reset accumulators
                sample_loss = 0
                last_time = current_time

            # Train reverse model
            if (self.reverse_model is not None
                    and (epoch % rev_update_freq) == 0):
                time_rev0 = time.perf_counter()
                if not rev_reuse_optimizer:
                    reverse_optimizer = torch.optim.Adam(
                        self.reverse_model.parameters(), lr=reverse_lr)
                self.train_reverse_model(
                    reverse_optimizer,
                    rev_epochs,
                    rev_batch_size,
                    progress_bar=False,
                    log_func=lambda l, e: rev_log_func(l, e, epoch),
                )
                time_rev1 = time.perf_counter()
                time_reverse_step = time_rev1 - time_rev0
                self.writer.add_scalar(
                    "time/avg_reverse_model_train_step",
                    time_reverse_step / rev_update_freq,
                    epoch,
                )

            # Generate and save samples
            if epoch % sample_freq == 0:
                _, z_sample = self.vi_model.sampling(num=sample_num)

                torch.save(
                    z_sample,
                    os.path.join(
                        sample_save_path,
                        f"samples_epoch_{epoch}.pt",
                    ),
                )
                logger.debug(f"Saved {sample_num} samples at epoch {epoch}.")

            # Save checkpoints
            if ckpt_enabled and (epoch % ckpt_freq == 0):
                epoch_ckpt_dir = os.path.join(ckpt_base_path, f"epoch_{epoch}")
                os.makedirs(epoch_ckpt_dir, exist_ok=True)
                # Save VI model
                vi_ckpt_path = os.path.join(epoch_ckpt_dir, "vi_model.pt")
                torch.save(self.vi_model.state_dict(), vi_ckpt_path)
                # Save VI optimizer and scheduler
                vi_opt_path = os.path.join(epoch_ckpt_dir, "vi_optim.pt")
                vi_sched_path = os.path.join(epoch_ckpt_dir, "vi_sched.pt")
                torch.save(optimizer_vi.state_dict(), vi_opt_path)
                torch.save(scheduler_vi.state_dict(), vi_sched_path)
                # Save reverse model (if exists)
                if self.reverse_model is not None:
                    rev_ckpt_path = os.path.join(
                        epoch_ckpt_dir,
                        "reverse_model.pt",
                    )
                    torch.save(self.reverse_model.state_dict(), rev_ckpt_path)
                    # Save reverse optimizer only if reusing optimizer
                    if rev_reuse_optimizer and reverse_optimizer is not None:
                        rev_opt_path = os.path.join(
                            epoch_ckpt_dir,
                            "reverse_optim.pt",
                        )
                        torch.save(
                            reverse_optimizer.state_dict(),
                            rev_opt_path,
                        )
                logger.debug(
                    f"Saved checkpoints at epoch {epoch} to {epoch_ckpt_dir}.")

            # Log KL divergence
            if kl_log_freq > 0 and (epoch % kl_log_freq == 0):
                t_kl0 = time.perf_counter()

                # Compute KL divergence against baseline
                baseline_kl = self.evaluate_vi_to_baseline_kl()
                self.writer.add_scalar(
                    "train/vi_kl_to_baseline",
                    baseline_kl,
                    epoch,
                )

                if self.reverse_model is not None:
                    # Compute KL divergence between true joint and reverse-induced joint
                    rev_kl_div = self.calculate_KL()
                    self.writer.add_scalar("train/reverse_kl_div", rev_kl_div,
                                           epoch)

                t_kl1 = time.perf_counter()
                time_kl_step = t_kl1 - t_kl0
                self.writer.add_scalar(
                    "time/avg_kl_calculation_step",
                    time_kl_step / kl_log_freq,
                    epoch,
                )
                kl_log_message = f"Epoch {epoch}, VI KL to baseline: {baseline_kl:.4f}"
                if self.reverse_model is not None:
                    kl_log_message += f", Reverse KL: {rev_kl_div:.4f}"
                logger.debug(kl_log_message)

            # Generate and save contour plots
            if epoch % plot_freq == 0:
                t_plot0 = time.perf_counter()
                _, z_plot = self.vi_model.sampling(num=plot_num)

                self.target_model.contour_plot(
                    DEFAULT_BBOX[self.target_dist],
                    fnet=None,
                    samples=z_plot.cpu().numpy(),
                    save_to_path=os.path.join(
                        plot_save_path,
                        f"contour_epoch_{epoch}.png",
                    ),
                    quiver=False,
                    t=epoch,
                )
                logger.debug(f"Saved contour plot at epoch {epoch}.")
                t_plot1 = time.perf_counter()
                time_plot_step = t_plot1 - t_plot0
                self.writer.add_scalar(
                    "time/avg_plotting_step",
                    time_plot_step / plot_freq,
                    epoch,
                )
            epoch_end_time = time.perf_counter()
            epoch_time = epoch_end_time - epoch_start_time
            self.writer.add_scalar(
                "time/epoch",
                epoch_time,
                epoch,
            )

        # Close writer at end
        total_time = time.perf_counter() - train_start_time
        avg_epoch_time = total_time / max(1, num_epochs)
        logger.info(
            f"Training completed. Total time: {total_time:.3f}s, Avg epoch time: {avg_epoch_time:.6f}s"
        )
        self.writer.close()


if __name__ == "__main__":
    runner = ReverseUIVI(device=device,
                         cfg=cfg,
                         resume_ckpt_dir=args.resume_ckpt_dir)
    runner.learn()
