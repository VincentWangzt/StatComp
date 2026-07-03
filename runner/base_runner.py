import torch
from models.target_models import target_distribution
from models.vi_model import VIModel
import os
from tqdm import tqdm
from datetime import datetime
from utils.logging import get_logger, set_file_handler
import ite
import time
import numpy as np
from omegaconf import OmegaConf, DictConfig
from utils.annealing import annealing
from utils.elm import kde_expected_log_marginal
from utils.metrics import compute_sliced_wasserstein, compute_ksd, compute_mmd
from utils.experiment_logging import ExperimentLogger, metric, timer

logger = get_logger()


class BaseSIVIRunner():
    '''
    The base Reverse SIVI class that encapsulates the model, training, and evaluation.

    Key components:
    - Target model: provides `logp` and plotting utilities.
    - VI model: parameterizes q_phi(z|epsilon)
    - Reverse model [Optional]: parameterizes q_psi(epsilon|z) via normalizing flow
    - Logging and artifact paths: W&B data and local files under `results/`.

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
        self.config.setdefault('output', {})
        results_dir = self.config.output.get('results_dir', 'results')
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.timestamp = timestamp
        self.save_path = os.path.join(results_dir, self.name, self.target_type,
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
        self.target_model = self._build_target_model()

        # Override z_dim/epsilon_dim in target config if the target model
        # reports a different dimensionality (data-dependent targets).
        model_z_dim = getattr(self.target_model, 'z_dim', None)
        if model_z_dim is not None:
            cfg_z_dim = self.config.target.get('z_dim', None)
            if cfg_z_dim is not None and cfg_z_dim != model_z_dim:
                logger.info(
                    f"Overriding target z_dim from config ({cfg_z_dim}) "
                    f"with model z_dim ({model_z_dim})")
            self.config.target.z_dim = model_z_dim
            self.config.target.epsilon_dim = model_z_dim

        # baseline sample
        self.baseline_samples = self._load_baseline_samples()

        # kl ite samples
        self.metric_kl_enabled = self.config['metric']['kl_ite'].setdefault(
            'enabled', True)
        if self.metric_kl_enabled and self.baseline_samples is None:
            logger.warning(
                "No baseline samples available; disabling KL metric.")
            self.metric_kl_enabled = False
        self.n_ite_samples = self.config['metric']['kl_ite']['num_samples']

        # w2 samples
        self.metric_w2_enabled = self.config['metric']['w2'].setdefault(
            'enabled', True)
        if self.metric_w2_enabled and self.baseline_samples is None:
            logger.warning(
                "No baseline samples available; disabling W2 metric.")
            self.metric_w2_enabled = False
        self.n_w2_samples = self.config['metric']['w2']['num_samples']
        self.n_w2_projections = self.config['metric']['w2']['num_projections']

        # ksd config (KSD uses target_model.score(), not baseline samples)
        self.config.metric.setdefault('ksd', {})
        self.metric_ksd_enabled = self.config['metric']['ksd'].setdefault(
            'enabled', False)
        self.n_ksd_samples = self.config['metric']['ksd'].setdefault(
            'num_samples', 1000)

        # mmd config
        self.config.metric.setdefault('mmd', {})
        self.metric_mmd_enabled = self.config['metric']['mmd'].setdefault(
            'enabled', False)
        if self.metric_mmd_enabled and self.baseline_samples is None:
            logger.warning(
                "No baseline samples available; disabling MMD metric.")
            self.metric_mmd_enabled = False
        self.n_mmd_samples = self.config['metric']['mmd'].setdefault(
            'num_samples', 1000)
        self._init_mmd_baseline_samples()

        # fisher divergence config
        self.config.metric.setdefault('fisher', {})
        self.metric_fisher_enabled = self.config['metric'][
            'fisher'].setdefault('enabled', False)
        self.n_fisher_samples = self.config['metric']['fisher'].setdefault(
            'num_samples', 1000)
        self.n_fisher_is_samples = self.config['metric']['fisher'].setdefault(
            'num_is_samples', 512)

        # elbo samples
        self.metric_elbo_enabled = self.config['metric']['elbo'].setdefault(
            'enabled', True)
        self.n_elbo_z_samples = self.config['metric']['elbo']['num_z_samples']
        self.n_elbo_batches = self.config['metric']['elbo']['num_batches']
        self.n_elbo_batch_size = self.config['metric']['elbo']['batch_size']

        # KDE expected log marginal samples
        self.config.metric.setdefault('expected_log_marginal', {})
        self.metric_expected_log_marginal_enabled = self.config['metric'][
            'expected_log_marginal'].setdefault('enabled', True)
        if (self.metric_expected_log_marginal_enabled
                and self.baseline_samples is None):
            logger.warning(
                "No baseline samples available; disabling expected log marginal metric."
            )
            self.metric_expected_log_marginal_enabled = False
        self.n_expected_log_marginal_ref_samples = self.config['metric'][
            'expected_log_marginal'].setdefault('num_ref_samples', 1000)
        self.n_expected_log_marginal_model_samples = self.config['metric'][
            'expected_log_marginal'].setdefault('num_model_samples', 5000)
        self.n_expected_log_marginal_sample_batch_size = self.config['metric'][
            'expected_log_marginal'].setdefault('sample_batch_size', 5000)
        self.n_expected_log_marginal_dim_chunk = self.config['metric'][
            'expected_log_marginal'].setdefault('dim_chunk', 25)
        self.n_expected_log_marginal_ref_chunk = self.config['metric'][
            'expected_log_marginal'].setdefault('ref_chunk', 500)
        self.n_expected_log_marginal_model_chunk = self.config['metric'][
            'expected_log_marginal'].setdefault('model_chunk', 20000)
        self.expected_log_marginal_min_bandwidth = self.config['metric'][
            'expected_log_marginal'].setdefault('min_bandwidth', 1.0e-6)
        self.expected_log_marginal_dtype = self.config['metric'][
            'expected_log_marginal'].setdefault('dtype', 'float32')
        self._expected_log_marginal_reference_samples = None

        # bnn metrics config (RMSE + test log-likelihood; BNN targets only)
        # Auto-detect: enable if target is a DataBoundTarget wrapping Bnn,
        # unless explicitly configured via metric.bnn.enabled in the runner config.
        self.config.metric.setdefault('bnn', {})
        from models.data_bound_target import DataBoundTarget
        from models.target_models import Bnn
        _is_bnn_target = (isinstance(self.target_model, DataBoundTarget)
                          and isinstance(self.target_model.inner, Bnn))
        self.metric_bnn_enabled = self.config['metric']['bnn'].setdefault(
            'enabled', _is_bnn_target)
        if self.metric_bnn_enabled and not _is_bnn_target:
            logger.warning(
                "BNN metrics enabled but target is not a BNN; disabling.")
            self.metric_bnn_enabled = False
        self.n_bnn_samples = self.config['metric']['bnn'].setdefault(
            'num_samples', 500)

        # Jacobian spectral norm (Assumption 1 validation)
        self.config.metric.setdefault('jacobian_spectral', {})
        self.metric_jacobian_spectral_enabled = self.config['metric'][
            'jacobian_spectral'].setdefault('enabled', False)
        self.n_jacobian_spectral_samples = self.config['metric'][
            'jacobian_spectral'].setdefault('num_samples', 128)

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
        self.vi_opt_betas = tuple(self.vi_opt_cfg.get('betas', (0.9, 0.999)))
        self.vi_var_lr = self.vi_opt_cfg.get('var_lr', None)
        assert self.vi_scheduler_cfg['type'] == 'StepLR', \
            "Only StepLR scheduler is supported for VI optimizer."

        # Create VI optimizer and scheduler
        if self.vi_var_lr is not None and hasattr(self.vi_model, 'var_raw'):
            other_params = [
                p for n, p in self.vi_model.named_parameters()
                if n != 'var_raw'
            ]
            self.optimizer_vi = torch.optim.Adam(
                [
                    {
                        'params': other_params,
                        'lr': self.vi_lr
                    },
                    {
                        'params': [self.vi_model.var_raw],
                        'lr': self.vi_var_lr
                    },
                ],
                betas=self.vi_opt_betas,
            )
        else:
            self.optimizer_vi = torch.optim.Adam(
                self.vi_model.parameters(),
                lr=self.vi_lr,
                betas=self.vi_opt_betas,
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
        self.experiment_logger = ExperimentLogger(
            save_path=self.save_path,
            config=self.config,
            runner_name=self.name,
            target_type=self.target_type,
            vi_model_type=self.vi_model_type,
            seed=int(self.config.get('seed', 42)),
            time_avg_window=self.train_time_avg_window,
        )

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
        self._log_groundtruth_plot()

        # Gradient clipping (None = disabled)
        self.grad_clip = self.training_cfg.get('grad_clip', None)
        if self.grad_clip is not None:
            logger.info(
                f"Gradient clipping enabled with max_norm={self.grad_clip}")

        # Optional VI pretraining on BNN dev splits.
        self.pretrain_cfg = self.training_cfg.get('pretrain', {})
        self.pretrain_enabled: bool = self.pretrain_cfg.get('enabled', False)
        self.pretrain_steps: int = int(self.pretrain_cfg.get('steps', 0))
        self.pretrain_lr: float = float(
            self.pretrain_cfg.get('lr', self.vi_lr))
        self.pretrain_batch_size: int = int(
            self.pretrain_cfg.get('batch_size', self.training_batch_size))
        self._vi_pretrained_done = False

        # EMA (Exponential Moving Average) for stable evaluation
        ema_cfg = self.training_cfg.get('ema', {})
        self.ema_enabled = ema_cfg.get('enabled', False)
        if self.ema_enabled:
            from utils.ema import EMA
            self.ema_beta = ema_cfg.get('beta', 0.999)
            self.ema = EMA(
                beta=self.ema_beta,
                model_params=self.vi_model.parameters(),
            )
            logger.info(f"EMA enabled with beta={self.ema_beta}")

    # Data-dependent target types that require the DataBoundTarget wrapper
    _DATA_DEPENDENT_TARGETS = frozenset({
        "LRwaveform",
        "Bnn_boston",
        "Bnn_concrete",
        "Bnn_power",
        "Bnn_protein",
        "Bnn_winered",
        "Bnn_yacht",
    })

    def _build_target_model(self):
        """Instantiate the target model.

        For standard targets, delegates to the ``target_distribution`` registry.
        For data-dependent targets (``LRwaveform``, ``Bnn_boston``), uses the
        :func:`~models.data_bound_target.build_data_bound_target` factory which
        loads data and wraps the inner model.
        """
        if self.target_type in self._DATA_DEPENDENT_TARGETS:
            from models.data_bound_target import build_data_bound_target

            target_cfg = OmegaConf.to_container(self.config.get('target', {}),
                                                resolve=True)
            return build_data_bound_target(
                target_type=self.target_type,
                target_cfg=target_cfg,
                device=self.device,
            )
        return target_distribution[self.target_type](device=self.device)

    def log_config(self):
        '''
        Log the full configuration to W&B and save as YAML file.
        '''
        config_str = OmegaConf.to_yaml(self.config, resolve=True)

        # Save to YAML file
        config_save_path = os.path.join(self.save_path, "full_config.yaml")
        with open(config_save_path, 'w') as f:
            f.write(config_str)
        self.experiment_logger.update_config(self.config)
        logger.info(f"Saved full configuration to {config_save_path}.")

    def _load_baseline_samples(self) -> np.ndarray | None:
        """
        Load baseline MCMC samples from a configured path (`self.config.target.baseline_path`) for the current target. If not available, use a default path `baselines/mcmc/{target_dist}.pt`.

        Returns:
            samples (np.ndarray | None): Loaded baseline samples on cpu, or None if unavailable.

        """
        baseline_path = self.config.target.get('baseline_path', None)

        if not baseline_path:
            baseline_path = f'baselines/mcmc/{self.target_type}.pt'
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
            logger.warning(
                f"Failed to load baseline samples from {baseline_path}: {e}. "
                f"KL and W2 metrics will be disabled.")
            return None

    def _groundtruth_plot_samples(self) -> torch.Tensor | None:
        """Return samples suitable for one-time ground-truth plotting."""
        num_samples = int(self.plot_num)
        if self.baseline_samples is not None:
            if self.baseline_samples.shape[0] > num_samples:
                indices = np.random.choice(
                    self.baseline_samples.shape[0],
                    num_samples,
                    replace=False,
                )
                samples = self.baseline_samples[indices]
            else:
                samples = self.baseline_samples
            return torch.as_tensor(
                samples,
                dtype=torch.float32,
                device=self.device,
            )

        sample = getattr(self.target_model, "sample", None)
        if callable(sample):
            try:
                return sample(num_samples).detach().to(self.device)
            except Exception as e:
                logger.debug(
                    f"Failed to draw target samples for ground-truth plot: {e}"
                )
        return None

    def _log_groundtruth_plot(self) -> None:
        """Save and log a one-time target/baseline sample plot when supported."""
        samples = self._groundtruth_plot_samples()
        if samples is None:
            logger.info(
                "Skipping ground-truth sample plot; no baseline or target sampler is available."
            )
            return

        contour_path = os.path.join(self.plot_save_path, "groundtruth_contour.png")
        trace_path = os.path.join(self.plot_save_path, "groundtruth_trace.png")

        try:
            self.target_model.contour_plot(
                self.config.target.bbox,
                fnet=None,
                samples=samples.detach().cpu().numpy(),
                save_to_path=contour_path,
                quiver=False,
                t=None,
            )
            plot_path = contour_path
            logger.info(f"Saved ground-truth contour plot to {plot_path}.")
        except (NotImplementedError, AttributeError, KeyError):
            try:
                self.target_model.trace_plot(
                    samples,
                    figpath=self.plot_save_path,
                    figname="groundtruth_trace.png",
                    figtitle="Ground Truth",
                )
                plot_path = trace_path
                logger.info(f"Saved ground-truth trace plot to {plot_path}.")
            except (NotImplementedError, AttributeError) as e:
                logger.info(f"Skipping ground-truth sample plot: {e}")
                return
            except Exception as e:
                logger.warning(f"Failed to create ground-truth trace plot: {e}")
                return
        except Exception as e:
            logger.warning(f"Failed to create ground-truth contour plot: {e}")
            return

        self.experiment_logger.log_image(
            "plots/groundtruth_samples",
            plot_path,
            step=0,
        )

    def evaluate_vi_to_baseline_kl(self) -> float:
        """
        Estimate KL divergence KL(q_phi(z) || q_baseline(z)) using `ite.cost.BDKL_KnnK`.

        ``BDKL_KnnK`` is a kNN-based estimator and is **sensitive to the
        relative sample sizes** of the two empirical distributions: feeding
        N q-samples vs. M baseline samples with N != M biases the estimate.
        To match the IVI reference implementation (which uses the same N
        for both), we subsample the baseline to ``n_ite_samples`` rows
        before estimating.

        Returns:
            kl_div (float): Estimated KL divergence value.
        """
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute KL divergence.")

        _, z = self.vi_model.sampling(num=self.n_ite_samples)
        z_np = z.cpu().numpy()

        # Subsample baseline to match number of q samples — fresh draw each
        # call, no replacement, so KL trajectories track the q_phi changes
        # rather than fluctuations in baseline sample size.
        if self.baseline_samples.shape[0] > self.n_ite_samples:
            indices = np.random.choice(
                self.baseline_samples.shape[0],
                self.n_ite_samples,
                replace=False,
            )
            baseline_subset = self.baseline_samples[indices]
        else:
            baseline_subset = self.baseline_samples

        cost_obj = ite.cost.BDKL_KnnK()
        try:
            kl_div = cost_obj.estimation(z_np, baseline_subset)
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

    @metric("metric/vi_model/kl_ite")
    @timer("kl_estimation")
    def eval_kl_ite(self, epoch: int) -> float:
        '''
        Evaluate KL divergence between VI and baseline and log centrally.
        Args:
            epoch (int): Current epoch number.
        '''
        kl_div = self.evaluate_vi_to_baseline_kl()
        logger.debug(f"Epoch {epoch}, VI KL to baseline: {kl_div:.4f}")
        return kl_div

    @metric("metric/vi_model/w2")
    @timer("w2_estimation")
    def eval_w2(self, epoch: int) -> float:
        '''
        Evaluate W2 distance between VI and baseline and log centrally.
        Args:
            epoch (int): Current epoch number.
        '''
        w2_dist = self.evaluate_vi_to_baseline_w2()
        logger.debug(f"Epoch {epoch}, VI W2 to baseline: {w2_dist:.4f}")
        return w2_dist

    def evaluate_elbo(self) -> tuple[float, float, float, float]:
        """
        Estimate ELBO using importance sampling for q_phi(z).
        ELBO = E_{z ~ q_phi} [log p(z) - log q_phi(z)]

        To estimate log q_phi(z), we use:
        q_phi(z) = E_{epsilon' ~ p(epsilon)} [q_phi(z|epsilon')]
        approximated by Monte Carlo integration over epsilon'.

        We use multiple batches to also estimate the standard error of q_phi(z) estimation.

        Returns:
            (elbo_mean, elbo_std_total, elbo_std_q, elbo_ci_half) (float, float, float, float): Estimated ELBO mean, total std, std from q(z) estimation, and 1/2 width of 0.95 CI.
        """
        # 1. Sample z from q_phi(z) and keep the generating epsilon so it can
        # be included in the Monte Carlo estimate for each sampled z.
        epsilon_samples, z_samples = self.vi_model.sampling(
            num=self.n_elbo_z_samples)
        # z_samples: [N_z, Dz]

        # 2. Estimate log q_phi(z) for each z sample
        # q_phi(z) \approx (1/K) \sum_{k=1}^K q_phi(z|epsilon'_k)
        # log q_phi(z) \approx logsumexp(log q_phi(z|epsilon'_k)) - log K

        # We perform this for multiple batches of epsilon' to get variance estimate
        # Batches: B batches of size S

        # Accumulate q(z) estimate from auxiliary epsilon draws.
        # We work in log space for stability and add the generating epsilon
        # once after aggregating all auxiliary batches.
        batch_log_q_z_sums = []
        total_aux_samples = self.n_elbo_batches * self.n_elbo_batch_size

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

        # Add the generating epsilon contribution once per sampled z.
        log_q_z_given_generating_eps = self.vi_model.logp(
            z_samples,
            epsilon_samples,
        )

        # --- Total Estimate (using B*S auxiliary samples plus one generating
        # epsilon contribution) ---
        log_aux_total_sum = torch.logsumexp(log_sums_tensor, dim=1)  # [N_z]
        log_total_sum = torch.logaddexp(
            log_aux_total_sum,
            log_q_z_given_generating_eps,
        )
        total_samples = total_aux_samples + 1
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

        # Squared standard error of mean estimator (of q(z)).
        # The one-time generating-epsilon term is fixed conditional on z, so
        # only the auxiliary-average term contributes Monte Carlo variance.
        aux_weight = total_aux_samples / total_samples
        sq_se_mean_q = (aux_weight**2) * var_estimators / self.n_elbo_batches

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

        # 1/2 width of 0.95 confidence interval
        elbo_ci_half = 1.96 * elbo_std_total / (self.n_elbo_z_samples**0.5)

        return elbo_mean.item(), elbo_std_total.item(), elbo_std_q.item(
        ), elbo_ci_half.item()

    @metric(prefix="metric/vi_model")
    @timer("elbo_estimation")
    def eval_elbo(self, epoch: int) -> dict[str, float]:
        '''
        Evaluate ELBO metric and log centrally.
        '''
        elbo_val, elbo_std_total, elbo_std_q, elbo_ci_half = self.evaluate_elbo(
        )
        logger.debug(
            f"Epoch {epoch}, ELBO: {elbo_val:.4f}, Std Total: {elbo_std_total:.4f}, Std Q: {elbo_std_q:.4f}, CI Half: {elbo_ci_half:.4f}"
        )
        return {
            "elbo": elbo_val,
            "elbo_std_total": elbo_std_total,
            "elbo_std_q": elbo_std_q,
            "elbo_ci_half": elbo_ci_half,
        }

    def _sample_reference_baseline_samples(self,
                                           num_samples: int) -> torch.Tensor:
        """Sample reference points from the baseline store without replacement."""
        if self.baseline_samples is None:
            raise RuntimeError(
                "Baseline samples not loaded; cannot compute expected log marginal."
            )

        if self.baseline_samples.shape[0] > num_samples:
            indices = np.random.choice(
                self.baseline_samples.shape[0],
                num_samples,
                replace=False,
            )
            reference_samples = self.baseline_samples[indices]
        else:
            reference_samples = self.baseline_samples

        return torch.as_tensor(reference_samples, device=self.device)

    def _expected_log_marginal_reference_set(self) -> torch.Tensor:
        """Return the fixed reference set used for training-time KDE ELM."""
        if self._expected_log_marginal_reference_samples is None:
            self._expected_log_marginal_reference_samples = (
                self._sample_reference_baseline_samples(
                    self.n_expected_log_marginal_ref_samples))
        return self._expected_log_marginal_reference_samples

    @torch.no_grad()
    def _sample_vi_model_for_kde(self, num_samples: int,
                                 batch_size: int) -> torch.Tensor:
        """Draw VI samples for the coordinate-wise KDE ELM metric."""
        if num_samples < 1:
            raise ValueError("num_model_samples must be at least 1.")
        if batch_size < 1:
            raise ValueError("sample_batch_size must be at least 1.")

        samples = torch.empty(
            (num_samples, self.z_dim),
            device=self.device,
            dtype=next(self.vi_model.parameters()).dtype,
        )
        was_training = self.vi_model.training
        self.vi_model.eval()
        try:
            for start in range(0, num_samples, batch_size):
                current = min(batch_size, num_samples - start)
                _, z = self.vi_model.sampling(num=current)
                samples[start:start + current].copy_(z.detach())
        finally:
            if was_training:
                self.vi_model.train()
        return samples

    def evaluate_expected_log_marginal(self):
        r"""Estimate paper-style KDE expected log marginal.

        The metric is

            E_{z ~ r}[sum_j log q_hat_phi,j(z_j)],

        where ``r`` is represented by reference baseline samples and each
        ``q_hat_phi,j`` is a one-dimensional Gaussian KDE fit from VI samples.

        Returns:
            KDEELMEstimate: Scalar metric, per-reference values, and
                diagnostics from the chunked KDE evaluator.
        """
        reference_samples = self._expected_log_marginal_reference_set()
        model_samples = self._sample_vi_model_for_kde(
            self.n_expected_log_marginal_model_samples,
            self.n_expected_log_marginal_sample_batch_size,
        )
        return kde_expected_log_marginal(
            reference_samples,
            model_samples,
            dim_chunk=self.n_expected_log_marginal_dim_chunk,
            ref_chunk=self.n_expected_log_marginal_ref_chunk,
            model_chunk=self.n_expected_log_marginal_model_chunk,
            min_bandwidth=self.expected_log_marginal_min_bandwidth,
            dtype=self.expected_log_marginal_dtype,
            device=self.device,
        )

    @metric()
    @timer("expected_log_marginal_estimation")
    def eval_expected_log_marginal(self, epoch: int) -> dict[str, float]:
        """Evaluate KDE expected log marginal and route it through the logger."""
        estimate = self.evaluate_expected_log_marginal()
        metric_mean = estimate.value
        diagnostics = estimate.diagnostics
        logger.debug(
            f"Epoch {epoch}, KDE Expected Log Marginal: {metric_mean:.4f}"
        )
        return {
            "metric/vi_model/expected_log_marginal": metric_mean,
            "metric/vi_model/kde_expected_log_marginal": metric_mean,
            "diagnostic/vi_model/kde_expected_log_marginal_std": diagnostics[
                "std_across_refs"
            ],
            "diagnostic/vi_model/kde_expected_log_marginal_clamped_dims": diagnostics[
                "num_bandwidth_clamped_dims"
            ],
        }

    def evaluate_ksd(self) -> float:
        '''
        Evaluate Kernelized Stein Discrepancy (KSD) for VI samples.
        Returns:
            ksd (float): Estimated KSD value.
        '''
        _, z = self.vi_model.sampling(num=self.n_ksd_samples)
        scores = self.target_model.score(z)
        try:
            ksd = compute_ksd(
                z,
                scores=scores,
            )
            return float(ksd)
        except Exception as e:
            logger.error(f"KSD estimation failed: {e}")
            raise e

    @metric("metric/vi_model/ksd")
    @timer("ksd_estimation")
    def eval_ksd(self, epoch: int) -> float:
        '''
        Evaluate KSD metric and log centrally.
        Args:
            epoch (int): Current epoch number.
        '''
        ksd_val = self.evaluate_ksd()
        logger.debug(f"Epoch {epoch}, VI KSD: {ksd_val:.4f}")
        return ksd_val

    def _init_mmd_baseline_samples(self):
        '''
        Initialize baseline samples and kernel for MMD evaluation.
        '''
        if not self.metric_mmd_enabled:
            return

        if self.baseline_samples.shape[0] > self.n_mmd_samples:
            self._mmd_baseline_subset = self.baseline_samples[np.random.choice(
                self.baseline_samples.shape[0],
                self.n_mmd_samples,
                replace=False,
            )]
        else:
            self._mmd_baseline_subset = self.baseline_samples

        self.mmd_baseline_subset = torch.as_tensor(
            self._mmd_baseline_subset).to(self.device)

        from utils.kernels import GaussianKernel
        self.mmd_baseline_kernel = GaussianKernel()
        self.mmd_baseline_kernel.fit_h(self.mmd_baseline_subset)

    def evaluate_mmd(self) -> float:
        '''
        Evaluate Maximum Mean Discrepancy (MMD) for VI samples.
        Returns:
            mmd (float): Estimated MMD value.
        '''
        _, z = self.vi_model.sampling(num=self.n_mmd_samples)

        try:
            mmd = compute_mmd(
                z,
                self.mmd_baseline_subset,
                self.mmd_baseline_kernel,
            )
            return float(mmd)
        except Exception as e:
            logger.error(f"MMD estimation failed: {e}")
            raise e

    @metric("metric/vi_model/mmd")
    @timer("mmd_estimation")
    def eval_mmd(self, epoch: int) -> float:
        '''
        Evaluate MMD metric and log centrally.
        Args:
            epoch (int): Current epoch number.
        '''
        mmd_val = self.evaluate_mmd()
        logger.debug(f"Epoch {epoch}, VI MMD: {mmd_val:.4f}")
        return mmd_val

    def evaluate_bnn_metrics(self) -> tuple[float, float]:
        """Sample from VI and compute BNN test RMSE and test log-likelihood.

        Returns:
            (rmse, test_llk): RMSE of ensemble mean predictions, and Monte Carlo
            marginalised test log-likelihood. NLL = -test_llk.
        """
        with torch.no_grad():
            _, z = self.vi_model.sampling(num=self.n_bnn_samples)
        return self.target_model.rmse_llk(z)

    @metric(prefix="metric/vi_model")
    @timer("bnn_estimation")
    def eval_bnn(self, epoch: int) -> dict[str, float]:
        """Evaluate BNN RMSE / test log-likelihood and log centrally.

        Args:
            epoch (int): Current epoch number.
        """
        rmse, test_llk = self.evaluate_bnn_metrics()
        logger.debug(
            f"Epoch {epoch}, BNN RMSE: {rmse:.4f}, Test LLK: {test_llk:.4f}, NLL: {-test_llk:.4f}"
        )
        return {"rmse": rmse, "test_llk": test_llk, "nll": -test_llk}

    def evaluate_fisher_divergence(self) -> float:
        """
        Estimate the Fisher divergence between q_phi(z) and the target p(z):

            FD(p || q) = E_{z ~ q_phi} [ || score_p(z) - score_q(z) ||^2 ]

        where:
            score_p(z) = nabla_z log p(z)  -- from target_model.score()
            score_q(z) = nabla_z log q_phi(z)
                       ~ sum_k softmax(log q(z|eps_k)) * nabla_z log q(z|eps_k)
                       = E_{eps ~ p(eps)} [ nabla_z log q(z|eps) ]  (IS estimate)

        The IS estimate of score_q uses the same logsumexp-softmax weighting as
        the ELBO estimator. vi_model.score(z, eps) = -(z - mu(eps)) / var(eps)
        gives nabla_z log q(z|eps) analytically.

        Returns:
            fisher_div (float): Estimated Fisher divergence.
        """
        with torch.no_grad():
            # 1. Sample z ~ q_phi
            _, z_samples = self.vi_model.sampling(num=self.n_fisher_samples)
            # z_samples: [N, Dz]

            # 2. Sample epsilon' ~ p(epsilon) for IS
            eps = self.vi_model.sample_epsilon(num=self.n_fisher_is_samples)
            # eps: [K, De]

            # Expand for joint computation: z [N,1,Dz], eps [1,K,De]
            z_exp = z_samples.unsqueeze(1).expand(-1, self.n_fisher_is_samples,
                                                  -1)
            eps_exp = eps.unsqueeze(0).expand(self.n_fisher_samples, -1, -1)

            # 3. log q(z|eps_k): [N, K]
            log_q_z_given_eps = self.vi_model.logp(z_exp, eps_exp)

            # 4. Softmax weights over K: [N, K]
            log_w = log_q_z_given_eps - torch.logsumexp(
                log_q_z_given_eps, dim=1, keepdim=True)
            w = torch.exp(log_w)  # [N, K]

            # 5. nabla_z log q(z|eps_k): [N, K, Dz]
            score_q_given_eps = self.vi_model.score(z_exp, eps_exp)

            # 6. Weighted sum -> score_q(z): [N, Dz]
            score_q = (w.unsqueeze(-1) * score_q_given_eps).sum(dim=1)

            # 7. Ground truth score: [N, Dz]
            score_p = self.target_model.score(z_samples)

            # 8. Fisher divergence: E[ || score_p - score_q ||^2 ]
            fisher_div = torch.mean(torch.sum((score_p - score_q)**2, dim=-1))

        return fisher_div.item()

    @metric("metric/vi_model/fisher_div")
    @timer("fisher_estimation")
    def eval_fisher(self, epoch: int) -> float:
        '''
        Evaluate Fisher divergence and log centrally.
        Args:
            epoch (int): Current epoch number.
        '''
        fisher_val = self.evaluate_fisher_divergence()
        logger.debug(f"Epoch {epoch}, Fisher Divergence: {fisher_val:.4f}")
        return fisher_val

    @metric(prefix="metric/vi_model")
    @timer("jacobian_spectral_estimation")
    def eval_jacobian_spectral(self, epoch: int) -> dict[str, float]:
        '''
        Evaluate Jacobian spectral norms (Bounded Reparameterization Assumption)
        and log centrally.

        Computes E_ε[‖∇_φ μ_φ(ε)‖₂⁴] and E_ε[‖∇_φ σ_φ(ε)‖₂⁴] where the
        norm is the matrix 2-norm (spectral norm) of the d_z × d_φ Jacobian.

        Args:
            epoch (int): Current epoch number.
        '''
        from utils.jacobian_spectral import evaluate_assumption_bound

        was_training = self.vi_model.training
        self.vi_model.eval()

        epsilon = self.vi_model.sample_epsilon(
            num=self.n_jacobian_spectral_samples)
        bound = evaluate_assumption_bound(self.vi_model, epsilon)

        logger.debug(
            f"Epoch {epoch}, VI_D4M: {bound.M_eps:.4f} "
            f"(mu: {bound.mean_sq_spectral_mu:.4f}, "
            f"std: {bound.mean_sq_spectral_std:.4f})")

        if was_training:
            self.vi_model.train()
        return {
            "jacobian_spectral_mu": bound.mean_sq_spectral_mu,
            "jacobian_spectral_std": bound.mean_sq_spectral_std,
            "vi_derivative_fourth_moment": bound.M_eps,
        }

    def log_reverse_score_l2_to_target(
        self,
        score_eval: torch.Tensor,
        z_eval: torch.Tensor,
    ) -> None:
        '''
        Log the mean squared L2 gap between a reverse-estimated score and the
        target score at the same latent samples.

        Args:
            score_eval (torch.Tensor): Reverse-estimated score with shape [B, Dz].
            z_eval (torch.Tensor): Latent samples with shape [B, Dz].
        '''
        with torch.no_grad():
            target_score = self.target_model.score(z_eval.detach())
            score_l2 = torch.mean(
                torch.sum((score_eval.detach() - target_score)**2, dim=-1))
            self.experiment_logger.log_scalars(
                {"diagnostic/reverse_model/score_l2_to_target": score_l2.item()},
                step=self.curr_epoch,
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

    def pretrain_vi(self):
        """Optional VI pretraining hook for BNN targets with dev splits."""
        from models.data_bound_target import DataBoundTarget

        if (not self.pretrain_enabled or self.pretrain_steps <= 0 or
                not isinstance(self.target_model, DataBoundTarget) or
                self.target_model.dev_data is None or
                not hasattr(self.target_model.inner, 'predict_y')):
            return

        X_dev, y_dev, mean_y, std_y = self.target_model.dev_data
        if (X_dev is None or y_dev is None or mean_y is None or std_y is None):
            return

        optimizer = torch.optim.Adam(
            self.vi_model.parameters(),
            lr=self.pretrain_lr,
            betas=self.vi_opt_betas,
        )
        log_freq = max(1, self.pretrain_steps // 10)
        logger.info(
            "Starting VI pretraining on dev split: "
            f"steps={self.pretrain_steps}, lr={self.pretrain_lr}"
        )

        self.vi_model.train()
        for step in range(1, self.pretrain_steps + 1):
            epsilon = self.vi_model.sample_epsilon(num=self.pretrain_batch_size)
            z, _ = self.vi_model.forward(epsilon)
            pred_y = self.target_model.inner.predict_y(z, X_dev, mean_y, std_y)
            loss = ((pred_y.mean(0) - y_dev)**2).mean()
            optimizer.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.vi_model.parameters(),
                    max_norm=self.grad_clip,
                )
            optimizer.step()

            if step % log_freq == 0 or step == self.pretrain_steps:
                self.experiment_logger.log_scalars(
                    {"pretrain/vi_model/loss": loss.item()}, step=step)
                logger.info(
                    f"VI pretrain step {step}/{self.pretrain_steps}: "
                    f"loss={loss.item():.6f}"
                )

        # Reset EMA after pretraining so evaluation starts from the pretrained
        # weights rather than the random initialization snapshot.
        if self.ema_enabled:
            from utils.ema import EMA
            self.ema = EMA(
                beta=self.ema_beta,
                model_params=self.vi_model.parameters(),
            )
        self._vi_pretrained_done = True

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

    def _compute_loss_and_step(self, epoch: int) -> dict:
        """
        Compute the training loss, perform the optimizer step, and return diagnostics.

        This default implementation computes the ELBO-based loss used by standard SIVI
        and its reverse-model variants. Subclasses (e.g., KSIVIRunner) may override this
        to implement alternative objectives such as KSD².

        Args:
            epoch (int): Current training epoch.

        Returns:
            dict with keys:
                - 'loss' (torch.Tensor): Scalar loss value.
                - 'grad_norm' (float): Gradient norm before step.
                - 'z' (torch.Tensor): Sampled z for diagnostic logging.
                - 'epsilon' (torch.Tensor): Sampled epsilon for diagnostic logging.
            Phase timings are emitted directly through ``ExperimentLogger``.
        """
        with self.experiment_logger.timer("vi_sample", step=epoch):
            epsilon = self.vi_model.sample_epsilon(num=self.training_batch_size)
            z, neg_score_implicit = self.vi_model.forward(epsilon)
            log_prob_target: torch.Tensor = self.target_model.score(
                z.clone().detach()) * z
            log_prob_target = log_prob_target.sum(dim=-1)
            anneal_factor = annealing(
                t=epoch,
                warm_up_interval=self.anneal_steps,
                anneal=self.use_annealing,
                scheme=self.anneal_scheme,
            )
            log_prob_target = log_prob_target * anneal_factor

        with self.experiment_logger.timer("neg_score", step=epoch):
            result = self.calc_log_q_phi_z(z, epsilon)
            if isinstance(result, tuple):
                log_q_phi_z, score_q = result
            else:
                log_q_phi_z = result
                score_q = None

        with self.experiment_logger.timer("backward", step=epoch):
            loss = -torch.mean(log_prob_target - log_q_phi_z)
            grad_norm = None

            if torch.isfinite(loss):
                self.optimizer_vi.zero_grad()
                loss.backward()
                if self.grad_clip is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.vi_model.parameters(), max_norm=self.grad_clip)
                else:
                    grad_norm = torch.nn.utils.get_total_norm(
                        [p.grad for p in self.vi_model.parameters() if p.grad is not None])
                self.optimizer_vi.step()
                self.scheduler_vi.step()
                if self.ema_enabled:
                    self.ema.update_params(self.vi_model.parameters())
            else:
                logger.warning(
                    f"NaN or Inf detected in VI loss at epoch {epoch}. Skipping update."
                )
                logger.debug(
                    f"Detected {(~torch.isfinite(log_prob_target)).sum()} non-finite values in log_prob_target."
                )
                logger.debug(
                    f"Detected {(~torch.isfinite(log_q_phi_z)).sum()} non-finite values in log_q_phi_z."
                )

        return {
            'loss': loss,
            'grad_norm': grad_norm,
            'z': z,
            'epsilon': epsilon,
            'score_q': score_q,
        }

    @timer("epoch")
    def _run_training_epoch(self, epoch: int) -> None:
        self.curr_epoch = epoch
        self.experiment_logger.set_step(epoch)

        diagnostics = self._compute_loss_and_step(epoch)
        loss = diagnostics['loss']
        z = diagnostics['z']
        epsilon = diagnostics['epsilon']
        grad_norm = diagnostics['grad_norm']
        score_q = diagnostics.get('score_q', None)

        weight_norm = torch.nn.utils.get_total_norm(self.vi_model.parameters())
        z_norm = torch.norm(z, dim=1)
        epsilon_norm = torch.norm(epsilon, dim=1)
        scalar_batch = {
            "train/vi_model/loss": loss.item(),
            "diagnostic/vi_model/weight_norm": weight_norm.item(),
            "diagnostic/vi_model/z_norm_avg": z_norm.mean().item(),
            "diagnostic/vi_model/z_norm_std": z_norm.std().item(),
            "diagnostic/vi_model/epsilon_norm_avg": epsilon_norm.mean().item(),
            "diagnostic/vi_model/epsilon_norm_std": epsilon_norm.std().item(),
        }
        if grad_norm is not None:
            scalar_batch["diagnostic/vi_model/grad_norm"] = grad_norm.item()
        if score_q is not None:
            with torch.no_grad():
                score_conditional = self.vi_model.score(z.detach(), epsilon.detach())
                score_gap = torch.mean(
                    torch.sum((score_q - score_conditional)**2, dim=-1))
            scalar_batch[
                "diagnostic/vi_model/marginal_conditional_score_l2_gap"
            ] = score_gap.item()
        self.experiment_logger.log_scalars(scalar_batch, step=epoch)

        self.training_sample_loss += loss.item()
        if epoch % self.training_loss_log_freq == 0:
            avg_loss = self.training_sample_loss / self.training_loss_log_freq
            logger.debug(f"Epoch {epoch}: Avg Loss: {avg_loss:.4f}")
            self.training_sample_loss = 0.0

        if self.reverse_train:
            with self.experiment_logger.timer("reverse_train", step=epoch):
                self.train_reverse_model(epoch)

        needs_sample = (epoch % self.training_sample_freq == 0)
        needs_metrics = (self.training_metric_log_freq > 0
                         and epoch % self.training_metric_log_freq == 0)
        needs_plot = (epoch % self.plot_freq == 0)

        ema_swapped = False
        if self.ema_enabled and (needs_sample or needs_metrics or needs_plot):
            self.ema.store(self.vi_model.parameters())
            self.ema.apply_shadow(self.vi_model.parameters())
            ema_swapped = True

        try:
            if needs_sample:
                with self.experiment_logger.timer("sampling", step=epoch):
                    self.save_samples(epoch)

            if self.ckpt_enabled and (epoch % self.ckpt_freq == 0):
                with self.experiment_logger.timer("checkpoint", step=epoch):
                    self.save_checkpoint(epoch)

            if needs_metrics:
                with self.experiment_logger.timer("metric_eval_tot", step=epoch):
                    if self.metric_kl_enabled:
                        self.eval_kl_ite(epoch)
                    if self.metric_w2_enabled:
                        self.eval_w2(epoch)
                    if self.metric_elbo_enabled:
                        self.eval_elbo(epoch)
                    if self.metric_expected_log_marginal_enabled:
                        self.eval_expected_log_marginal(epoch)
                    if self.metric_mmd_enabled:
                        self.eval_mmd(epoch)
                    if self.metric_ksd_enabled:
                        self.eval_ksd(epoch)
                    if self.metric_bnn_enabled:
                        self.eval_bnn(epoch)
                    if self.metric_fisher_enabled:
                        self.eval_fisher(epoch)
                    if self.metric_jacobian_spectral_enabled:
                        self.eval_jacobian_spectral(epoch)

            if needs_plot:
                with self.experiment_logger.timer("plot", step=epoch):
                    _, z_plot = self.vi_model.sampling(num=self.plot_num)
                    contour_path = os.path.join(
                        self.plot_save_path, f"contour_epoch_{epoch}.png")
                    trace_path = os.path.join(
                        self.plot_save_path, f"trace_epoch_{epoch}.png")
                    try:
                        self.target_model.contour_plot(
                            self.config.target.bbox,
                            fnet=None,
                            samples=z_plot.cpu().numpy(),
                            save_to_path=contour_path,
                            quiver=False,
                            t=epoch,
                        )
                        plot_path = contour_path
                        logger.debug(f"Saved contour plot at epoch {epoch}.")
                    except (NotImplementedError, AttributeError, KeyError):
                        self.target_model.trace_plot(
                            z_plot,
                            figpath=self.plot_save_path,
                            figname=f"trace_epoch_{epoch}.png",
                            figtitle=f"Trace Plot at Epoch {epoch}",
                        )
                        plot_path = trace_path
                        logger.debug(f"Saved trace plot at epoch {epoch}.")
                    self.experiment_logger.log_image(
                        "plots/posterior", plot_path, step=epoch)
        finally:
            if ema_swapped:
                self.ema.restore(self.vi_model.parameters())

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
        elif not self._vi_pretrained_done:
            self.pretrain_vi()

        # Main training loop
        self.vi_model.train()

        self.train_start_time = time.perf_counter()

        for epoch in tqdm(
                range(self.train_start_epoch, self.training_num_epochs + 1),
                desc="Main Training",
                initial=self.train_start_epoch - 1,
                total=self.training_num_epochs,
        ):
            self._run_training_epoch(epoch)

        # Close writer at end
        total_time = time.perf_counter() - self.train_start_time
        avg_epoch_time = total_time / max(1, self.training_num_epochs)
        logger.info(
            f"Training completed. Total time: {total_time:.3f}s, Avg epoch time: {avg_epoch_time:.6f}s"
        )
        self.experiment_logger.log_scalars(
            {
                "summary/total_training_time": total_time,
                "summary/avg_epoch_time": avg_epoch_time,
            },
            step=self.training_num_epochs,
        )
