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

# 1207 TODO:
# 1. Refine and add comments [Done]
# 2. Add KL to baseline as a metric
# 3. Add environment.yml or requirements.txt [Done]
# 4. Revise the loss implementation
# 5. Saving and loading checkpoints


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
    parser = argparse.ArgumentParser(description="Reverse UIVI Runner")
    default_cfg = os.path.join(os.path.dirname(__file__), 'configs',
                               'reverse_uivi.yaml')
    parser.add_argument('--config',
                        type=str,
                        default=default_cfg,
                        help='Path to YAML config file')
    return parser.parse_args()


# Parse CLI and load configuration
args = parse_args()
CONFIG_PATH = args.config
cfg = load_config(CONFIG_PATH)
# Setup logging (console + optional file under save path)
logger = get_logger("reverse_uivi")

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

    def __init__(self, device, cfg):
        self.target_dist = cfg['experiment']['target_dist']
        self.device = device
        self.cfg = cfg

        # target
        self.target_model = target_distribution[self.target_dist](
            device=device)

        # vi model from config
        model_config = cfg['models']
        assert model_config['vi_model_type'] == 'ConditionalGaussian', \
            "Only ConditionalGaussian VI model is supported in Reverse UIVI."
        assert model_config['reverse_model_type'] == 'ConditionalRealNVP', \
            "Only ConditionalRealNVP reverse model is supported in Reverse UIVI."
        self.epsilon_dim = model_config['epsilon_dim']
        self.z_dim = model_config['z_dim']

        self.vi_model = ConditionalGaussian(
            epsilon_dim=model_config['epsilon_dim'],
            hidden_dim=model_config['vi_hidden_dim'],
            z_dim=model_config['z_dim'],
            device=self.device,
        ).to(self.device)

        # reverse model from config
        self.reverse_model = ConditionalRealNVP(
            z_dim=model_config['z_dim'],
            epsilon_dim=model_config['epsilon_dim'],
            hidden_dim=model_config['reverse_hidden_dim'],
            num_layers=model_config['reverse_num_layers'],
            device=self.device,
        ).to(self.device)

        # save path
        self.exp_name = cfg['experiment']['name']
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

    def calculate_KL(self) -> float:
        '''
        Calculate the KL divergence between the true joint distribution and the
        joint distribution induced by the reverse model using the ITE package.
        Returns:
            kl_div (float): Estimated KL divergence value.
        '''
        n_ite_samples = self.cfg['kl']['n_ite_samples']

        # Sample from true joint
        true_eps, true_z = self.vi_model.sampling(num=n_ite_samples)
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
        wcfg = self.cfg['warmup']
        if not wcfg['enabled']:
            return
        lr = wcfg['lr']
        batch_size = wcfg['batch_size']
        epochs = wcfg['epochs']
        kl_freq = wcfg['kl_log_freq']
        loss_log_freq = wcfg['loss_log_freq']
        sample_loss = 0

        optimizer = torch.optim.Adam(self.reverse_model.parameters(), lr=lr)

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
                sample_loss = 0
                logger.debug(
                    f"Warmup Epoch {epoch}, Average Reverse Model Loss: {avg_loss:.4f}"
                )

        self.train_reverse_model(
            optimizer,
            epochs,
            batch_size,
            progress_bar=True,
            log_func=log_func,
        )

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
        
        Returns:
            None
        '''
        # initial warmup
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

        # Reverse training cfg
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
        reverse_log_freq = log_cfg['reverse_log_freq']
        sample_loss = 0
        sample_reverse_loss = 0

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

        for epoch in tqdm(range(1, num_epochs + 1), desc="Main Training"):

            # Sample epsilon
            epsilon = torch.randn(batch_size,
                                  self.vi_model.epsilon_dim).to(self.device)

            # Sample z from variational distribution
            z, neg_score_implicit = self.vi_model.forward(epsilon)

            # Compute log prob under target distribution
            log_prob_target = self.target_model.logp(z)

            with torch.no_grad():
                self.reverse_model.eval()
                z_aux, epsilon_aux = self.reverse_model.sample(
                    z,
                    num_samples=reverse_sample_num,
                )
                neg_score = self.vi_model.neg_score(z_aux, epsilon_aux)
                neg_score = neg_score.mean(dim=1)
                neg_score = neg_score.clone().detach()

            # Compute loss
            loss = -torch.mean(log_prob_target + torch.sum(
                z * neg_score,
                dim=-1,
            ))

            optimizer_vi.zero_grad()
            loss.backward()
            optimizer_vi.step()

            # TensorBoard scalars
            self.writer.add_scalar("train/loss", loss, epoch)
            scheduler_vi.step()
            sample_loss += loss.item()

            if epoch % loss_log_freq == 0:
                avg_loss = sample_loss / loss_log_freq
                logger.debug(f"Epoch {epoch}, VI Model Loss: {avg_loss:.4f}")
                sample_loss = 0

            # Train reverse model
            if (epoch % rev_update_freq) == 0:
                if not rev_reuse_optimizer:
                    reverse_optimizer = torch.optim.Adam(
                        self.reverse_model.parameters(), lr=reverse_lr)
                # logger.debug(
                #     f"Training reverse model at epoch {epoch} for {rev_epochs} epochs..."
                # )
                self.train_reverse_model(
                    reverse_optimizer,
                    rev_epochs,
                    rev_batch_size,
                    progress_bar=False,
                    log_func=lambda l, e: rev_log_func(l, e, epoch),
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

            # Log KL divergence
            if epoch % kl_log_freq == 0:
                kl_div = self.calculate_KL()
                logger.debug(f"Epoch {epoch}, KL Divergence: {kl_div:.4f}")
                self.writer.add_scalar("train/kl_div", kl_div, epoch)

            # Generate and save contour plots
            if epoch % plot_freq == 0:
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

        # Close writer at end
        self.writer.close()
        logger.info("Training completed.")


if __name__ == "__main__":
    reverse_uivi = ReverseUIVI(device=device, cfg=cfg)
    reverse_uivi.learn()
