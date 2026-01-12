import torch
from omegaconf import DictConfig
from runner.base_reverse_runner import BaseReverseConditionalRunner
from runner.base_runner import BaseSIVIRunner
from models.denoise_model import BaseDenoiseModel, DenoiseModel
from typing import Callable
from utils.logging import get_logger
from tqdm import tqdm
import time

logger = get_logger()


class DSIVIRunner(BaseReverseConditionalRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "DSIVI",
    ):
        super().__init__(config=config, name=name)
        self.reverse_model: BaseDenoiseModel

    def _get_reverse_model(self) -> BaseDenoiseModel:
        '''
        Construct the reverse model based on the specified type and configuration.

        Returns:
            reverse_model (BaseReverseConditionalModel): Instantiated reverse model.
        '''
        if self.reverse_model_type not in DenoiseModel:
            raise ValueError(
                f"Unsupported reverse model type: {self.reverse_model_type}")
        reverse_model = DenoiseModel[self.reverse_model_type](
            config=self.config.reverse_model)
        return reverse_model

    def eval_kl_ite(self, epoch: int):
        """
        Evaluate KL divergence between VI and baseline using ITE and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        """
        BaseSIVIRunner.eval_kl_ite(self, epoch)

    def eval_w2(self, epoch: int):
        """
        Evaluate Wasserstein-2 distance between VI and baseline and log to TensorBoard.
        Args:
            epoch (int): Current epoch number.
        """
        BaseSIVIRunner.eval_w2(self, epoch)

    def _train_reverse_model(
        self,
        optimizer: torch.optim.Optimizer | None,
        epochs: int,
        batch_size: int,
        initialize: bool = False,
        progress_bar: bool = True,
        log_func: Callable[[float, int, int], None] | None = None,
    ) -> None:
        '''
        Train the reverse model. Generate samples from the VI model and then optimize the reverse model to maximize the log-reverse-probablity of these samples.
        
        Args:
            optimizer (torch.optim.Optimizer | None): Optimizer for the reverse model. If None, the reverse model's own fit() method will be used.
            epochs (int): Number of training epochs.
            batch_size (int): Batch size for training.
            initialize (bool, optional): Whether to re-initialize parameters before fitting. Defaults to False.
            progress_bar (bool, optional): Whether to display a progress bar. Defaults to True.
            log_func (Callable[[float, int, int], None], optional): Function to log training progress. Defaults to None. The first argument is the loss, the second is the steps, and the third is the epoch.
        Returns:
            None
        '''
        iterator = range(epochs)
        if progress_bar:
            iterator = tqdm(iterator, desc="Reverse Model Training")
        # Two modes:
        # - nn.Module-based reverse model: use optimizer to train.
        # - Statistic-based reverse model: use its fit() method.
        if optimizer is not None:
            self.reverse_model.train()
            self.vi_model.eval()
            for epoch in iterator:
                with torch.no_grad():
                    epsilon_samples, z_samples = self.vi_model.sampling(
                        num=batch_size)
                    score = self.vi_model.score(z_samples, epsilon_samples)
                optimizer.zero_grad()
                score_pred = self.reverse_model.score(z_samples)
                loss = torch.mean((score_pred - score)**2)
                if torch.isfinite(loss):
                    loss.backward()
                    optimizer.step()
                else:
                    logger.warning(
                        f"NaN or Inf detected in reverse model loss at epoch {self.curr_epoch}. Skipping update."
                    )
                if log_func is not None:
                    log_func(loss.item(), 1, epoch)
            self.vi_model.train()
        else:
            epsilon_samples, z_samples = self.vi_model.sampling(num=batch_size)
            score = self.vi_model.score(z_samples, epsilon_samples)
            mse, steps = self.reverse_model.fit(
                z_samples,
                score,
                initialize=initialize,
            )
            if log_func is not None:
                log_func(mse, steps, epochs)

    def _warmup_log_func(self, loss: float, steps: int, epoch: int) -> None:
        '''
        Logging hook for reverse model warmup.
        Args:
            loss (float): Loss value for the current epoch.
            steps (int): Number of steps taken in the current epoch.
            epoch (int): Current epoch number.
        '''
        self.writer.add_scalar("warmup/reverse_model_loss", loss, epoch)
        self.warmup_sample_loss += loss
        self.warmup_steps += steps

        if epoch % self.warmup_loss_log_freq == 0:
            avg_loss = self.warmup_sample_loss / self.warmup_loss_log_freq
            current_time = time.perf_counter()
            avg_step_time = (current_time -
                             self.warmup_last_time) / self.warmup_loss_log_freq
            self.warmup_last_time = current_time
            self.warmup_sample_loss = 0.0
            avg_steps = self.warmup_steps / self.warmup_loss_log_freq
            logger.debug(
                f"Warmup Epoch {epoch}, Average Reverse Model Loss: {avg_loss:.4f}, Avg Step Time: {avg_step_time:.4f}s, Avg Steps: {avg_steps:.4f}"
            )

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate log q_phi(z|epsilon) using the reverse denoising model.

        Args:
            z (torch.Tensor): Samples of z with shape [Batch, Dz].
            epsilon (torch.Tensor): Samples of epsilon with shape [Batch, De].

        Returns:
            log_q_phi (torch.Tensor): Log probability log q_phi(z|epsilon) with shape [Batch].
        """
        with torch.no_grad():
            score = self.reverse_model.score(z).clone().detach()

        if self.normalize_reverse_score:
            score = score - score.mean(dim=0, keepdim=True)

        with torch.no_grad():
            # Log the average norm of the score function
            avg_score_norm = torch.mean(torch.norm(score, dim=-1)).item()
            self.writer.add_scalar(
                "norm/avg_score_norm",
                avg_score_norm,
                self.curr_epoch,
            )

            # Log the norm of the average of the score function
            avg_of_score_norm = torch.norm(score.mean(dim=0)).item()
            self.writer.add_scalar(
                "norm/norm_of_avg_score",
                avg_of_score_norm,
                self.curr_epoch,
            )

        return torch.sum(score * z, dim=-1)
