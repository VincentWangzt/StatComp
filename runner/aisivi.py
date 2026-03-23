import torch
from omegaconf import DictConfig
from runner.base_reverse_runner import BaseReverseConditionalRunner
from utils.metrics import compute_ksd
from utils.kernels import GaussianKernel
from utils.logging import get_logger

logger = get_logger()


class AISIVIRunner(BaseReverseConditionalRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "AISIVI",
    ):
        super().__init__(config=config, name=name)

    def calculate_rev_KSD(self) -> tuple[float, float]:
        '''
        Calculate the Kernelized Stein Discrepancy (KSD) using the reverse denoising model.

        Returns:
            (ksd, h) (float, float): Estimated KSD value and kernel bandwidth.
        '''
        with torch.no_grad():
            _, z_samples = self.vi_model.sampling(num=self.n_ksd_samples)
            self.reverse_ksd_kernel = GaussianKernel()
            h = self.reverse_ksd_kernel.fit_h(z_samples)
            z_aux, epsilon_aux, log_q_psi_epsilon_given_z = self.reverse_model.sample(
                z_samples,
                num_samples=self.training_reverse_sample_num,
            )
            importance_sampling_weights = self.vi_model.log_q_epsilon(
                epsilon_aux) - log_q_psi_epsilon_given_z
            importance_sampling_weights = importance_sampling_weights.detach()
        z_aux.requires_grad_(True)
        log_q_phi_z_aux = self.vi_model.logp(
            z_aux, epsilon_aux) + importance_sampling_weights
        log_q_phi_z = torch.logsumexp(
            log_q_phi_z_aux,
            dim=1,
        ) - torch.log(
            torch.tensor(
                self.training_reverse_sample_num,
                device=z_samples.device,
                dtype=z_samples.dtype,
            ))
        score = torch.autograd.grad(
            log_q_phi_z.sum(),
            z_aux,
            create_graph=False,
        )[0]
        score = score.sum(dim=1)
        # shape (batch_size, z_dim)
        score = score.clone().detach()
        if self.normalize_reverse_score:
            score = score - score.mean(dim=0, keepdim=True)

        ksd = compute_ksd(
            x=z_samples,
            scores=score,
            kernel=self.reverse_ksd_kernel,
        )
        return ksd, h

    def eval_ksd(self, epoch: int):
        super().eval_ksd(epoch)
        rev_ksd, rev_h = self.calculate_rev_KSD()
        self.writer.add_scalar("metric/reverse_model/ksd", rev_ksd, epoch)
        self.writer.add_scalar("metric/reverse_model/ksd_h", rev_h, epoch)

    def calc_log_q_phi_z(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Estimate log q_phi(z) via the gradient.
        ```
        nabla_z log q_phi(z) ~ nabla_z log E_{epsilon^prime ~ q_psi(epsilon|z)} [q_phi(z|epsilon^prime) * q(epsilon^prime) / q_psi(epsilon^prime|z)]
        ```

        Args:
            z (torch.Tensor): Samples from q_phi(z|epsilon), shape (batch_size, z_dim).
            epsilon (torch.Tensor): Corresponding epsilon samples, shape (batch_size, epsilon_dim).
        
        Returns:
            log_q_phi_z (torch.Tensor): Estimated log q_phi(z), shape (batch_size,).
        '''
        with torch.no_grad():
            self.reverse_model.eval()
            z_aux, epsilon_aux, log_q_psi_epsilon_given_z = self.reverse_model.sample(
                z,
                num_samples=self.training_reverse_sample_num,
            )

            log_q_epsilon = self.vi_model.log_q_epsilon(epsilon_aux)

            if torch.isnan(log_q_epsilon).any():
                logger.debug(
                    f"{torch.isnan(log_q_epsilon).sum()} NaN detected in log_q_epsilon."
                )
            if torch.isnan(log_q_psi_epsilon_given_z).any():
                logger.debug(
                    f"{torch.isnan(log_q_psi_epsilon_given_z).sum()} NaN detected in log_q_psi_epsilon_given_z."
                )

            importance_sampling_weights = log_q_epsilon - log_q_psi_epsilon_given_z

            importance_sampling_weights = importance_sampling_weights.detach()

        if torch.isnan(importance_sampling_weights).any():
            logger.debug(
                f"{torch.isnan(importance_sampling_weights).sum()} NaN detected in importance sampling weights."
            )

        z_aux.requires_grad_(True)

        # shape (batch_size, training_reverse_sample_num)
        log_q_phi_z_aux = self.vi_model.logp(
            z_aux, epsilon_aux) + importance_sampling_weights

        # shape (batch_size,)
        log_q_phi_z = torch.logsumexp(
            log_q_phi_z_aux,
            dim=1,
        ) - torch.log(
            torch.tensor(
                self.training_reverse_sample_num,
                device=z.device,
                dtype=z.dtype,
            ))

        # shape (batch_size, training_reverse_sample_num, z_dim)
        score = torch.autograd.grad(
            log_q_phi_z.sum(),
            z_aux,
            create_graph=False,
        )[0]

        # shape (batch_size, z_dim)
        score = score.sum(dim=1)
        score = score.clone().detach()

        if self.normalize_reverse_score:
            score = score - score.mean(dim=0, keepdim=True)

        with torch.no_grad():
            # Log the average distance from epsilon_aux to original epsilon
            avg_eps_distance = torch.mean(
                torch.norm(
                    epsilon_aux - epsilon.unsqueeze(1),
                    dim=-1,
                )).item()
            self.writer.add_scalar(
                "diagnostic/reverse_model/avg_epsilon_distance",
                avg_eps_distance,
                self.curr_epoch,
            )
            # Log the average norm of the score function
            avg_score_norm = torch.mean(torch.norm(score, dim=-1)).item()
            self.writer.add_scalar(
                "diagnostic/reverse_model/avg_score_norm",
                avg_score_norm,
                self.curr_epoch,
            )

            # Log the norm of the average of the score function
            avg_of_score_norm = torch.norm(score.mean(dim=0)).item()
            self.writer.add_scalar(
                "diagnostic/reverse_model/norm_of_avg_score",
                avg_of_score_norm,
                self.curr_epoch,
            )

        return torch.sum(score * z, dim=-1)  # shape (batch_size,)
