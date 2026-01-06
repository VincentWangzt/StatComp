import torch
from omegaconf import DictConfig
from runner.base_reverse_runner import BaseReverseConditionalRunner


class AISIVIRunner(BaseReverseConditionalRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "AISIVI",
    ):
        super().__init__(config=config, name=name)

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
            z_aux, epsilon_aux = self.reverse_model.sample(
                z,
                num_samples=self.training_reverse_sample_num,
            )

            importance_sampling_weights = self.vi_model.log_q_epsilon(
                epsilon_aux) - self.reverse_model.log_prob(epsilon_aux, z_aux)

            importance_sampling_weights = importance_sampling_weights.detach()

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

        with torch.no_grad():
            # Log the average distance from epsilon_aux to original epsilon
            avg_eps_distance = torch.mean(
                torch.norm(
                    epsilon_aux - epsilon.unsqueeze(1),
                    dim=-1,
                )).item()
            self.writer.add_scalar(
                "train/avg_epsilon_distance",
                avg_eps_distance,
                self.curr_epoch,
            )

        return torch.sum(score * z, dim=-1)  # shape (batch_size,)
