import torch
from omegaconf import DictConfig
from runner.base_reverse_runner import BaseReverseConditionalRunner


class RSIVIRunner(BaseReverseConditionalRunner):

    def __init__(
        self,
        config: DictConfig,
        name: str = "RSIVI",
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
        nabla_z log q_phi(z) ~ E_{epsilon^prime ~ q_psi(epsilon|z)} [nabla_z log q_phi(z|epsilon^prime)]
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

        with torch.no_grad():
            score = self.vi_model.score(z_aux, epsilon_aux)
            score = score.mean(dim=1)
            score = score.clone().detach()

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
