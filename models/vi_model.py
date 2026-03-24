import torch
import torch.nn as nn
import math
import torch.distributions as dist
from utils.logging import get_logger
from omegaconf.dictconfig import DictConfig

logger = get_logger()


class BaseVIModel(nn.Module):
    """
    Base class for variational inference models q_phi(z|epsilon).

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
    Args:
        config (DictConfig): Configuration object.
        name (str): Name of the model.
    """

    def __init__(
        self,
        config: DictConfig,
        name: str = '',
    ):
        super().__init__()
        assert name != '', "Please provide a name for the model."
        self.name = name
        self.config: DictConfig = config
        self.epsilon_dim: int = config.epsilon_dim
        self.z_dim: int = config.z_dim
        self.device: torch.device = torch.device(config.device)

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from prior.

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        raise NotImplementedError

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing `z ~ q_phi(z|epsilon)` and its negative score term.

        Args:
            epsilon (torch.Tensor): Input noise `epsilon`.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score.
        """
        raise NotImplementedError

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the variational model. No gradients computed.

        Args:
            num (int): Number of samples.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        raise NotImplementedError

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the score term.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            score (torch.Tensor): Score with shape `[..., D_z]`.
        """
        raise NotImplementedError

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_phi(z|epsilon)` for given `z` and `epsilon`.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            log_prob (torch.Tensor): Log probability `log q_phi(z|epsilon)`.
        """
        raise NotImplementedError

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) of prior.

        Args: 
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        raise NotImplementedError


class ConditionalGaussian(BaseVIModel):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP.

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalGaussian")
        self.hidden_dim: int = config.hidden_dim
        self.num_layers: int = config.num_layers
        self.out_dim = self.z_dim * 2
        self.var_min = 1e-4
        # The network outputs both mean and variance
        layers = []
        input_dim = self.epsilon_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            input_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        self.net = nn.Sequential(*layers)

    def reparameterize(
        self,
        mu: torch.Tensor,
        var_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for `q_phi(z|epsilon)`.

        Args:
            mu (torch.Tensor): Mean.
            var_raw (torch.Tensor): Raw variance. Will be reparameterized by softplus.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std` where
            `z = mu + std * u` and `u ~ N(0, I)`.
        """
        var = torch.nn.functional.softplus(var_raw)
        var = var.clamp(min=self.var_min)
        std = torch.sqrt(var)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `mu(epsilon)` from the network output split."""
        return self.net(epsilon).chunk(2, dim=-1)[0]

    def getstd(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `std(epsilon)` by clamping variance and taking square root."""
        var_raw = self.net(epsilon).chunk(2, dim=-1)[1]
        var = torch.nn.functional.softplus(var_raw)
        var = var.clamp(min=self.var_min)
        std = torch.sqrt(var)
        return std

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing `z ~ q_phi(z|epsilon)` and its negative score term.

        Args:
            epsilon (torch.Tensor): Input noise `epsilon`.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std`.
        """
        mu, var_raw = self.net(epsilon).chunk(2, dim=-1)
        z, neg_score_implicit = self.reparameterize(mu, var_raw)
        return z, neg_score_implicit

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from standard normal.

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.randn([num, self.epsilon_dim], ).to(self.device)

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the conditional Gaussian. No gradients computed.

        Args:
            num (int): Number of samples.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        with torch.no_grad():
            epsilon = self.sample_epsilon(num=num)
            Z, _ = self.forward(epsilon)
        return epsilon.clone().detach(), Z.clone().detach()

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the score term `-(z - mu) / var` used in objectives.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            score (torch.Tensor): Score `-(z - mu(epsilon)) / var(epsilon)`.
        """
        mu, var_raw = self.net(epsilon).chunk(2, dim=-1)
        var = torch.nn.functional.softplus(var_raw)
        var = var.clamp(min=self.var_min)
        score = -(z - mu) / (var)
        return score

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_phi(z|epsilon)` for given `z` and `epsilon`.
        Supports broadcasting over leading dimensions.
        Args:
            z (torch.Tensor): shape [..., Dz]
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_prob (torch.Tensor): shape [...]
        """
        mu, var_raw = self.net(epsilon).chunk(2, dim=-1)
        var = torch.nn.functional.softplus(var_raw)
        var = var.clamp(min=self.var_min)
        # Gaussian log-likelihood per sample
        const = -0.5 * z.shape[-1] * math.log(2 * math.pi)
        ll = const - 0.5 * (var.log().sum(dim=-1) +
                            ((z - mu)**2 / var).sum(dim=-1))
        return ll

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) of prior.

        Args: 
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        const = -0.5 * epsilon.shape[-1] * math.log(2 * math.pi)
        return const - 0.5 * (epsilon**2).sum(dim=-1)


class ConditionalGaussianUniform(ConditionalGaussian):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP. The prior on epsilon is `U[0,1]`.

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.
        - device: Device for computation.
    
    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config)
        self.name = "ConditionalGaussianUniform"

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from uniform [0,1].

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.rand([num, self.epsilon_dim], ).to(self.device)

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) under uniform [0,1] prior.

        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        # 1. Check if ALL elements in the last dimension are within [0, 1]
        # This results in a boolean tensor of shape [...]
        in_bounds = (epsilon >= 0) & (epsilon <= 1)
        all_in_bounds = in_bounds.all(dim=-1)

        # 2. Initialize log_prob with zeros (log(1) = 0)
        log_prob = torch.zeros_like(
            all_in_bounds,
            device=self.device,
            dtype=epsilon.dtype,
        )

        # 3. Set out-of-bounds entries to -inf
        log_prob[~all_in_bounds] = float('-inf')

        return log_prob


class ConditionalGaussianGlobalUniform(BaseVIModel):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP.

    Required config parameters:
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.

    Args:
        config (DictConfig): Configuration object.
    """

    def __init__(
        self,
        config: DictConfig,
    ):
        super().__init__(config, name="ConditionalGaussianGlobalUniform")
        self.hidden_dim: int = config.hidden_dim
        self.num_layers: int = config.num_layers
        self.out_dim = self.z_dim
        self.var_min = 1e-4
        # The network outputs both mean and variance, with variance being global
        layers = []
        input_dim = self.epsilon_dim
        for _ in range(self.num_layers):
            layers.append(nn.Linear(input_dim, self.hidden_dim))
            layers.append(nn.SiLU())
            input_dim = self.hidden_dim
        layers.append(nn.Linear(self.hidden_dim, self.out_dim))
        self.net = nn.Sequential(*layers)
        self.var_raw = nn.Parameter(torch.ones((self.z_dim, )))

    def reparameterize(
        self,
        mu: torch.Tensor,
        var_raw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for `q_phi(z|epsilon)`.

        Args:
            mu (torch.Tensor): Mean.
            var_raw (torch.Tensor): Raw variance. Will be reparameterized by softplus.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std` where
            `z = mu + std * u` and `u ~ N(0, I)`.
        """
        var = torch.nn.functional.softplus(self.var_raw)
        var = var.clamp(min=self.var_min)
        std = torch.sqrt(var)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `mu(epsilon)` from the network output split."""
        return self.net(epsilon)

    def getstd(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `std(epsilon)` by clamping variance and taking square root."""
        var = torch.nn.functional.softplus(self.var_raw)
        var = var.clamp(min=self.var_min)
        std = torch.sqrt(var)
        return std

    def forward(
        self,
        epsilon: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass producing `z ~ q_phi(z|epsilon)` and its negative score term.

        Args:
            epsilon (torch.Tensor): Input noise `epsilon`.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std`.
        """
        mu = self.net(epsilon)
        z, neg_score_implicit = self.reparameterize(mu, self.var_raw)
        return z, neg_score_implicit

    def sampling(
        self,
        num: int = 1000,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the conditional Gaussian. No gradients computed.

        Args:
            num (int): Number of samples.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        with torch.no_grad():
            epsilon = self.sample_epsilon(num=num)
            Z, _ = self.forward(epsilon)
        return epsilon.clone().detach(), Z.clone().detach()

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the score term `-(z - mu) / var` used in objectives.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            score (torch.Tensor): Score `-(z - mu(epsilon)) / var(epsilon)`.
        """
        mu = self.net(epsilon)
        var = torch.nn.functional.softplus(self.var_raw)
        var = var.clamp(min=self.var_min)
        score = -(z - mu) / (var)
        return score

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log probability `log q_phi(z|epsilon)` for given `z` and `epsilon`.
        Supports broadcasting over leading dimensions.
        Args:
            z (torch.Tensor): shape [..., Dz]
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_prob (torch.Tensor): shape [...]
        """
        mu = self.net(epsilon)
        var = torch.nn.functional.softplus(self.var_raw)
        var = var.clamp(min=self.var_min)
        # Gaussian log-likelihood per sample
        const = -0.5 * z.shape[-1] * math.log(2 * math.pi)
        ll = const - 0.5 * (var.log().sum(dim=-1) +
                            ((z - mu)**2 / var).sum(dim=-1))
        return ll

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from uniform [0,1].

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.rand([num, self.epsilon_dim], ).to(self.device)

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) under uniform [0,1] prior.

        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        # 1. Check if ALL elements in the last dimension are within [0, 1]
        # This results in a boolean tensor of shape [...]
        in_bounds = (epsilon >= 0) & (epsilon <= 1)
        all_in_bounds = in_bounds.all(dim=-1)

        # 2. Initialize log_prob with zeros (log(1) = 0)
        log_prob = torch.zeros_like(
            all_in_bounds,
            device=self.device,
            dtype=epsilon.dtype,
        )

        # 3. Set out-of-bounds entries to -inf
        log_prob[~all_in_bounds] = float('-inf')

        return log_prob


class ConditionalGaussianGlobal(ConditionalGaussianGlobalUniform):
    """
    Conditional Gaussian `q_phi(z|epsilon)` with global learnable variance
    and standard Normal prior on epsilon.

    This corresponds to KSIVI's SIMINet architecture: an MLP maps epsilon to
    mu(epsilon), with a shared (global) learnable variance parameter, and
    epsilon is sampled from N(0, I).

    Inherits all architecture and methods from ConditionalGaussianGlobalUniform,
    only overriding the prior distribution on epsilon.

    Required config parameters (same as parent):
        - epsilon_dim: Dimension of epsilon.
        - z_dim: Dimension of latent z.
        - device: Device for computation.
        - hidden_dim: Hidden size of the MLP.
        - num_layers: Number of hidden layers.
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.name = "ConditionalGaussianGlobal"

    def sample_epsilon(
        self,
        num: int = 1000,
    ) -> torch.Tensor:
        """
        Sample `num` epsilon from standard Normal N(0, I).

        Args:
            num (int): Number of samples.
        Returns:
            epsilon (torch.Tensor): Samples of shape `[num, D_epsilon]`.
        """
        return torch.randn([num, self.epsilon_dim]).to(self.device)

    def log_q_epsilon(
        self,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute log q(epsilon) under standard Normal N(0, I) prior.

        Args:
            epsilon (torch.Tensor): shape [..., De]
        Returns:
            log_q (torch.Tensor): shape [...]
        """
        const = -0.5 * epsilon.shape[-1] * math.log(2 * math.pi)
        return const - 0.5 * (epsilon**2).sum(dim=-1)


VIModel: dict[str, type[BaseVIModel]] = {
    "ConditionalGaussian": ConditionalGaussian,
    "ConditionalGaussianUniform": ConditionalGaussianUniform,
    "ConditionalGaussianGlobalUniform": ConditionalGaussianGlobalUniform,
    "ConditionalGaussianGlobal": ConditionalGaussianGlobal,
}
