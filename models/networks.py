import torch
import torch.nn as nn
from argparse import Namespace


class SIMINet(nn.Module):
    """
    Semi Implicit Model for conditional Gaussian outputs.

    The network maps latent `Z` to a mean vector `mu(Z)` and uses a learned
    global log-variance parameter to define a Gaussian. Sampling is
    performed via the reparameterization trick returning both the sample and
    the corresponding negative score term.

    Args:
        namedict (Namespace): Configuration with attributes:
            - z_dim (int): Dimension of latent `Z`.
            - h_dim (int): Hidden layer size.
            - out_dim (int): Output dimension.
            - log_var_ini (float): Initial value for log-variance.
            - log_var_min (float): Minimum clamp value for log-variance.
        device (torch.device): Device for computation.
    """

    def __init__(
        self,
        namedict: Namespace,
        device: torch.device,
    ):
        super(SIMINet, self).__init__()
        self.z_dim = namedict.z_dim
        self.h_dim = namedict.h_dim
        self.out_dim = namedict.out_dim

        self.mu = nn.Sequential(
            nn.Linear(self.z_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.h_dim),
            nn.ReLU(),
            nn.Linear(self.h_dim, self.out_dim),
        )
        self.log_var = nn.Parameter(
            torch.zeros(namedict.out_dim) + namedict.log_var_ini,
            requires_grad=True,
        )
        self.device = device
        self.log_var_min = namedict.log_var_min

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for Gaussian sampling.

        Given mean `mu` and log-variance `log_var`, sample
        `X = mu + exp(log_var/2) * eps` with `eps ~ N(0, I)`, and return the
        negative score term `eps / std` which is used in certain objectives.

        Args:
            mu (torch.Tensor): Mean of the Gaussian.
            log_var (torch.Tensor): Log-variance of the Gaussian.
        Returns:
            (X, neg_score) (torch.Tensor, torch.Tensor): Sample `X` and negative score `eps/std`.
        """
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(mu)
        return mu + std * eps, eps / std

    def getmu(self, Z: torch.Tensor) -> torch.Tensor:
        """Return mean `mu(Z)` for input latent `Z`."""
        return self.mu(Z)

    def getstd(self) -> torch.Tensor:
        """Return standard deviation derived from the clamped log-variance."""
        log_var = self.log_var.clamp(min=self.log_var_min)
        std = torch.exp(log_var / 2)
        return std

    def forward(self, Z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: produce a sample and its negative score term.

        Args:
            Z (torch.Tensor): Latent input.
        Returns:
            (X, neg_score) (torch.Tensor, torch.Tensor): Sample `X` and negative score `eps/std`.
        """
        mu = self.mu(Z)
        log_var = self.log_var.clamp(min=self.log_var_min)
        X, neg_score_implicit = self.reparameterize(mu, log_var)
        return X, neg_score_implicit

    def sampling(self, num: int = 1000, sigma: float = 1) -> torch.Tensor:
        """
        Sample `num` outputs of X by drawing `Z ~ N(0, sigma^2)`. No gradient is computed.

        Args:
            num (int): Number of samples.
            sigma (float): Scale applied to latent `Z`.
        Returns:
            torch.Tensor: Samples `X`.
        """
        with torch.no_grad():
            Z = torch.randn([num, self.z_dim], ).to(self.device)
            Z = Z * sigma
            X, _ = self.forward(Z)
        return X


class ConditionalRealNVP(nn.Module):
    """
    RealNVP for conditional density estimation. Estimates q_psi(epsilon|z) via normalizing flow.

    Args:
        z_dim (int): Dimension of `z`.
        epsilon_dim (int): Dimension of `epsilon`.
        hidden_dim (int): Hidden size for scale/translate nets.
        num_layers (int): Number of coupling layers.
        device (torch.device): Device for computation.
    """

    def __init__(
            self,
            z_dim: int,
            epsilon_dim: int,
            hidden_dim: int = 128,
            num_layers: int = 4,
            device: torch.device = torch.device("cpu"),
    ):
        super().__init__()
        self.epsilon_dim = epsilon_dim
        self.device = device

        # Scaling and translation networks for each coupling layer
        self.scale_nets = nn.ModuleList()
        self.translate_nets = nn.ModuleList()

        for _ in range(num_layers):
            # Each coupling layer conditions on z and part of epsilon
            self.scale_nets.append(
                nn.Sequential(
                    nn.Linear(
                        z_dim + epsilon_dim // 2,
                        hidden_dim,
                    ), nn.SiLU(), nn.Linear(
                        hidden_dim,
                        hidden_dim,
                    ), nn.SiLU(), nn.Linear(
                        hidden_dim,
                        epsilon_dim // 2,
                    )))
            self.translate_nets.append(
                nn.Sequential(
                    nn.Linear(
                        z_dim + epsilon_dim // 2,
                        hidden_dim,
                    ), nn.SiLU(), nn.Linear(
                        hidden_dim,
                        hidden_dim,
                    ), nn.SiLU(), nn.Linear(
                        hidden_dim,
                        epsilon_dim // 2,
                    )))

    def forward(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass: map `epsilon` to base variable `u` and compute `log q_psi(epsilon|z)`.

        Args:
            epsilon (torch.Tensor): Input `epsilon` of shape `[..., D_epsilon]`.
            z (torch.Tensor): Conditioning variable of shape `[..., D_z]`.
        Returns:
            (u, log_prob) (torch.Tensor, torch.Tensor): Base variable `u` and `log q_psi(epsilon|z)`.
        """
        batch_size = epsilon.shape[0]
        log_det = torch.zeros(batch_size).to(self.device)

        # Forward: epsilon -> u (base distribution)
        u = epsilon.clone()
        for i in range(len(self.scale_nets)):
            # Split
            u1, u2 = u.chunk(2, dim=-1)

            # Compute scale and translation
            # scale = torch.tanh(self.scale_nets[i](torch.cat([u1, z], dim=-1)))
            scale = self.scale_nets[i](torch.cat([u1, z], dim=-1))
            translate = self.translate_nets[i](torch.cat([u1, z], dim=-1))
            # Transform
            u2 = u2 * torch.exp(scale) + translate
            log_det += scale.sum(dim=-1)

            # Concatenate (alternate split)
            u = torch.cat([u2, u1], dim=-1) if i % 2 == 0 else torch.cat(
                [u1, u2], dim=-1)

        log_prob_u = -0.5 * torch.sum(u**2, dim=-1)
        log_prob = log_prob_u + log_det

        return u, log_prob

    def sample(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample `epsilon ~ q_psi(epsilon|z)` using the inverse of the flow. No gradients computed.

        Args:
            z (torch.Tensor): Conditioning batch `[..., D_z]`.
            num_samples (int): Number of samples per `z`.
        Returns:
            (z_aux, epsilon_aux) (torch.Tensor, torch.Tensor): Tiled conditioning `z_aux` and samples `epsilon_aux` with
            shapes `[..., num_samples, D_z]` and `[..., num_samples, D_epsilon]`.
        """
        # Sample from base distribution
        with torch.no_grad():
            batch_size = z.shape[0]
            u = torch.randn(batch_size, num_samples,
                            self.epsilon_dim).to(self.device)
            z_aux = z.clone().detach().unsqueeze(1).repeat(1, num_samples, 1)
            for i in reversed(range(len(self.scale_nets))):
                # Split
                u1, u2 = u.chunk(2, dim=-1)

                # Compute scale and translation
                # scale = torch.tanh(self.scale_nets[i](torch.cat(
                #     [u1, z_aux],
                #     dim=-1,
                # )))
                scale = self.scale_nets[i](torch.cat(
                    [u1, z_aux],
                    dim=-1,
                ))
                translate = self.translate_nets[i](torch.cat(
                    [u1, z_aux],
                    dim=-1,
                ))

                # Inverse transform
                u2 = (u2 - translate) * torch.exp(-scale)

                # Concatenate (alternate split)
                u = torch.cat([u2, u1], dim=-1) if i % 2 == 0 else torch.cat(
                    [u1, u2], dim=-1)
            return z_aux.clone().detach(), u.clone().detach()


class ConditionalGaussianReverse(nn.Module):
    """
    Gaussian reverse model for q_psi(epsilon|z) via joint Gaussian. Assume the joint distribution is Gaussian.

    Provides:
    - fit(epsilon, z): estimate joint mean/cov from provided data (no gradients)
    - sample(z, num_samples): sample epsilon' from conditional Gaussian q(epsilon|z)

    Args:
        z_dim (int): Dimension of z.
        epsilon_dim (int): Dimension of epsilon.
        device (torch.device): Storage and compute device for parameters and outputs.
    """

    def __init__(
        self,
        z_dim: int,
        epsilon_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.z_dim = z_dim
        self.epsilon_dim = epsilon_dim
        self.device = device
        self.jitter = 1e-6

        # Parameters of joint Gaussian over [epsilon, z]
        self.mu: torch.Tensor
        self.Sigma: torch.Tensor
        self.register_buffer(
            'mu',
            torch.zeros(epsilon_dim + z_dim, dtype=torch.float32),
        )
        self.register_buffer(
            'Sigma',
            torch.eye(epsilon_dim + z_dim, dtype=torch.float32),
        )

    @torch.no_grad()
    def fit(
        self,
        epsilon: torch.Tensor,
        z: torch.Tensor,
    ) -> None:
        """
        Fit joint Gaussian parameters from provided samples.
        epsilon: [..., De], z: [..., Dz]

        Args:
            epsilon (torch.Tensor): Samples of epsilon with shape [..., De].
            z (torch.Tensor): Samples of z with shape [..., Dz].
            jitter (float): Small value added to the diagonal of covariance for numerical stability.
        """
        assert epsilon.shape[-1] == self.epsilon_dim and z.shape[
            -1] == self.z_dim, "Dimension mismatch in fit inputs."
        epsilon = epsilon.clone().detach().reshape(-1, self.epsilon_dim)
        z = z.clone().detach().reshape(-1, self.z_dim)
        # Concatenate samples
        X = torch.cat([epsilon, z], dim=1)  # [N, z_dim + epsilon_dim]
        mu = X.mean(dim=0)  # (z_dim + epsilon_dim,)
        Xc = X - mu
        # Covariance with variables in columns: Sigma = (Xc^T Xc)/(N-1)
        N = X.shape[0]
        Sigma = (Xc.t() @ Xc) / max(1, N - 1)
        Sigma = Sigma + self.jitter * torch.eye(
            self.epsilon_dim + self.z_dim,
            device=self.device,
            dtype=Sigma.dtype,
        )
        self.mu.copy_(mu)
        self.Sigma.copy_(Sigma)

    def _conditional_params(
            self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute A, b, cond_cov for epsilon|z from stored mu/Sigma.
        Returns:
            (A, b, cond_cov) (torch.Tensor, torch.Tensor, torch.Tensor): where epsilon|z ~ N(Az + b, cond_cov)
        """
        mu = self.mu
        Sigma = self.Sigma
        mu_e = mu[:self.epsilon_dim]
        mu_z = mu[self.epsilon_dim:]
        S_ee = Sigma[:self.epsilon_dim, :self.epsilon_dim]
        S_ez = Sigma[:self.epsilon_dim, self.epsilon_dim:]
        S_ze = Sigma[self.epsilon_dim:, :self.epsilon_dim]
        S_zz = Sigma[self.epsilon_dim:, self.epsilon_dim:]
        # Regularized inverse for stability
        S_zz_reg = S_zz + self.jitter * torch.eye(
            self.z_dim,
            device=self.device,
            dtype=Sigma.dtype,
        )
        S_zz_inv = torch.linalg.inv(S_zz_reg)
        A = S_ez @ S_zz_inv
        b = mu_e - A @ mu_z
        cond_cov = S_ee - A @ S_ze
        # Ensure SPD
        try:
            torch.linalg.cholesky(cond_cov)  #TODO
        except RuntimeError:
            eigvals, eigvecs = torch.linalg.eigh(cond_cov)
            eigvals = torch.clamp(eigvals, min=1e-10)
            cond_cov = (eigvecs * eigvals) @ eigvecs.transpose(0, 1)
        return A, b, cond_cov

    @torch.no_grad()
    def sample(
        self,
        z: torch.Tensor,
        num_samples: int = 100,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample epsilon' ~ q_psi(epsilon|z) using conditional Gaussian.

        Args:
            z (torch.Tensor): [B, Dz]
            num_samples (int): samples per z
        Returns:
            (z_aux, epsilon_aux): shapes [B, S, Dz], [B, S, De]
        """
        assert z.dim() == 2 and z.shape[1] == self.z_dim
        B = z.shape[0]
        S = num_samples
        z_aux = z.detach().to(self.device).unsqueeze(1).repeat(1, S, 1)

        A, b, cond_cov = self._conditional_params()
        # Means: [B,S,De] = (z_aux @ A^T) + b
        mean = torch.matmul(z_aux, A.transpose(0, 1)) + b.view(1, 1, -1)
        # Cholesky of cond_cov
        L = torch.linalg.cholesky(cond_cov)
        noise = torch.randn(B, S, self.epsilon_dim, device=self.device)
        eps = mean + torch.matmul(noise, L.transpose(0, 1))
        return z_aux.clone().detach(), eps.clone().detach()


class ConditionalGaussian(nn.Module):
    """
    Conditional Gaussian `q_phi(z|epsilon)` parameterized by an MLP.

    Args:
        epsilon_dim (int): Dimension of `epsilon`.
        hidden_dim (int): Hidden size of the MLP.
        z_dim (int): Dimension of latent `z`.
        device (torch.device): Device for computation.
    """

    def __init__(
        self,
        epsilon_dim: int,
        hidden_dim: int,
        z_dim: int,
        device: torch.device,
    ):
        super().__init__()
        self.epsilon_dim = epsilon_dim
        self.hidden_dim = hidden_dim
        self.out_dim = z_dim * 2
        self.log_var_min = -10
        # The network outputs both mean and log-variance
        self.net = nn.Sequential(
            nn.Linear(self.epsilon_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.device = device

    def reparameterize(
        self,
        mu: torch.Tensor,
        log_var: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Reparameterization trick for `q_phi(z|epsilon)`.

        Args:
            mu (torch.Tensor): Mean.
            log_var (torch.Tensor): Log-variance.
        Returns:
            (z, neg_score) (torch.Tensor, torch.Tensor): Sample `z` and negative score `u/std` where
            `z = mu + std * u` and `u ~ N(0, I)`.
        """
        std = torch.exp(log_var / 2)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `mu(epsilon)` from the network output split."""
        return self.net(epsilon).chunk(2, dim=-1)[0]

    def getstd(self, epsilon: torch.Tensor) -> torch.Tensor:
        """Return `std(epsilon)` by clamping log-var and exponentiating."""
        log_var = self.net(epsilon).chunk(
            2, dim=-1)[1].clamp(min=self.log_var_min)
        std = torch.exp(log_var / 2)
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
        mu, log_var = self.net(epsilon).chunk(2, dim=-1)
        log_var = log_var.clamp(min=self.log_var_min)
        z, neg_score_implicit = self.reparameterize(mu, log_var)
        return z, neg_score_implicit

    def sampling(
        self,
        num: int = 1000,
        sigma: float = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample pairs `(epsilon, z)` from the conditional Gaussian. No gradients computed.

        Args:
            num (int): Number of samples.
            sigma (float, optional): Scale of the `epsilon` prior, default to 1.
        Returns:
            (epsilon, z) (torch.Tensor, torch.Tensor): `epsilon` and `z` samples.
        """
        with torch.no_grad():
            epsilon = torch.randn([num, self.epsilon_dim], ).to(self.device)
            epsilon = epsilon * sigma
            Z, _ = self.forward(epsilon)
        return epsilon.clone().detach(), Z.clone().detach()

    def neg_score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute the negative score term `(z - mu) / var` used in objectives.

        Args:
            z (torch.Tensor): Latent sample with shape `[..., D_z]`.
            epsilon (torch.Tensor): Conditioning input with shape `[..., D_epsilon]`.
        Returns:
            neg_score (torch.Tensor): Negative score `(z - mu(epsilon)) / var(epsilon)`.
        """
        mu, log_var = self.net(epsilon).chunk(2, dim=-1)
        log_var = log_var.clamp(min=self.log_var_min)
        var = torch.exp(log_var)
        neg_score = (z - mu) / (var)
        return neg_score
