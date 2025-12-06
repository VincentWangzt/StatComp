import torch
import torch.nn as nn


class SIMINet(nn.Module):

    def __init__(self, namedict, device):
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
        self.log_var = nn.Parameter(torch.zeros(namedict.out_dim) +
                                    namedict.log_var_ini,
                                    requires_grad=True)
        self.device = device
        self.log_var_min = namedict.log_var_min

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        eps = torch.randn_like(mu)
        return mu + std * eps, eps / std

    def getmu(self, Z):
        return self.mu(Z)

    def getstd(self):
        log_var = self.log_var.clamp(min=self.log_var_min)
        std = torch.exp(log_var / 2)
        return std

    def forward(self, Z):
        mu = self.mu(Z)
        log_var = self.log_var.clamp(min=self.log_var_min)
        X, neg_score_implicit = self.reparameterize(mu, log_var)
        return X, neg_score_implicit

    def sampling(self, num=1000, sigma=1):
        with torch.no_grad():
            Z = torch.randn([num, self.z_dim], ).to(self.device)
            Z = Z * sigma
            X, _ = self.forward(Z)
        return X


class ConditionalRealNVP(nn.Module):
    """RealNVP for conditional density estimation"""

    def __init__(
        self,
        z_dim,
        epsilon_dim,
        hidden_dim=128,
        num_layers=4,
        device='cpu',
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
                nn.Sequential(nn.Linear(z_dim + epsilon_dim // 2, hidden_dim),
                              nn.Tanh(), nn.Linear(hidden_dim, hidden_dim),
                              nn.Tanh(), nn.Linear(hidden_dim,
                                                   epsilon_dim // 2)))
            self.translate_nets.append(
                nn.Sequential(nn.Linear(z_dim + epsilon_dim // 2, hidden_dim),
                              nn.Tanh(), nn.Linear(hidden_dim, hidden_dim),
                              nn.Tanh(), nn.Linear(hidden_dim,
                                                   epsilon_dim // 2)))

    def forward(self, epsilon, z):
        """Forward pass: epsilon -> base_dist"""
        batch_size = epsilon.shape[0]
        log_det = torch.zeros(batch_size).to(self.device)

        # Forward: epsilon -> u (base distribution)
        u = epsilon.clone()
        for i in range(len(self.scale_nets)):
            # Split
            u1, u2 = u.chunk(2, dim=-1)

            # Compute scale and translation
            scale = torch.tanh(self.scale_nets[i](torch.cat([u1, z], dim=-1)))
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

    def sample(self, z, num_samples=100):
        """Sample epsilon given a batch of z"""
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
                scale = torch.tanh(self.scale_nets[i](torch.cat(
                    [u1, z_aux],
                    dim=-1,
                )))
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


class ConditionalGaussian(nn.Module):

    def __init__(self, epsilon_dim, hidden_dim, z_dim, device):
        super().__init__()
        self.epsilon_dim = epsilon_dim
        self.hidden_dim = hidden_dim
        self.out_dim = z_dim * 2
        self.log_var_min = -10
        self.net = nn.Sequential(
            nn.Linear(self.epsilon_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.out_dim),
        )
        self.device = device

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon):
        return self.net(epsilon).chunk(2, dim=-1)[0]

    def getstd(self, epsilon):
        log_var = self.net(epsilon).chunk(
            2, dim=-1)[1].clamp(min=self.log_var_min)
        std = torch.exp(log_var / 2)
        return std

    def forward(self, epsilon):
        mu, log_var = self.net(epsilon).chunk(2, dim=-1)
        log_var = log_var.clamp(min=self.log_var_min)
        z, neg_score_implicit = self.reparameterize(mu, log_var)
        return z, neg_score_implicit

    def sampling(self, num=1000, sigma=1):
        with torch.no_grad():
            epsilon = torch.randn([num, self.epsilon_dim], ).to(self.device)
            epsilon = epsilon * sigma
            Z, _ = self.forward(epsilon)
        return epsilon.clone().detach(), Z.clone().detach()

    def neg_score(self, z, epsilon):
        mu, log_var = self.net(epsilon).chunk(2, dim=-1)
        log_var = log_var.clamp(min=self.log_var_min)
        var = torch.exp(log_var)
        neg_score = (z - mu) / (var)
        return neg_score
