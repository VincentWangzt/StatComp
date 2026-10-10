import torch
import torch.nn as nn
import math


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
            nn.Linear(self.h_dim, self.out_dim)
        )
        self.variance_mode = getattr(namedict, "variance_mode", "global")
        if self.variance_mode == "global":
            self.log_var = nn.Parameter(torch.zeros(namedict.out_dim) + namedict.log_var_ini, requires_grad = True)
        elif self.variance_mode == "conditional":
            # Share the original mean network's two hidden layers. Preserve its
            # initialization and the subsequent training-noise RNG stream.
            rng = torch.random.get_rng_state()
            self.variance_head = nn.Linear(self.h_dim, self.out_dim)
            torch.random.set_rng_state(rng)
            nn.init.zeros_(self.variance_head.weight)
            nn.init.constant_(self.variance_head.bias, namedict.log_var_ini)
            self.register_parameter("log_var", None)
        else:
            raise ValueError(f"Unknown variance_mode: {self.variance_mode}")
        self.device = device
        self.log_var_min = namedict.log_var_min
        variance_floor = getattr(namedict, "variance_floor", None)
        if variance_floor is not None:
            if variance_floor <= 0:
                raise ValueError("variance_floor must be positive")
            self.log_var_min = max(self.log_var_min, math.log(variance_floor))
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(mu)
        return mu + std * eps, eps/std
    
    def getmu(self, Z):
        return self.mu(Z)
    
    def getstd(self, Z=None):
        if self.variance_mode == "conditional":
            if Z is None:
                raise ValueError("Conditional standard deviations require latent Z")
            log_var = self.variance_head(self.mu[:-1](Z))
        else:
            log_var = self.log_var
        log_var = log_var.clamp(min = self.log_var_min)
        std = torch.exp(log_var/2)
        return std
    
    def forward(self, Z):
        if self.variance_mode == "conditional":
            features = self.mu[:-1](Z)
            mu = self.mu[-1](features)
            log_var = self.variance_head(features)
        else:
            mu = self.mu(Z)
            log_var = self.log_var
        log_var = log_var.clamp(min = self.log_var_min)
        X, neg_score_implicit = self.reparameterize(mu, log_var)
        return X, neg_score_implicit

    def sampling(self, num = 1000, sigma = 1):
        with torch.no_grad():
            Z = torch.randn([num, self.z_dim], ).to(self.device)
            Z = Z * sigma
            X, _ = self.forward(Z)
        return X
