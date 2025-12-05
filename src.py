import torch
from models.target_models import target_distribution
from models.networks import SIMINet
import os
import torch.nn as nn
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
                              nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                              nn.ReLU(), nn.Linear(hidden_dim,
                                                   epsilon_dim // 2)))
            self.translate_nets.append(
                nn.Sequential(nn.Linear(z_dim + epsilon_dim // 2, hidden_dim),
                              nn.ReLU(), nn.Linear(hidden_dim, hidden_dim),
                              nn.ReLU(), nn.Linear(hidden_dim,
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
            return z_aux, u


# Training for flow matching
def train_flow_matching(dataloader, model, optimizer, epochs=100):
    model.train()
    optimizer.zero_grad()
    # Training loop
    for epoch in range(epochs):
        total_loss = 0
        for batch in dataloader:
            # maximize log likelihood
            z, epsilon = batch
            z = z.to(model.device)
            epsilon = epsilon.to(model.device)

            u, log_prob = model(epsilon, z)
            loss = -torch.mean(log_prob)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    return total_loss / len(dataloader)


class MixtureGaussian(nn.Module):

    def __init__(self, epsilon_dim, h_dim, out_dim, device):
        super(MixtureGaussian, self).__init__()
        self.epsilon_dim = epsilon_dim
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.log_var_min = -10
        self.mu = nn.Sequential(nn.Linear(self.epsilon_dim, self.h_dim),
                                nn.ReLU(), nn.Linear(self.h_dim, self.h_dim),
                                nn.ReLU(), nn.Linear(self.h_dim, self.out_dim))
        self.log_var = nn.Parameter(torch.ones(out_dim), requires_grad=True)
        self.device = device

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var / 2)
        u = torch.randn_like(mu)
        return mu + std * u, u / std

    def getmu(self, epsilon):
        return self.mu(epsilon)

    def getstd(self):
        log_var = self.log_var.clamp(min=self.log_var_min)
        std = torch.exp(log_var / 2)
        return std

    def forward(self, epsilon):
        mu = self.mu(epsilon)
        log_var = self.log_var.clamp(min=self.log_var_min)
        z, neg_score_implicit = self.reparameterize(mu, log_var)
        return z, neg_score_implicit

    def sampling(self, num=1000, sigma=1):
        with torch.no_grad():
            epsilon = torch.randn([num, self.epsilon_dim], ).to(self.device)
            epsilon = epsilon * sigma
            Z, _ = self.forward(epsilon)
        return Z

    def neg_score(self, z, epsilon):
        mu = self.mu(epsilon)
        log_var = self.log_var.clamp(min=self.log_var_min)
        var = torch.exp(log_var)
        neg_score = (z - mu) / (var)
        return neg_score


class ReverseUIVI():

    def __init__(self, target_dist: str, device):
        self.target_dist = target_dist
        self.device = device
        self.target_model = target_distribution[target_dist](device=device)
        self.vi_model = MixtureGaussian(
            epsilon_dim=4,
            h_dim=64,
            out_dim=2,
            device=self.device,
        ).to(self.device)
        self.reverse_model = ConditionalRealNVP(
            z_dim=2,
            epsilon_dim=4,
            hidden_dim=64,
            num_layers=4,
            device=self.device,
        ).to(self.device)
        self.save_path = os.path.join("results", "reverse_uivi", target_dist,
                                      datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.save_path, exist_ok=True)

    def warmup(self, epochs=10, lr=1e-4, batch_size=512, epoch_size=8192):

        # Create dataset for warm-up
        optimizer = torch.optim.Adam(self.reverse_model.parameters(), lr=lr)

        for epoch in range(epochs):
            epsilon_samples = torch.randn(
                epoch_size,
                self.vi_model.epsilon_dim,
            ).to(self.device)
            with torch.no_grad():
                z_samples, _ = self.vi_model.forward(epsilon_samples)
                z_samples = z_samples.detach()
            dataset = TensorDataset(z_samples, epsilon_samples)
            dataloader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
            )

            loss = train_flow_matching(
                dataloader,
                self.reverse_model,
                optimizer,
                epochs=1,
            )

            print(f"Warm-up Loss(epoch {epoch + 1}): {loss}")
            self.print_KL()

    def print_KL(self, n_ite_samples=10000):
        import ite
        print("\nCalculating KL Divergence using 'ite' package...")
        # A. True Joint Distribution Samples: (epsilon, z)
        true_eps = torch.randn(n_ite_samples,
                               self.vi_model.epsilon_dim).to(self.device)
        with torch.no_grad():
            true_z, _ = self.vi_model.forward(true_eps)
        true_joint = torch.cat([true_eps, true_z], dim=1).cpu().numpy()

        with torch.no_grad():
            generated_eps = self.reverse_model.sample(
                true_z, num_samples=1)[1].squeeze(1)
        generated_joint = torch.cat([generated_eps, true_z],
                                    dim=1).cpu().numpy()

        # C. Estimate KL
        # Compare true joint vs generated joint
        cost_obj = ite.cost.BDKL_KnnK()
        kl_div = cost_obj.estimation(generated_joint, true_joint)
        print(f"   KL Divergence: {kl_div:.4f}")

    def learn(self, num_epochs=100, num_per_epoch=100, batch_size=64):
        self.warmup()
        optimizer_vi = torch.optim.Adam(self.vi_model.parameters(), lr=1e-3)
        scheduler_vi = torch.optim.lr_scheduler.StepLR(
            optimizer_vi,
            step_size=1000,
            gamma=0.9,
        )
        for epoch in range(1, num_epochs + 1):

            total_loss = 0
            reverse_total_loss = 0
            for _ in range(num_per_epoch):
                # Sample epsilon
                epsilon = torch.randn(
                    batch_size,
                    self.vi_model.epsilon_dim,
                ).to(self.device)

                # Sample z from variational distribution
                z, neg_score_implicit = self.vi_model.forward(epsilon)

                # Compute log prob under target distribution
                log_prob_target = self.target_model.logp(z)

                with torch.no_grad():
                    z_aux, epsilon_aux = self.reverse_model.sample(
                        z, num_samples=100)
                    neg_score = self.vi_model.neg_score(z_aux, epsilon_aux)
                    neg_score = neg_score.mean(dim=1)
                    neg_score = neg_score.detach()

                # Compute loss
                loss = -torch.mean(
                    log_prob_target + torch.sum(
                        z * neg_score,
                        dim=-1,
                    ), )

                optimizer_vi.zero_grad()
                loss.backward()
                optimizer_vi.step()

                # Train reverse model
                with torch.no_grad():
                    epsilon_rev = torch.randn(
                        1024, self.vi_model.epsilon_dim).to(self.device)
                    z_rev, _ = self.vi_model.forward(epsilon_rev)
                dataset = TensorDataset(z_rev.clone().detach(),
                                        epsilon_rev.clone().detach())
                dataloader = DataLoader(
                    dataset,
                    batch_size=64,
                    shuffle=True,
                )
                reverse_loss = train_flow_matching(
                    dataloader,
                    self.reverse_model,
                    torch.optim.Adam(self.reverse_model.parameters(), lr=1e-3),
                    epochs=1,
                )
                reverse_total_loss += reverse_loss

                total_loss += loss.item()

            scheduler_vi.step()
            print(f"Epoch {epoch}, Loss: {total_loss / num_per_epoch}")
            print(
                f"Epoch {epoch}, Reverse Loss: {reverse_total_loss / num_per_epoch}"
            )
            with torch.no_grad():
                Z = self.vi_model.sampling()
                os.makedirs(os.path.join(self.save_path, "samples"),
                            exist_ok=True)
                torch.save(
                    Z,
                    os.path.join(self.save_path, "samples",
                                 "samples_epoch_" + str(epoch) + '.pt'),
                )
            os.makedirs(os.path.join(self.save_path, "figures"), exist_ok=True)
            save_to_path = os.path.join(
                self.save_path,
                "figures",
                "figure_epoch_" + str(epoch) + '.png',
            )
            bbox = {
                "multimodal": [-5, 5, -5, 5],
                "banana": [-3.5, 3.5, -6, 1],
                "x_shaped": [-5, 5, -5, 5],
            }
            quiver_plot = False
            self.target_model.contour_plot(
                bbox[self.target_dist],
                fnet=None,
                samples=Z.cpu().numpy(),
                save_to_path=save_to_path,
                quiver=quiver_plot,
                t=epoch,
            )
            self.print_KL()


if __name__ == "__main__":
    reverse_uivi = ReverseUIVI(target_dist="banana", device=device)
    reverse_uivi.learn(num_epochs=100, num_per_epoch=100, batch_size=64)
