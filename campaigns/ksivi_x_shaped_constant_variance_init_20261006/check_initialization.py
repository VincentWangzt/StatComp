"""Check constant variance initialization without freezing conditional variance."""

from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from models.vi_model import ConditionalGaussian, ConditionalGaussianGlobal

VARIANCE = torch.nn.functional.softplus(torch.tensor(1.0)).item()
BASE = {"z_dim": 2, "epsilon_dim": 2, "hidden_dim": 128,
        "num_layers": 2, "device": "cpu"}


def build(seed, **overrides):
    torch.manual_seed(seed)
    return ConditionalGaussian(OmegaConf.create({**BASE, **overrides}))


def main():
    for seed in (42, 43, 44):
        baseline = build(seed)
        null_option = build(seed, variance_init=None)
        fixed = build(seed, variance_init=VARIANCE)
        for name, value in baseline.state_dict().items():
            assert torch.equal(value, null_option.state_dict()[name])
            if name.startswith("net.4."):
                assert torch.equal(value[:2], fixed.state_dict()[name][:2])
            else:
                assert torch.equal(value, fixed.state_dict()[name])
        epsilon = torch.randn(256, 2)
        initial_var = fixed.getstd(epsilon).square()
        assert torch.allclose(initial_var, torch.full_like(initial_var, VARIANCE))
        assert initial_var.std(0).max().item() < 1e-6
        assert baseline.getstd(epsilon).square().std(0).min().item() > 1e-5
        assert torch.count_nonzero(fixed.net[-1].weight[2:]).item() == 0
        assert all(parameter.requires_grad for parameter in fixed.parameters())
        assert torch.equal(baseline.getmu(epsilon), fixed.getmu(epsilon))
        # A likelihood gradient must update the variance head and produce input dependence.
        optimizer = torch.optim.Adam(fixed.parameters(), lr=0.001)
        observation = torch.stack((2 * epsilon[:, 0], epsilon[:, 1] - 3), dim=-1)
        loss = -fixed.logp(observation, epsilon).mean()
        loss.backward()
        gradient = fixed.net[-1].weight.grad[2:]
        assert torch.isfinite(gradient).all().item() and gradient.norm().item() > 0
        optimizer.step()
        after = fixed.getstd(epsilon).square()
        assert after.std(0).min().item() > 1e-5
        assert not torch.equal(initial_var, after)
        print(f"seed {seed}: constant initial variance {initial_var[0].tolist()}, "
              f"trainable variance std after one update {after.std(0).tolist()}")
    global_model = ConditionalGaussianGlobal(OmegaConf.create(BASE))
    assert torch.allclose(global_model.getstd(torch.zeros(1, 2)).square(),
                          torch.full((2,), VARIANCE))
    logvar = build(42, variance_parameterization="logvar", log_var_init=-8.0)
    assert torch.allclose(logvar.getstd(torch.zeros(10, 2)).square(),
                          torch.full((10, 2), torch.exp(torch.tensor(-8.0)).item()))
    logvar_fixed = build(42, variance_parameterization="logvar", variance_init=0.25)
    assert torch.allclose(logvar_fixed.getstd(torch.randn(10, 2)).square(),
                          torch.full((10, 2), 0.25))
    for invalid in (0, -1, float("nan"), float("inf"), 1e-5):
        try:
            build(42, variance_init=invalid)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid variance_init accepted: {invalid}")
    print("Default/logvar initialization, global starting scale, and input validation passed.")


if __name__ == "__main__":
    main()
