"""Regression checks for splitting the original conditional Gaussian readout."""

import unittest

import torch
from omegaconf import OmegaConf
from torch import nn

from models.target_models import X_shaped
from models.vi_model import ConditionalGaussian, ConditionalGaussianUniform
from utils.annealing import annealing
from utils.kernels import GaussianKernel


class LegacyJointGaussian(nn.Module):
    """The pre-split initialization and sampling calculation, kept as a reference."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        input_dim = config.epsilon_dim
        for _ in range(config.num_layers):
            layers.append(nn.Linear(input_dim, config.hidden_dim))
            layers.append(nn.ReLU() if config.get('activation') == 'relu' else nn.SiLU())
            input_dim = config.hidden_dim
        layers.append(nn.Linear(config.hidden_dim, 2 * config.z_dim))
        self.net = nn.Sequential(*layers)
        if config.get('variance_parameterization') == 'logvar':
            with torch.no_grad():
                self.net[-1].weight[config.z_dim:].zero_()
                self.net[-1].bias[config.z_dim:].fill_(config.get('log_var_init', 0.0))

    def forward(self, epsilon):
        mu, raw = self.net(epsilon).chunk(2, dim=-1)
        if self.config.get('variance_parameterization') == 'logvar':
            variance = raw.clamp(min=self.config.get('log_var_min', -20.0)).exp()
        else:
            variance = torch.nn.functional.softplus(raw).clamp(
                min=self.config.get('var_min', 1e-4))
        std = variance.sqrt()
        noise = torch.randn_like(mu)
        return mu + std * noise, noise / std


def packed_parameters(model):
    if isinstance(model, LegacyJointGaussian):
        return list(model.parameters())
    return list(model.backbone.parameters()) + [
        torch.cat((model.mean_head.weight, model.variance_head.weight)),
        torch.cat((model.mean_head.bias, model.variance_head.bias)),
    ]


def ksivi_loss(model, target, step):
    epsilon_x = torch.randn(128, 2)
    epsilon_y = torch.randn(128, 2)
    x, b = model(epsilon_x)
    y, c = model(epsilon_y)
    factor = annealing(t=step, warm_up_interval=25000, anneal=True, scheme='linear')
    kernel = GaussianKernel().pair_eval(x, y, fit_h=True, detach_h=False)
    return (((factor * target.score(x) + b) @ (factor * target.score(y) + c).T)
            * kernel).mean()


class ConditionalGaussianTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.addCleanup(torch.random.set_rng_state, torch.random.get_rng_state())
        self.config = OmegaConf.create(dict(
            device='cpu', epsilon_dim=2, z_dim=2, hidden_dim=128, num_layers=2))

    def assert_parameters_equal(self, first, second):
        self.assertEqual(len(packed_parameters(first)), len(packed_parameters(second)))
        for index, (left, right) in enumerate(zip(
                packed_parameters(first), packed_parameters(second))):
            self.assertTrue(torch.equal(left, right), f'parameter {index}')

    def test_initialization_rng_and_sampling_match_legacy(self):
        variants = [
            (ConditionalGaussian, {}),
            (ConditionalGaussianUniform, {}),
            (ConditionalGaussian, dict(
                epsilon_dim=3, z_dim=31, hidden_dim=10, activation='relu',
                variance_parameterization='logvar', log_var_init=-8.0)),
        ]
        for model_type, overrides in variants:
            config = OmegaConf.merge(self.config, overrides)
            for seed in (42, 43, 44):
                with self.subTest(model=model_type.__name__, overrides=overrides, seed=seed):
                    torch.manual_seed(seed)
                    legacy = LegacyJointGaussian(config)
                    expected_rng = torch.random.get_rng_state()
                    torch.manual_seed(seed)
                    model = model_type(config)
                    self.assertTrue(torch.equal(expected_rng, torch.random.get_rng_state()))
                    self.assert_parameters_equal(legacy, model)
                    epsilon = torch.randn(3, 5, config.epsilon_dim)
                    torch.manual_seed(100)
                    expected = legacy(epsilon)
                    torch.manual_seed(100)
                    actual = model(epsilon)
                    for left, right in zip(expected, actual):
                        self.assertTrue(torch.equal(left, right))
                    torch.manual_seed(101)
                    prior = torch.rand if model_type is ConditionalGaussianUniform else torch.randn
                    expected_epsilon = prior(16, config.epsilon_dim)
                    expected_z = legacy(expected_epsilon)[0]
                    torch.manual_seed(101)
                    actual_epsilon, actual_z = model.sampling(16)
                    self.assertTrue(torch.equal(expected_epsilon, actual_epsilon))
                    self.assertTrue(torch.equal(expected_z, actual_z))

    def test_missing_and_equal_lr_match_legacy_training(self):
        target = X_shaped(torch.device('cpu'))
        for variance_lr in (None, 1e-3):
            for foreach in (None, True):
                for seed in (42, 43, 44):
                    with self.subTest(variance_lr=variance_lr, foreach=foreach, seed=seed):
                        torch.manual_seed(seed)
                        legacy = LegacyJointGaussian(self.config)
                        torch.manual_seed(seed)
                        model = ConditionalGaussian(self.config)
                        old_optimizer = torch.optim.Adam(
                            legacy.parameters(), lr=1e-3, foreach=foreach)
                        optimizer = torch.optim.Adam(
                            model.optimizer_parameter_groups(1e-3, variance_lr),
                            lr=1e-3, foreach=foreach)
                        schedulers = [torch.optim.lr_scheduler.StepLR(opt, 10, 0.9)
                                      for opt in (old_optimizer, optimizer)]
                        for step in range(1, 41):
                            losses = []
                            for candidate, opt in ((legacy, old_optimizer), (model, optimizer)):
                                torch.manual_seed(10000 + step)
                                opt.zero_grad()
                                loss = ksivi_loss(candidate, target, step)
                                loss.backward()
                                opt.step()
                                losses.append(loss.detach())
                            for scheduler in schedulers:
                                scheduler.step()
                            self.assertTrue(torch.equal(*losses), f'loss at step {step}')
                            self.assert_parameters_equal(legacy, model)
                            self.assertEqual(
                                [group['lr'] for group in optimizer.param_groups],
                                [old_optimizer.param_groups[0]['lr']] * 2)

    def test_variance_lr_scales_only_variance_head_update(self):
        torch.manual_seed(42)
        equal = ConditionalGaussian(self.config)
        torch.manual_seed(42)
        slow = ConditionalGaussian(self.config)
        groups = slow.optimizer_parameter_groups(1e-3, 1e-5)
        grouped_ids = [id(parameter) for group in groups for parameter in group['params']]
        self.assertEqual(len(grouped_ids), len(set(grouped_ids)))
        self.assertEqual(set(grouped_ids), {id(parameter) for parameter in slow.parameters()})
        self.assertEqual(
            {id(parameter) for parameter in groups[1]['params']},
            {id(parameter) for parameter in slow.variance_head.parameters()})
        before = {name: parameter.detach().clone() for name, parameter in slow.named_parameters()}
        optimizers = [torch.optim.Adam(model.optimizer_parameter_groups(1e-3, rate), lr=1e-3)
                      for model, rate in ((equal, 1e-3), (slow, 1e-5))]
        target = X_shaped(torch.device('cpu'))
        for model, optimizer in zip((equal, slow), optimizers):
            torch.manual_seed(10000)
            optimizer.zero_grad()
            ksivi_loss(model, target, 1).backward()
            optimizer.step()
        for (name, reference), (_, actual) in zip(equal.named_parameters(), slow.named_parameters()):
            if name.startswith('variance_head.'):
                torch.testing.assert_close(
                    actual - before[name], 0.01 * (reference - before[name]),
                    rtol=1e-3, atol=1e-8)
            else:
                self.assertTrue(torch.equal(reference, actual), name)


if __name__ == '__main__':
    unittest.main()
