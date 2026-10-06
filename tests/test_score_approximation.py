from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from omegaconf import OmegaConf

from finalization.score_approximation import (
    REPO_ROOT, assess_hmc_reference_quality, build_frozen_estimators,
    compute_score_metrics, load_score_config, native_aisivi_score,
    native_sivi_score, native_uivi_score, posterior_hmc_reference_scores,
    run_analysis, select_checkpoints, shared_input_bank,
)
from finalization.runner_eval import remove_file_handlers
from models.vi_model import ConditionalGaussian
from runner.runners import Runners


class FrozenCheckpointTest(unittest.TestCase):
    """Exercise actual checkpoint loading, proposal fitting, and cache reuse."""

    @classmethod
    def setUpClass(cls) -> None:
        (REPO_ROOT / "results").mkdir(exist_ok=True)
        (REPO_ROOT / "tb_logs").mkdir(exist_ok=True)
        cls.results = tempfile.TemporaryDirectory(prefix="score_test_", dir=REPO_ROOT / "results")
        cls.logs = tempfile.TemporaryDirectory(prefix="score_test_", dir=REPO_ROOT / "tb_logs")
        root = Path(cls.results.name)
        source_config = OmegaConf.load(REPO_ROOT / "configs/dsivi_banana.yaml")
        source_config = OmegaConf.merge(source_config, {
            "device": "cpu", "use_cuda": False,
            "config_path": str(REPO_ROOT / "configs/dsivi_banana.yaml"),
            "vi_model": {"hidden_dim": 8, "num_layers": 1},
            "reverse_model": {"hidden_dim": 8, "num_layers": 1},
            "output": {"results_dir": str(root / "source"), "tb_dir": cls.logs.name},
        })
        torch.manual_seed(19)
        runner = Runners["DSIVI"](source_config)
        runner.log_config()
        runner.save_checkpoint(1)
        cls.run_dir = Path(runner.save_path)
        runner.writer.close()
        remove_file_handlers()
        cls.cfg = load_score_config(overrides=["selection.targets=[banana]", "selection.checkpoint_epochs=[1]",
                                              "evaluation.device=cpu", "evaluation.forward_batch_size=4",
                                              "evaluation.aisivi_refit.steps=2", "evaluation.aisivi_refit.batch_size=4",
                                              "evaluation.reference.total_samples=16", "evaluation.reference.num_chains=2",
                                              "evaluation.reference.burn_in_steps=2", "evaluation.reference.leapfrog_steps=2"])
        cls.cfg.selection.run_dirs = [str(cls.run_dir)]
        cls.cfg.output = {
            "report_dir": str(root / "report"), "cache_dir": str(root / "cache"),
            "scratch_results_dir": str(root / "scratch"), "scratch_tb_dir": cls.logs.name,
        }
        for method in ("SIVI", "UIVI", "AISIVI"):
            config = OmegaConf.load(REPO_ROOT / f"configs/{method.lower()}_banana.yaml")
            config.train.reverse_sample_num = 4
            if method == "AISIVI":
                config.reverse_model = {"hidden_dim": 4, "num_layers": 2}
            path = root / f"{method}.yaml"
            OmegaConf.save(config, path)
            cls.cfg.estimator_configs[method] = str(path)

    @classmethod
    def tearDownClass(cls) -> None:
        remove_file_handlers()
        cls.results.cleanup()
        cls.logs.cleanup()

    def test_shared_checkpoint_is_frozen_and_fit_cache_is_reused(self) -> None:
        checkpoint = select_checkpoints(self.cfg)[0]
        runners = build_frozen_estimators(self.cfg, checkpoint)
        expected_vi = torch.load(checkpoint.checkpoint_dir / "vi_model.pt", weights_only=True)
        expected_score = torch.load(checkpoint.checkpoint_dir / "reverse_model.pt", weights_only=True)
        for method, runner in runners.items():
            self.assertFalse(any(parameter.requires_grad for parameter in runner.vi_model.parameters()))
            for key, value in expected_vi.items():
                torch.testing.assert_close(value, runner.vi_model.state_dict()[key])
            if method == "DSIVI":
                for key, value in expected_score.items():
                    torch.testing.assert_close(value, runner.reverse_model.state_dict()[key])
        inputs = [shared_input_bank(runner, checkpoint, count=4, seed=7) for runner in runners.values()]
        for epsilon, z in inputs[1:]:
            torch.testing.assert_close(inputs[0][0], epsilon, rtol=0, atol=0)
            torch.testing.assert_close(inputs[0][1], z, rtol=0, atol=0)
        fit_metadata = runners["AISIVI"].score_refit_metadata
        self.assertEqual(fit_metadata["steps"], 2)
        with patch.object(torch.optim.Adam, "step", side_effect=AssertionError("Refit should use its cache")):
            reloaded = build_frozen_estimators(self.cfg, checkpoint)
        self.assertEqual(reloaded["AISIVI"].score_refit_metadata, fit_metadata)
        for key, value in runners["AISIVI"].reverse_model.state_dict().items():
            torch.testing.assert_close(value, reloaded["AISIVI"].reverse_model.state_dict()[key])

    def test_analysis_reuses_hmc_and_keeps_method_diagnostics_separate(self) -> None:
        rows = run_analysis(self.cfg)
        self.assertEqual({row["method"] for row in rows}, {"SIVI", "UIVI", "AISIVI", "DSIVI"})
        self.assertEqual(len({row["vi_model_sha256"] for row in rows}), 1)
        self.assertTrue(all(row["method_l2"] >= 0 for row in rows))
        with patch("finalization.score_approximation.posterior_hmc_reference_scores", side_effect=AssertionError("Use HMC cache")):
            cached_rows = run_analysis(self.cfg)
        self.assertEqual(cached_rows, rows)
        metadata = json.loads((Path(self.cfg.output.report_dir) / "metadata.json").read_text())
        self.assertEqual(metadata["checkpoints"][0]["aisivi_refit"]["steps"], 2)

    def test_selection_rejects_missing_and_duplicate_checkpoints(self) -> None:
        cfg = OmegaConf.create(OmegaConf.to_container(self.cfg, resolve=True))
        cfg.selection.checkpoint_epochs = [2]
        with self.assertRaisesRegex(FileNotFoundError, "Missing checkpoint epochs"):
            select_checkpoints(cfg)
        cfg.selection.checkpoint_epochs = [1]
        cfg.selection.run_dirs = [str(self.run_dir), str(self.run_dir)]
        with self.assertRaisesRegex(ValueError, "Duplicate source checkpoint"):
            select_checkpoints(cfg)

    def test_cpu_evaluation_overrides_devices_in_gpu_training_snapshot(self) -> None:
        checkpoint = select_checkpoints(self.cfg)[0]
        snapshot = OmegaConf.load(checkpoint.config_path)
        original = checkpoint.config_path.read_bytes()
        snapshot.device = "cuda"
        snapshot.vi_model.device = "cuda"
        snapshot.reverse_model.device = "cuda"
        OmegaConf.save(snapshot, checkpoint.config_path)
        try:
            runners = build_frozen_estimators(self.cfg, checkpoint)
            for runner in runners.values():
                self.assertEqual(runner.device, "cpu")
                self.assertEqual(str(runner.vi_model.device), "cpu")
                epsilon, z = runner.vi_model.sampling(num=2)
                self.assertEqual(z.device.type, "cpu")
        finally:
            checkpoint.config_path.write_bytes(original)

    def test_score_error_and_reference_mc_error_definitions(self) -> None:
        method = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        chains = torch.tensor([[[0.0, 0.0], [0.0, 0.0]], [[2.0, 0.0], [0.0, 2.0]]])
        metrics = compute_score_metrics(method, chains)
        self.assertEqual(metrics["method_l2"], 0)
        self.assertEqual(metrics["reference_internal_l2"], 1)
        self.assertEqual(metrics["reference_mean_mcse_l2"], 1)

    def test_checkpoint_identity_includes_saved_variational_configuration(self) -> None:
        checkpoint = select_checkpoints(self.cfg)[0]
        before = checkpoint.key
        original = checkpoint.config_path.read_bytes()
        saved = OmegaConf.load(checkpoint.config_path)
        saved.vi_model.uniform = True
        OmegaConf.save(saved, checkpoint.config_path)
        try:
            self.assertNotEqual(before, checkpoint.key)
        finally:
            checkpoint.config_path.write_bytes(original)

    def test_quality_flags_nonfinite_rhat_even_when_finite_quantiles_pass(self) -> None:
        diagnostics = {"hmc_divergence_fraction": 0.0, "hmc_score_rhat_p95": 1.0,
                       "hmc_epsilon_rhat_p95": 1.0, "hmc_post_burn_acceptance_rate": 0.9,
                       "hmc_post_burn_acceptance_min": 0.8, "hmc_score_rhat_nonfinite_fraction": 0.01}
        quality, issues = assess_hmc_reference_quality(diagnostics, self.cfg.evaluation.reference.quality)
        self.assertEqual(quality, "warning")
        self.assertEqual(len(issues), 1)
        self.assertIn("nonfinite_fraction", issues[0])



def make_model(*, dtype: torch.dtype = torch.float64) -> ConditionalGaussian:
    cfg = OmegaConf.create({
        "z_dim": 2,
        "epsilon_dim": 4,
        "hidden_dim": 8,
        "num_layers": 1,
        "device": "cpu",
        "uniform": False,
    })
    return ConditionalGaussian(cfg).to(dtype=dtype)

class FakeReverse:

    def __init__(
        self,
        epsilon: torch.Tensor,
        log_prob: torch.Tensor,
    ) -> None:
        self.epsilon = epsilon
        self.log_prob_value = log_prob

    def sample(
        self,
        z: torch.Tensor,
        *,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_samples != self.epsilon.shape[1]:
            raise AssertionError("Unexpected sample count")
        z_aux = z.unsqueeze(1).expand(-1, num_samples, -1)
        return z_aux, self.epsilon, self.log_prob_value

class BatchLimitedReverse:

    def __init__(self, *, epsilon_dim: int) -> None:
        self.epsilon_dim = epsilon_dim

    def sample(
        self,
        z: torch.Tensor,
        *,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if z.shape[0] > 1:
            raise RuntimeError(
                "Failed to obtain finite samples from RealNVP after "
                "3 attempts."
            )
        z_aux = z.unsqueeze(1).expand(-1, num_samples, -1)
        epsilon = torch.zeros(
            z.shape[0],
            num_samples,
            self.epsilon_dim,
            dtype=z.dtype,
            device=z.device,
        )
        log_prob = torch.zeros(
            z.shape[0],
            num_samples,
            dtype=z.dtype,
            device=z.device,
        )
        return z_aux, epsilon, log_prob

class SampleLimitedReverse:

    def __init__(
        self,
        *,
        epsilon_dim: int,
        maximum_samples: int,
    ) -> None:
        self.epsilon_dim = epsilon_dim
        self.maximum_samples = maximum_samples

    def sample(
        self,
        z: torch.Tensor,
        *,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if num_samples > self.maximum_samples:
            raise RuntimeError(
                "Failed to obtain finite samples from RealNVP after "
                "3 attempts."
            )
        z_aux = z.unsqueeze(1).expand(-1, num_samples, -1)
        epsilon = torch.zeros(
            z.shape[0],
            num_samples,
            self.epsilon_dim,
            dtype=z.dtype,
            device=z.device,
        )
        log_prob = torch.zeros(
            z.shape[0],
            num_samples,
            dtype=z.dtype,
            device=z.device,
        )
        return z_aux, epsilon, log_prob

class LinearGaussianVI(torch.nn.Module):
    """One-dimensional model with an analytic epsilon posterior."""

    def __init__(self, conditional_variance: float = 0.5) -> None:
        super().__init__()
        self.conditional_variance = conditional_variance
        self.epsilon_dim = 1

    def log_q_epsilon(self, epsilon: torch.Tensor) -> torch.Tensor:
        return -0.5 * (
            torch.log(
                torch.tensor(
                    2.0 * torch.pi,
                    dtype=epsilon.dtype,
                    device=epsilon.device,
                )
            )
            + epsilon.square().sum(dim=-1)
        )

    def logp(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        variance = torch.as_tensor(
            self.conditional_variance,
            dtype=z.dtype,
            device=z.device,
        )
        return -0.5 * (
            torch.log(2.0 * torch.pi * variance)
            + ((z - epsilon).square() / variance).sum(dim=-1)
        )

    def score(
        self,
        z: torch.Tensor,
        epsilon: torch.Tensor,
    ) -> torch.Tensor:
        return -(z - epsilon) / self.conditional_variance

class ScoreApproximationTest(unittest.TestCase):

    def setUp(self) -> None:
            torch.manual_seed(123)

    def test_posterior_hmc_recovers_linear_gaussian_score(self) -> None:
            torch.manual_seed(321)
            model = LinearGaussianVI().to(dtype=torch.float64)
            z = torch.tensor(
                [[-1.5], [-0.5], [0.5], [1.5]],
                dtype=torch.float64,
            )
            posterior_variance = 0.5 / 1.5
            posterior_mean = z / 1.5
            generating_epsilon = (
                posterior_mean
                + posterior_variance**0.5 * torch.randn_like(z)
            )
            chain_scores, diagnostics = posterior_hmc_reference_scores(
                model,
                z,
                generating_epsilon,
                total_samples=800,
                num_chains=4,
                burn_in_steps=100,
                thinning=1,
                step_size=0.1,
                leapfrog_steps=5,
                init_jitter_scale=0.1,
                adapt_step_size=True,
                target_acceptance=0.8,
                adaptation_rate=0.1,
                min_step_size=0.01,
                max_step_size=0.2,
                divergence_threshold=1000.0,
                accumulator_dtype=torch.float64,
            )
            expected = -z / 1.5
            actual = chain_scores.mean(dim=0)
            torch.testing.assert_close(
                actual,
                expected,
                rtol=0.0,
                atol=0.16,
            )
            self.assertEqual(tuple(chain_scores.shape), (4, 4, 1))
            self.assertGreater(
                diagnostics["hmc_post_burn_acceptance_rate"],
                0.6,
            )
            self.assertLess(
                diagnostics["hmc_score_rhat_p95"],
                1.2,
            )
            self.assertEqual(diagnostics["hmc_divergence_fraction"], 0.0)

    def test_posterior_hmc_rejects_nondivisible_sample_budget(self) -> None:
            model = LinearGaussianVI().to(dtype=torch.float64)
            z = torch.zeros(2, 1, dtype=torch.float64)
            with self.assertRaisesRegex(ValueError, "divisible"):
                posterior_hmc_reference_scores(
                    model,
                    z,
                    z.clone(),
                    total_samples=101,
                    num_chains=4,
                    burn_in_steps=1,
                    thinning=1,
                    step_size=0.05,
                    leapfrog_steps=1,
                    init_jitter_scale=0.0,
                    adapt_step_size=False,
                    target_acceptance=0.8,
                    adaptation_rate=0.0,
                    min_step_size=0.01,
                    max_step_size=0.1,
                    divergence_threshold=1000.0,
                )

    def test_posterior_hmc_conditional_gaussian_cpu_smoke(self) -> None:
            model = make_model(dtype=torch.float32)
            generating_epsilon, z = model.sampling(num=8)
            chain_scores, diagnostics = posterior_hmc_reference_scores(
                model,
                z,
                generating_epsilon,
                total_samples=40,
                num_chains=4,
                burn_in_steps=5,
                thinning=1,
                step_size=0.02,
                leapfrog_steps=2,
                init_jitter_scale=0.01,
                adapt_step_size=True,
                target_acceptance=0.8,
                adaptation_rate=0.1,
                min_step_size=0.001,
                max_step_size=0.05,
                divergence_threshold=1000.0,
                accumulator_dtype=torch.float64,
            )
            self.assertEqual(tuple(chain_scores.shape), (4, 8, 2))
            self.assertTrue(torch.isfinite(chain_scores).all())
            self.assertTrue(
                0.0 <= diagnostics["hmc_acceptance_rate"] <= 1.0
            )

    def test_hmc_quality_checks_warn_without_dropping_metrics(self) -> None:
            quality = OmegaConf.create({
                "max_divergence_fraction": 0.01,
                "max_score_rhat_p95": 1.1,
                "max_epsilon_rhat_p95": 2.0,
                "min_post_burn_acceptance_rate": 0.6,
                "min_worst_chain_acceptance_rate": 0.05,
            })
            diagnostics = {
                "hmc_divergence_fraction": 0.0,
                "hmc_score_rhat_p95": 1.2,
                "hmc_epsilon_rhat_p95": 1.5,
                "hmc_post_burn_acceptance_rate": 0.8,
                "hmc_post_burn_acceptance_min": 0.1,
            }
            status, issues = assess_hmc_reference_quality(
                diagnostics,
                quality,
            )
            self.assertEqual(status, "warning")
            self.assertEqual(len(issues), 1)
            self.assertIn("hmc_score_rhat_p95", issues[0])

    def test_native_sivi_score_matches_training_mixture_autograd(self) -> None:
            model = make_model()
            z = torch.randn(4, 2, dtype=torch.float64)
            generating_epsilon = torch.randn(4, 4, dtype=torch.float64)
            auxiliary = torch.randn(5, 4, dtype=torch.float64)
            runner = SimpleNamespace(
                vi_model=model,
                training_reverse_sample_num=5,
            )
            with patch.object(model, "sample_epsilon", return_value=auxiliary):
                actual, diagnostics = native_sivi_score(
                    runner,
                    z,
                    generating_epsilon,
                )

            z_grad = z.detach().clone().requires_grad_(True)
            epsilon_aux = auxiliary.unsqueeze(0).expand(z.shape[0], -1, -1)
            epsilon_all = torch.cat(
                [epsilon_aux, generating_epsilon.unsqueeze(1)],
                dim=1,
            )
            z_all = z_grad.unsqueeze(1).expand(-1, epsilon_all.shape[1], -1)
            log_terms = model.logp(z_all, epsilon_all)
            expected = torch.autograd.grad(
                torch.logsumexp(log_terms, dim=1).sum(),
                z_grad,
            )[0]
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            self.assertEqual(diagnostics["native_auxiliary_samples"], 6)

    def test_native_uivi_acceptance_has_method_specific_key(self) -> None:
            z = torch.randn(3, 2)
            epsilon = torch.randn(3, 4)

            class FakeUIVIVI:

                @staticmethod
                def score(
                    z_aux: torch.Tensor,
                    epsilon_aux: torch.Tensor,
                ) -> torch.Tensor:
                    return z_aux + epsilon_aux[..., :2]

            class FakeUIVIRunner:
                vi_model = FakeUIVIVI()
                training_reverse_sample_num = 5
                hmc_burn_in_steps = 5
                hmc_step_size = 0.2
                hmc_leapfrog_steps = 5

                @staticmethod
                def sample_epsilon_hmc(
                    z_value: torch.Tensor,
                    *,
                    eps_init: torch.Tensor,
                    num_samples: int,
                    burn_in_steps: int,
                    step_size: float,
                    leapfrog_steps: int,
                ) -> tuple[torch.Tensor, torch.Tensor, float]:
                    del burn_in_steps, step_size, leapfrog_steps
                    z_aux = z_value.unsqueeze(1).expand(
                        -1,
                        num_samples,
                        -1,
                    )
                    epsilon_aux = eps_init.unsqueeze(1).expand(
                        -1,
                        num_samples,
                        -1,
                    )
                    return z_aux, epsilon_aux, 0.375

            score, diagnostics = native_uivi_score(
                FakeUIVIRunner(),
                z,
                epsilon,
            )
            self.assertEqual(tuple(score.shape), (3, 2))
            self.assertEqual(diagnostics["native_auxiliary_samples"], 5)
            self.assertAlmostEqual(
                diagnostics["uivi_hmc_acceptance_rate"],
                0.375,
            )
            self.assertNotIn("hmc_acceptance_rate", diagnostics)

    def test_native_aisivi_score_matches_detached_weight_autograd(self) -> None:
            model = make_model()
            n, k = 3, 4
            z = torch.randn(n, 2, dtype=torch.float64)
            epsilon = torch.randn(n, k, 4, dtype=torch.float64)
            log_q_reverse = torch.randn(n, k, dtype=torch.float64)
            reverse = FakeReverse(epsilon, log_q_reverse)
            runner = SimpleNamespace(
                vi_model=model,
                reverse_model=reverse,
                training_reverse_sample_num=k,
                normalize_reverse_score=False,
            )
            actual, diagnostics = native_aisivi_score(runner, z)

            with torch.no_grad():
                importance = (
                    model.log_q_epsilon(epsilon) - log_q_reverse
                ).clamp(max=10.0)
            z_grad = z.detach().clone().requires_grad_(True)
            z_aux = z_grad.unsqueeze(1).expand(-1, k, -1)
            log_terms = model.logp(z_aux, epsilon) + importance
            expected = torch.autograd.grad(
                torch.logsumexp(log_terms, dim=1).sum(),
                z_grad,
            )[0]
            torch.testing.assert_close(
                actual,
                expected,
                rtol=1.0e-10,
                atol=1.0e-10,
            )
            self.assertEqual(diagnostics["native_auxiliary_samples"], k)

    def test_native_aisivi_adaptively_splits_failed_z_chunks(self) -> None:
            model = make_model()
            z = torch.randn(4, 2, dtype=torch.float64)
            runner = SimpleNamespace(
                vi_model=model,
                reverse_model=BatchLimitedReverse(epsilon_dim=4),
                training_reverse_sample_num=3,
                normalize_reverse_score=False,
            )
            actual, diagnostics = native_aisivi_score(
                runner,
                z,
                z_chunk_size=4,
            )
            self.assertEqual(tuple(actual.shape), (4, 2))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(diagnostics["native_auxiliary_samples"], 3)
            self.assertEqual(
                diagnostics["aisivi_min_effective_z_chunk_size"],
                1,
            )
            self.assertEqual(
                diagnostics["aisivi_adaptive_split_count"],
                3,
            )

    def test_native_aisivi_adaptively_splits_auxiliary_samples(self) -> None:
            model = make_model()
            z = torch.randn(3, 2, dtype=torch.float64)
            runner = SimpleNamespace(
                vi_model=model,
                reverse_model=SampleLimitedReverse(
                    epsilon_dim=4,
                    maximum_samples=2,
                ),
                training_reverse_sample_num=5,
                normalize_reverse_score=False,
            )
            actual, diagnostics = native_aisivi_score(runner, z)
            self.assertEqual(tuple(actual.shape), (3, 2))
            self.assertTrue(torch.isfinite(actual).all())
            self.assertEqual(diagnostics["native_auxiliary_samples"], 5)
            self.assertEqual(
                diagnostics[
                    "aisivi_min_effective_auxiliary_chunk_size"
                ],
                1,
            )
            self.assertEqual(
                diagnostics["aisivi_auxiliary_split_count"],
                2,
            )
