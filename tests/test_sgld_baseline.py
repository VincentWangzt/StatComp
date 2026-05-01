from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import torch

from utils.elm import load_baseline_sample_store
from utils.mcmc import SGLDConfig, SGLDSampler


class StandardNormalTarget:
    z_dim = 3

    def score(self, z: torch.Tensor) -> torch.Tensor:
        return -z


class SGLDBaselineTests(unittest.TestCase):
    def test_sgld_sampler_returns_requested_sample_count_from_parallel_chains(self) -> None:
        sampler = SGLDSampler(
            score_fn=StandardNormalTarget().score,
            dim=3,
            cfg=SGLDConfig(
                step_size=1.0e-3,
                num_samples=11,
                burn_in=2,
                thinning=2,
                num_chains=4,
                seed=123,
                device=torch.device("cpu"),
            ),
        )

        samples = sampler.sample()

        self.assertEqual(tuple(samples.shape), (11, 3))
        self.assertTrue(torch.isfinite(samples).all())

    def test_sgld_sampler_is_reproducible_for_same_seed(self) -> None:
        cfg = SGLDConfig(
            step_size=1.0e-3,
            num_samples=8,
            burn_in=1,
            thinning=1,
            num_chains=2,
            seed=9,
            device=torch.device("cpu"),
        )

        first = SGLDSampler(StandardNormalTarget().score, dim=3, cfg=cfg).sample()
        second = SGLDSampler(StandardNormalTarget().score, dim=3, cfg=cfg).sample()

        self.assertTrue(torch.equal(first, second))

    def test_load_baseline_sample_store_accepts_saved_dict_samples(self) -> None:
        expected = torch.arange(12, dtype=torch.float32).reshape(4, 3)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "samples.pt"
            torch.save({"samples": expected, "metadata": {"source": "test"}}, path)

            loaded = load_baseline_sample_store(path)

        self.assertTrue(torch.equal(loaded, expected))


if __name__ == "__main__":
    unittest.main()
