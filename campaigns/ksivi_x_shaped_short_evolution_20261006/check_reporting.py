"""Check that initial/dense reporting preserves KSIVI's training trajectory."""

from datetime import datetime
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from runner.ksivi import KSIVIRunner


def main():
    torch.set_num_threads(1)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_states, initial_states = {}, {}
    for name, dense in (("control", False), ("dense", True)):
        root = REPO / "results/ksivi_x_shaped_short_evolution_20261006/local_checks" / timestamp / name
        tb = REPO / "tb_logs/ksivi_x_shaped_short_evolution_20261006/local_checks" / timestamp / name
        canonical = REPO / "configs/ksivi_x_shaped.yaml"
        cfg = OmegaConf.merge(OmegaConf.load(canonical), {
            "seed": 43, "device": "cpu", "use_cuda": False, "config_path": str(canonical),
            "output": {"results_dir": str(root), "tb_dir": str(tb)},
            "train": {"epochs": 12, "log": {"metric_log_freq": 0, "loss_log_freq": 6},
                      "checkpoint": {"freq": 12},
                      "sample": {"freq": 6 if dense else 12, "num": 256, "preserve_rng": dense},
                      "plot": {"freq": 6 if dense else 12, "num": 256, "initial": dense}},
            "metric": {"kl_ite": {"enabled": False}, "w2": {"enabled": False},
                       "elbo": {"enabled": False}},
        })
        torch.manual_seed(cfg.seed)
        runner = KSIVIRunner(cfg)
        runner.log_config()
        initial_states[name] = {key: value.clone() for key, value in runner.vi_model.state_dict().items()}
        if dense:
            rng = torch.get_rng_state().clone()
            epsilon, z = runner._sample_for_reporting(256)
            assert torch.equal(rng, torch.get_rng_state())
            repeated_epsilon, repeated_z = runner._sample_for_reporting(256)
            assert torch.equal(epsilon, repeated_epsilon) and torch.equal(z, repeated_z)
        runner.learn()
        final_states[name] = {key: value.clone() for key, value in runner.vi_model.state_dict().items()}
        if dense:
            run = Path(runner.save_path)
            initial = torch.load(run / "checkpoints/epoch_0/vi_model.pt", weights_only=True)
            assert all(torch.equal(initial[key], value) for key, value in initial_states[name].items())
            for step in (0, 6, 12):
                samples = torch.load(run / f"samples/samples_epoch_{step}.pt", weights_only=True)
                assert samples["epoch"] == step and samples["z"].shape == (256, 2)
                assert torch.isfinite(samples["z"]).all().item()
                assert (run / f"plots/contour_epoch_{step}.png").is_file()
    for states in (initial_states, final_states):
        assert all(torch.equal(value, states["dense"][key]) for key, value in states["control"].items())
    print("PASS: initial samples/checkpoint/plot captured before updates.")
    print("PASS: reporting preserves RNG; dense and control training weights are identical.")


if __name__ == "__main__":
    main()
