import os
from datetime import datetime
import argparse

import torch

from models.target_models import target_distribution, DEFAULT_BBOX
from utils.mcmc import HMCSampler, HMCConfig
from utils.logging import get_logger, set_file_handler
import time


def parse_args():
    parser = argparse.ArgumentParser(description="HMC Baseline Runner")
    parser.add_argument(
        "--target",
        type=str,
        default="banana",
        help="Target distribution key (see models.target_models)")
    parser.add_argument("--num-samples",
                        type=int,
                        default=5000,
                        help="Number of HMC samples after burn-in")
    parser.add_argument("--burn-in",
                        type=int,
                        default=1000,
                        help="Burn-in iterations")
    parser.add_argument("--step-size",
                        type=float,
                        default=0.05,
                        help="HMC step size")
    parser.add_argument("--num-steps",
                        type=int,
                        default=10,
                        help="Leapfrog steps per proposal")
    parser.add_argument("--thinning",
                        type=int,
                        default=1,
                        help="Record every k-th sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.target not in target_distribution:
        raise ValueError(
            f"Unknown target '{args.target}'. Available: {list(target_distribution.keys())}"
        )

    target = target_distribution[args.target](device=device)

    # Try to infer dimensionality by a dummy input if model exposes z_dim, else fallback to 2
    z_dim = getattr(target, "z_dim", None)
    if z_dim is None:
        # Attempt: many target models accept batch NxD. Use D=2 as default.
        z_dim = 2

    cfg = HMCConfig(
        step_size=args.step_size,
        num_steps=args.num_steps,
        num_samples=args.num_samples,
        burn_in=args.burn_in,
        thinning=args.thinning,
        seed=args.seed,
        device=device,
    )

    # Prepare save paths and logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join("results", "baseline", args.target, timestamp)
    os.makedirs(save_root, exist_ok=True)
    samples_path = os.path.join(save_root, "samples.pt")
    figures_dir = os.path.join(save_root, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    logger = get_logger("mcmc_baseline")
    set_file_handler(save_root, filename="run.log")
    logger.info(
        f"Starting HMC baseline | target={args.target} | device={device}")
    logger.info(
        f"Config: step_size={args.step_size}, num_steps={args.num_steps}, num_samples={args.num_samples}, burn_in={args.burn_in}, thinning={args.thinning}, seed={args.seed}"
    )

    sampler = HMCSampler(logp_fn=target.logp, dim=z_dim, cfg=cfg)
    t0 = time.time()
    samples, acc_rate = sampler.sample(progress_bar=True)
    elapsed = time.time() - t0

    logger.info(
        f"Sampling completed | accept_rate={acc_rate:.3f} | elapsed_sec={elapsed:.3f}"
    )

    torch.save(
        {
            "samples": samples,
            "accept_rate": acc_rate,
            "config": {
                "step_size": args.step_size,
                "num_steps": args.num_steps,
                "num_samples": args.num_samples,
                "burn_in": args.burn_in,
                "thinning": args.thinning,
                "seed": args.seed,
                "device": str(device),
                "z_dim": z_dim,
                "target": args.target,
            }
        }, samples_path)

    print(f"Saved HMC samples to: {samples_path}")
    print(f"Acceptance rate: {acc_rate:.3f}")
    print(f"Elapsed seconds: {elapsed:.3f}")

    # Plot contour with sampled points overlay
    try:
        bbox = DEFAULT_BBOX.get(args.target, [-5, 5, -5, 5])
        target.contour_plot(
            bbox=bbox,
            fnet=None,
            samples=samples,
            save_to_path=os.path.join(figures_dir, "contour.png"),
            quiver=False,
            t=args.num_samples,
        )
        print(
            f"Saved contour plot to: {os.path.join(figures_dir, 'contour.png')}"
        )
    except Exception as e:
        print(f"Plotting failed: {e}")


if __name__ == "__main__":
    main()
