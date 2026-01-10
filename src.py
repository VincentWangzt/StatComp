import torch
import os
import argparse
from utils.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from runner.runners import Runners

# 0107 TODO:
# 1. Clip the CNF scale outputs to avoid extreme values.
# 2. [Restructure] Merge HMC code to only support batch processing. Create HMC sampler class.
# 3. Add annealing to logp calculation [Done]
# 4. Add logging of configs to hParam
# 5. Add W2 distance as metric [Done]
# 6. tweak the lr and scheduler settings, improve the training and inference speed.
# 7. implement dsm-uivi
# 8. Add ELBO, use E_{epsilon^prime ~ q} q_phi(z|epsilon^prime) to estimate q_phi(z). [Done]
# 9. Add typing hints to runners
# 10. [Bugfix] Add logp to ReverseGaussian and ReverseMixureOfGaussian [Done]
# 11. Add conditional diffusion target model support

logger = get_logger()

CONFIG_DIR = os.path.join(os.path.dirname(__file__), 'configs')

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="UIVI Runner")
    default_cfg = os.path.join(CONFIG_DIR, 'reverse_uivi.yaml')

    parser.add_argument(
        '--config',
        type=str,
        default=default_cfg,
        help='Path to main YAML config file',
    )

    parser.add_argument(
        'overrides',
        nargs='*',
        help='Override .yaml config options with key=value pairs',
    )

    cli_args = parser.parse_args()

    main_cfg: DictConfig = OmegaConf.load(cli_args.config)  # type: ignore
    overrides_cfg = OmegaConf.from_dotlist(cli_args.overrides)
    main_cfg = OmegaConf.merge(main_cfg, overrides_cfg)  # type: ignore
    main_cfg.config_path = cli_args.config

    seed = main_cfg.get('seed', 42)

    use_cuda = main_cfg.use_cuda and torch.cuda.is_available()
    if use_cuda:
        device = 'cuda'
        os.environ['CUDA_VISIBLE_DEVICES'] = main_cfg.get(
            'cuda_visible_devices',
            '0',
        )
        torch.cuda.manual_seed_all(seed)
    else:
        device = 'cpu'

    torch.manual_seed(seed)
    main_cfg.device = device

    runner_type = main_cfg.runner_type

    runner = Runners[runner_type](config=main_cfg)
    runner.log_config()
    runner.learn()
