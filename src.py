import torch
import os
import argparse
from utils.logging import get_logger
from omegaconf import OmegaConf, DictConfig
from runner.runners import Runners

# 0105 TODO:
# 1. Support for different initial distribution (q_epsilon) in VI model (and reverse model?). [Done for VI]
# 2. Add NaN and Inf detection all over the code.
# 3. Clip the CNF scale outputs to avoid extreme values.
# 4. [Restructure] Create base trainer class for UIVI, implement the other methods as subclasses. [Done]
# 5. [Restructure] Use OmegaConf to manage configurations. Seperate target and model configs. [Done]
# 6. [Restructure] Merge HMC code to only support batch processing. Create HMC sampler class.
# 7. Add annealing to logp calculation
# 8. Integrate AISIVI into the codebase [Done]
# 9. Add logging of configs to tensorboard, text and hParam
# 10. [Restructure] Rewrite the ConditionalRealNVP for better documentation. [Done]
# 11. Add W2 distance as metric
# 12. tweak the lr and scheduler settings, improve the training and inference speed.
# 13. log grad variance and norms
# 14. implement sivi and dsm-uivi
# 15. Add not-ite based kl divergence, directly sample from prior or perhaps use EM / Gaussian .

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
