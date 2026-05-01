from __future__ import annotations

import _bootstrap  # noqa: F401

import run_toy_sivi_dsivi_ema_grid as grid


CAMPAIGN_SLUG = "toy_sivi_dsivi_ema_verylowbeta_grid_20260428"
CONFIG_HASH_VERSION = "toy-sivi-dsivi-ema-verylowbeta-grid-effective-v1"


def main() -> None:
    grid.CAMPAIGN_SLUG = CAMPAIGN_SLUG
    grid.CONFIG_HASH_VERSION = CONFIG_HASH_VERSION
    grid.EMA_BETAS = (0.2, 0.4)
    grid.RESULTS_DIR = f"results/{CAMPAIGN_SLUG}"
    grid.TB_DIR = f"tb_logs/{CAMPAIGN_SLUG}"
    grid.main()


if __name__ == "__main__":
    main()
