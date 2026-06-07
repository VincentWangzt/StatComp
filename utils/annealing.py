import math


def annealing(
    t: int,
    warm_up_interval: int = 10000,
    scheme: str = 'linear',
    anneal: bool = False,
) -> float:
    """
    warmup the log probability during training, linearly from 0.1 to 1.0 over warm_up_interval steps.
    
    Args:
        t (int): current training step
        warm_up_interval (int): number of steps to warm up
        scheme (str): annealing scheme to use ('linear' or 'sigmoid')
        anneal (bool): whether to apply annealing
    Returns:
        float: annealing factor
    """
    if not anneal:
        return 1.0
    else:
        progress = min(1.0, t / warm_up_interval)
        if scheme == 'linear':
            return 0.1 + 0.9 * progress  # linear annealing
        elif scheme == 'sigmoid':
            return 0.1 + 0.9 * (1 / (1 + math.exp(-10 * (progress - 0.5))))
        else:
            raise ValueError(f"Unknown annealing scheme: {scheme}")


def mcmc_step_schedule(
    t: int,
    min_steps: int = 1,
    max_steps: int = 10,
    warmup_epochs: int = 10000,
) -> int:
    """Compute the number of MCMC transition steps K at epoch t.

    Linearly ramps K from min_steps to max_steps over warmup_epochs::

        K(t) = min_steps + floor((max_steps - min_steps) * min(1, t / warmup_epochs))

    After warmup_epochs, K stays fixed at max_steps.

    Args:
        t: Current training epoch.
        min_steps: Minimum K (used at start of training).
        max_steps: Maximum K (used after warmup).
        warmup_epochs: Number of epochs over which to linearly ramp K.

    Returns:
        Integer number of MCMC steps for epoch t.
    """
    if warmup_epochs <= 0:
        return max_steps
    progress = min(1.0, t / warmup_epochs)
    return min_steps + int((max_steps - min_steps) * progress)
