def annealing(
    t: int,
    warm_up_interval: int = 10000,
    anneal: bool = False,
) -> float:
    """
    warmup the log probability during training, linearly from 0.1 to 1.0 over warm_up_interval steps.
    
    Args:
        t (int): current training step
        warm_up_interval (int): number of steps to warm up
        anneal (bool): whether to apply annealing
    Returns:
        float: annealing factor
    """
    if not anneal:
        return 1.0
    else:
        return min(1.0, 0.1 + t / warm_up_interval)
