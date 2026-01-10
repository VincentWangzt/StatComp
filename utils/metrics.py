import torch
import torch.nn.functional as F


def compute_sliced_wasserstein(
        x1: torch.Tensor,
        x2: torch.Tensor,
        num_projections: int = 1000,
        device: torch.device = torch.device("cpu"),
        p: int = 2,
) -> float:
    """
    Compute the Sliced Wasserstein Distance between two sets of samples.

    Args:
        x1 (torch.Tensor): Samples from the first distribution, shape (N, D).
        x2 (torch.Tensor): Samples from the second distribution, shape (M, D).
        num_projections (int): Number of random projections.
        device (torch.device): Device to perform computations on.
        p (int): The order of the Wasserstein distance.

    Returns:
        float: The estimated Sliced Wasserstein Distance.
    """

    # Ensure data is on the correct device
    x1 = x1.to(device)
    x2 = x2.to(device)

    dim = x1.shape[1]

    # Generate random directions
    projections = torch.randn(dim, num_projections).to(device)
    # L2 normalization
    projections = projections / torch.norm(projections, dim=0, keepdim=True)

    # Project the samples
    x1_projections = torch.matmul(x1, projections)  # (N, num_projections)
    x2_projections = torch.matmul(x2, projections)  # (M, num_projections)

    # Sort the projections
    x1_sorted, _ = torch.sort(x1_projections, dim=0)
    x2_sorted, _ = torch.sort(x2_projections, dim=0)

    # --- CORRECTION: Handle N != M using Interpolation ---
    n = x1_sorted.shape[0]
    m = x2_sorted.shape[0]

    if n != m:
        # We need to interpolate the quantile functions to the same size.
        # We assume x1 and x2 are independent samples from distributions.

        # Determine target size (usually the larger one preserves more info)
        target_size = max(n, m)

        # Helper to interpolate: (Batch, Channels, Length)
        def interpolate_projections(tensor, target_len):
            # Tensor shape: (Len, Projections) -> (1, Projections, Len)
            t = tensor.permute(1, 0).unsqueeze(0)
            # Interpolate
            t = F.interpolate(t,
                              size=target_len,
                              mode='linear',
                              align_corners=True)
            # Reshape back: (Target_Len, Projections)
            return t.squeeze(0).permute(1, 0)

        if n != target_size:
            x1_sorted = interpolate_projections(x1_sorted, target_size)

        if m != target_size:
            x2_sorted = interpolate_projections(x2_sorted, target_size)

    # Calculate L_p distance between sorted projections
    # Now shapes are guaranteed to match
    diff = torch.abs(x1_sorted - x2_sorted)
    distance = torch.pow(diff, p)

    # Average over projections and samples, then take p-th root
    sw_distance = torch.pow(torch.mean(distance), 1.0 / p)

    return sw_distance.item()
