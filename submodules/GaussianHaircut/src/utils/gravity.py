import numpy as np
import torch
import torch.nn.functional as F


def gravity_hemisphere_loss(X, gravity_directions, weighted=True):
    """
    Penalizes strand segments whose tangent vectors are more than 90° away from gravity.

    Args:
        X: Tensor of shape (N, 100, 3) — strand points
        gravity_directions: Tensor of shape (N, 1, 3) — per-strand gravity direction
        weighted: Bool — whether to weight tip segments more

    Returns:
        Scalar loss
    """
    tangents = X[:, 1:] - X[:, :-1]  # (N, 99, 3)
    tangents = tangents / (tangents.norm(dim=-1, keepdim=True) + 1e-8)  # (N, 99, 3)

    gravity = gravity_directions / (gravity_directions.norm(dim=-1, keepdim=True) + 1e-8)  # (N, 1, 3)

    dot = (tangents * gravity).sum(dim=-1)  # (N, 99), cosine of angle

    # Penalize only when angle > 90° (dot < 0)
    penalty = F.relu(-dot)  # (N, 99)

    if weighted:
        weights = torch.linspace(0, 1, steps=penalty.shape[1], device=X.device).view(1, -1)
        penalty = penalty * weights

    return penalty.mean()