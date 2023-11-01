import torch


def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor):
    return torch.sqrt(torch.sum(torch.square(x1 - x2), axis=1))


def triplet_loss(
    anchor: torch.Tensor,
    positive: torch.Tensor,
    negative: torch.Tensor,
    margin: float = 5.0,
) -> torch.Tensor:
    """
    Computes the triplet loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        anchor (torch.Tensor): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (torch.Tensor): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (torch.Tensor): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The triplet loss
    """
    dist_ap = euclidean_distance(anchor, positive)
    dist_an = euclidean_distance(anchor, negative)

    loss = torch.maximum(
        torch.zeros(size=(dist_ap.shape[0],)), dist_ap - dist_an + margin
    )

    return torch.mean(loss)
