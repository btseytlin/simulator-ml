import torch


def euclidean_distance(x1: torch.Tensor, x2: torch.Tensor):
    return torch.sqrt(torch.sum(torch.square(x1 - x2), axis=1))


def contrastive_loss(
    x1: torch.Tensor, x2: torch.Tensor, y: torch.Tensor, margin: float = 5.0
) -> torch.Tensor:
    """
    Computes the contrastive loss using pytorch.
    Using Euclidean distance as metric function.

    Args:
        x1 (torch.Tensor): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (torch.Tensor): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (torch.Tensor): Ground truth labels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        torch.Tensor: The contrastive loss
    """
    dist = euclidean_distance(x1, x2)
    # print("y*dist**2", y * dist**2)
    # print(
    #     "(1 - y) * ((np.max(margin - dist, 0)) ** 2)",
    #     (1 - y) * ((np.max(margin - dist, 0)) ** 2),
    # )
    loss = y * dist**2 + (1 - y) * (
        (torch.maximum(margin - dist, torch.zeros(size=(dist.shape[0],)))) ** 2
    )
    loss = torch.mean(loss)
    return loss
