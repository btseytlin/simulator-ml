import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    return np.sqrt(np.sum(np.square(x1 - x2), axis=1))


def contrastive_loss(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, margin: float = 5.0
) -> float:
    """
    Computes the contrastive loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        x1 (np.ndarray): Embedding vectors of the
            first objects in the pair (shape: (N, M))
        x2 (np.ndarray): Embedding vectors of the
            second objects in the pair (shape: (N, M))
        y (np.ndarray): Ground truthlabels (1 for similar, 0 for dissimilar)
            (shape: (N,))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The contrastive loss
    """
    dist = euclidean_distance(x1, x2)
    # print("y*dist**2", y * dist**2)
    # print(
    #     "(1 - y) * ((np.max(margin - dist, 0)) ** 2)",
    #     (1 - y) * ((np.max(margin - dist, 0)) ** 2),
    # )
    loss = y * dist**2 + (1 - y) * ((np.maximum(margin - dist, 0)) ** 2)
    loss = float(np.mean(loss))
    return loss
