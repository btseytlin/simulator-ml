import numpy as np


def euclidean_distance(x1: np.ndarray, x2: np.ndarray):
    return np.sqrt(np.sum(np.square(x1 - x2), axis=1))


def triplet_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    negative: np.ndarray,
    margin: float = 5.0,
) -> float:
    """
    Computes the triplet loss using numpy.
    Using Euclidean distance as metric function.

    Args:
        anchor (np.ndarray): Embedding vectors of
            the anchor objects in the triplet (shape: (N, M))
        positive (np.ndarray): Embedding vectors of
            the positive objects in the triplet (shape: (N, M))
        negative (np.ndarray): Embedding vectors of
            the negative objects in the triplet (shape: (N, M))
        margin (float): Margin to enforce dissimilar samples to be farther apart than

    Returns:
        float: The triplet loss
    """
    dist_ap = euclidean_distance(anchor, positive)
    dist_an = euclidean_distance(anchor, negative)

    loss = np.maximum(0, dist_ap - dist_an + margin)

    return float(np.mean(loss))
