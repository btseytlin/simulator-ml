"""Solution's template for user."""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    num_neighbors: int :
        number of neighbors to estimate uniqueness

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    model = NearestNeighbors(
        n_neighbors=num_neighbors,
        metric="euclidean",
        algorithm="auto",
    )
    model.fit(embeddings)

    distances, _ = model.kneighbors(
        embeddings, n_neighbors=num_neighbors, return_distance=True
    )

    uniqueness_scores = np.mean(distances, axis=1)

    return uniqueness_scores
