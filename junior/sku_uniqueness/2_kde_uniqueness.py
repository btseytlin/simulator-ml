"""Solution's template for user."""
import numpy as np
from sklearn.neighbors import KernelDensity


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """

    model = KernelDensity()
    model.fit(embeddings)
    scores = model.score_samples(embeddings)
    uniqueness_scores = 1 / np.exp(scores)
    return uniqueness_scores
