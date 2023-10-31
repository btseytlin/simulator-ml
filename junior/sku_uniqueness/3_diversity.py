"""Template for user."""
from typing import Tuple

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


def group_diversity(
    embeddings: np.ndarray, threshold: float
) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    uniqueness_scores = kde_uniqueness(embeddings)
    diversity_score = np.mean(uniqueness_scores)
    reject = diversity_score < threshold
    return reject, diversity_score
