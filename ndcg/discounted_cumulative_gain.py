from typing import List

import numpy as np


def discounted_cumulative_gain(
    relevance: List[float], k: int, method: str = "standard"
) -> float:
    """Discounted Cumulative Gain

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values:
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    relevance = np.array(relevance)
    denominator = np.log2(np.arange(len(relevance)) + 2)

    if method == "standard":
        numerator = relevance
    elif method == "industry":
        numerator = 2**relevance - 1
    else:
        raise ValueError()
    score = np.sum(numerator[:k] / (denominator[:k]))
    return score
