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


def normalized_dcg(
    relevance: List[float], k: int, method: str = "standard"
) -> float:
    """Normalized Discounted Cumulative Gain.

    Parameters
    ----------
    relevance : `List[float]`
        Video relevance list
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values
        `standard` - adds weight to the denominator
        `industry` - adds weights to the numerator and denominator
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """
    dcg = discounted_cumulative_gain(relevance, k, method)

    ideal_dcg = discounted_cumulative_gain(
        sorted(relevance, reverse=True), k, method
    )

    return dcg / ideal_dcg


def avg_ndcg(
    list_relevances: List[List[float]], k: int, method: str = "standard"
) -> float:
    """Average nDCG

    Parameters
    ----------
    list_relevances : `List[List[float]]`
        Video relevance matrix for various queries
    k : `int`
        Count relevance to compute
    method : `str`, optional
        Metric implementation method, takes the values ​​\
        `standard` - adds weight to the denominator\
        `industry` - adds weights to the numerator and denominator\
        `raise ValueError` - for any value

    Returns
    -------
    score : `float`
        Metric score
    """

    ndcg_scores = []
    for relevances in list_relevances:
        ndcg_scores.append(normalized_dcg(relevances, k, method))
    score = np.mean(ndcg_scores)
    return score
