from typing import Tuple

import numpy as np
from scipy import stats


def cpc_sample(
    n_samples: int,
    conversion_rate: float,
    reward_avg: float,
    reward_std: float,
) -> np.ndarray:
    """Sample data."""

    sample_conversions = np.random.binomial(
        size=n_samples, p=conversion_rate, n=1
    )
    sample_rewards = np.random.normal(reward_avg, reward_std, n_samples)
    cpc = sample_conversions * sample_rewards
    return cpc


def t_test(
    cpc_a: np.ndarray,
    cpc_b: np.ndarray,
    alpha=0.05,
) -> Tuple[bool, float]:
    """Perform t-test.

    Parameters
    ----------
    cpc_a: np.ndarray :
        first samples
    cpc_b: np.ndarray :
        second samples
    alpha :
         (Default value = 0.05)

    Returns
    -------
    Tuple[bool, float] :
        True if difference is significant, False otherwise
        p-value
    """
    result = stats.ttest_ind(cpc_a, cpc_b)
    p_value = float(result.pvalue)
    verdict = bool(p_value < alpha)
    return verdict, p_value
