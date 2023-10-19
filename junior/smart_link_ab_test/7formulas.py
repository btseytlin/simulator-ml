import math

import numpy as np
from scipy import stats
from scipy.stats import norm


def calculate_sample_size(
    reward_avg: float, reward_std: float, mde: float, alpha: float, beta: float
) -> int:
    """Calculate sample size.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    mde: float :
        minimum detectable effect
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    int :
        sample size

    """
    assert mde > 0, "mde should be greater than 0"

    # Implement your solution here

    # Calculate the z-scores for alpha and beta
    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    mde_absolute = mde * reward_avg

    # Calculate sample size using the formula
    sample_size = ((z_alpha + z_beta) ** 2 * 2 * reward_std**2) / (
        mde_absolute**2
    )

    return math.ceil(sample_size)


def calculate_mde(
    reward_std: float,
    sample_size: int,
    alpha: float,
    beta: float,
) -> float:
    """Calculate minimal detectable effect.

    Parameters
    ----------
    reward_avg: float :
        average reward
    reward_std: float :
        standard deviation of reward
    sample_size: int :
        sample size
    alpha: float :
        significance level
    beta: float :
        type 2 error probability

    Returns
    -------
    float :
        minimal detectable effect

    """

    z_alpha = norm.ppf(1 - alpha / 2)
    z_beta = norm.ppf(1 - beta)

    # Calculate the minimal detectable effect using the formula
    mde = (z_alpha + z_beta) * np.sqrt(2) * reward_std / np.sqrt(sample_size)

    return mde
