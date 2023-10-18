from typing import List, Tuple

from scipy import stats


def ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
) -> Tuple[float, bool]:
    """Two-sample t-test for the means of two independent samples"""

    test_result = stats.ttest_ind(control, experiment)
    p_value = test_result.pvalue
    result = p_value < alpha

    return p_value, bool(result)
