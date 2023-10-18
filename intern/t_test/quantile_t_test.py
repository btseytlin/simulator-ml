from typing import List, Tuple

import numpy as np
from scipy.stats import ttest_ind


def quantile_ttest(
    control: List[float],
    experiment: List[float],
    alpha: float = 0.05,
    quantile: float = 0.95,
    n_bootstraps: int = 1000,
) -> Tuple[float, bool]:
    """
    Bootstrapped t-test for quantiles of two samples.
    """
    quantiles_control = []
    quantiles_experiment = []
    for n in range(n_bootstraps):
        idx = np.random.choice(range(len(control)), size=len(control))
        control_bootstrap = np.array(control)[idx]
        experiment_bootstrap = np.array(experiment)[idx]
        quantiles_control.append(np.quantile(control_bootstrap, quantile))
        quantiles_experiment.append(
            np.quantile(experiment_bootstrap, quantile)
        )

    test_result = ttest_ind(quantiles_control, quantiles_experiment)
    p_value = test_result.pvalue
    result = bool(p_value < alpha)
    return p_value, result
