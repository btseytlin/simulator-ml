from typing import Tuple

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.metrics import roc_auc_score


def roc_auc_ci(
    classifier: ClassifierMixin,
    X: np.ndarray,
    y: np.ndarray,
    conf: float = 0.95,
    n_bootstraps: int = 10_000,
) -> Tuple[float, float]:
    """Returns confidence bounds of the ROC-AUC"""

    idx = np.arange(len(y))
    y_pred = classifier.predict_proba(X)[:, 1]

    aucs = []
    for i in range(n_bootstraps):
        idx_boot = np.random.choice(idx, size=len(idx), replace=True)
        y_boot = y[idx_boot]
        y_boot_pred = y_pred[idx_boot]

        try:
            auc = roc_auc_score(y_boot, y_boot_pred)
        except ValueError:
            auc = np.nan
        aucs.append(auc)

    aucs = np.array(aucs)
    aucs = aucs[~np.isnan(aucs)]
    lcb = np.quantile(aucs, (1 - conf) / 2)
    ucb = np.quantile(aucs, 1 - (1 - conf) / 2)
    return (lcb, ucb)
