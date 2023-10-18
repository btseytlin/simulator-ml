from typing import Tuple

import numpy as np
from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
)


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Precision"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tp / (tp + fp)


def recall_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Recall"""
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    return tp / (tp + fn)


def specificity_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Specificity"""
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return tn / (tn + fp)


def pr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_precision: float,
) -> Tuple[float, float]:
    # """Returns threshold and recall (from Precision-Recall Curve)"""
    # precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # max_recall = 0
    # threshold_proba = 0
    # for i in range(len(precision)):
    #     if precision[i] >= min_precision and recall[i] > max_recall:
    #         max_recall = recall[i]
    #         threshold_proba = thresholds[i]

    # return threshold_proba, max_recall
    """Returns threshold and recall (from Precision-Recall Curve)"""
    max_recall = 0
    threshold_proba = 1

    for prob in y_prob:
        y_pred = y_prob >= prob
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        if precision >= min_precision and recall > max_recall:
            max_recall = recall
            threshold_proba = prob

    return threshold_proba, max_recall


def sr_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    min_specificity: float,
) -> Tuple[float, float]:
    """Returns threshold and recall (from Specificity-Recall Curve)"""
    max_recall = 0
    threshold_proba = 1

    for prob in y_prob:
        y_pred = y_prob >= prob
        specificity = specificity_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        if specificity >= min_specificity and recall > max_recall:
            max_recall = recall
            threshold_proba = prob

    return threshold_proba, max_recall


def one_pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    all_probs = [1] + list(y_prob) + [0]

    recalls = np.zeros((len(all_probs),))
    precisions = np.ones((len(all_probs),))

    for i, threshold in enumerate(all_probs):
        y_pred = y_prob >= threshold
        recalls[i] = recall_score(y_true, y_pred)
        precisions[i] = precision_score(y_true, y_pred)
    return recalls, precisions


def pr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Precision-Recall curve and it's (LCB, UCB)"""

    idx = np.arange(len(y_true))

    recalls = []
    precisions = []
    for i in range(n_bootstrap):
        idx_boot = np.random.choice(idx, size=len(idx), replace=True)
        y_true_boot = y_true[idx_boot]
        y_prob_boot = y_prob[idx_boot]
        recalls_boot, precisions_boot = one_pr_curve(y_true_boot, y_prob_boot)
        recalls.append(recalls_boot)
        precisions.append(precisions_boot)

    recalls = np.vstack(recalls)
    precisions = np.vstack(precisions)

    precision_lcb = np.quantile(precisions, (1 - conf) / 2, axis=0)
    precision_ucb = np.quantile(precisions, 1 - (1 - conf) / 2, axis=0)

    recall = np.mean(recalls, axis=0)
    precision = np.mean(precisions, axis=0)

    return recall, precision, precision_lcb, precision_ucb


def sr_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    conf: float = 0.95,
    n_bootstrap: int = 10_000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns Specificity-Recall curve and it's (LCB, UCB)"""
    pass
    return recall, specificity, specificity_lcb, specificity_ucb
