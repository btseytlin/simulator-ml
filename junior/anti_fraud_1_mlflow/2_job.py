import os
from typing import Any, Tuple

import fire
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    PrecisionRecallDisplay,
    RocCurveDisplay,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

IDENTIFIER = f'antifraud-{os.environ.get("KCHECKER_USER_USERNAME", "default")}'
TRACKING_URI = os.environ.get("TRACKING_URI")


def recall_at_precision(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_precision: float = 0.95,
) -> float:
    """Compute recall at precision

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_precision (float, optional): Min precision for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    precision, recall, _ = precision_recall_curve(true_labels, pred_scores)
    metric = max(
        [(r, p) for p, r in zip(precision, recall) if p >= min_precision]
    )[0]
    return metric


def recall_at_specificity(
    true_labels: np.ndarray,
    pred_scores: np.ndarray,
    min_specificity: float = 0.95,
) -> float:
    """Compute recall at specificity

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores
        min_specificity (float, optional): Min specificity for recall. Defaults to 0.95.

    Returns:
        float: Metric value
    """

    fpr, tpr, _ = roc_curve(true_labels, pred_scores)
    specificities = 1 - fpr
    metric = max(
        [(r, s) for s, r in zip(specificities, tpr) if s >= min_specificity]
    )[0]
    return metric


def curves(
    true_labels: np.ndarray, pred_scores: np.ndarray
) -> Tuple[np.ndarray]:
    """Return ROC and FPR curves

    Args:
        true_labels (np.ndarray): True labels
        pred_scores (np.ndarray): Target scores

    Returns:
        Tuple[np.ndarray]: ROC and FPR curves
    """

    def fig2numpy(fig: Any) -> np.ndarray:
        fig.canvas.draw()
        img = fig.canvas.buffer_rgba()
        img = np.asarray(img)
        return img

    pr_curve = PrecisionRecallDisplay.from_predictions(
        true_labels, pred_scores
    )
    pr_curve = fig2numpy(pr_curve.figure_)

    roc_curve = RocCurveDisplay.from_predictions(true_labels, pred_scores)
    roc_curve = fig2numpy(roc_curve.figure_)

    return pr_curve, roc_curve


def job(
    train_path: str = "",
    test_path: str = "",
    target: str = "target",
):
    """Model training job

    Args:
        train_path (str): Train dataset path
        test_path (str): Test dataset path
        target (str): Target column name
    """
    mlflow.set_tracking_uri(TRACKING_URI)

    mlflow.set_experiment(IDENTIFIER)
    mlflow.start_run(run_name=IDENTIFIER)
    mlflow.set_tags(tags={"task_type": "anti-fraud", "framework": "sklearn"})

    train_dataset = pd.read_csv(train_path)
    test_dataset = pd.read_csv(test_path)
    model = IsolationForest(n_estimators=100)

    mlflow.log_params(
        params={
            "features": list(train_dataset.drop(target, axis=1).columns),
            "target": target,
            "model_type": str(model.__class__.__name__),
        }
    )

    train_X, train_Y = (
        train_dataset.drop(target, axis=1),
        train_dataset[target],
    )
    test_X, test_Y = test_dataset.drop(target, axis=1), test_dataset[target]

    model.fit(train_X)

    test_targets = test_Y
    pred_scores = -model.score_samples(test_X)

    roc_auc = roc_auc_score(test_targets, pred_scores)
    recall_precision_95 = recall_at_precision(
        test_targets,
        pred_scores,
        min_precision=0.95,
    )
    recall_specificity_95 = recall_at_specificity(
        test_targets,
        pred_scores,
        min_specificity=0.95,
    )
    pr, roc = curves(test_targets, pred_scores)

    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("recall_precision_95", recall_precision_95)
    mlflow.log_metric("recall_specificity_95", recall_specificity_95)

    mlflow.log_artifact(train_path, "data/")
    mlflow.log_artifact(test_path, "data/")

    mlflow.log_image(pr, "metrics/pr.png")
    mlflow.log_image(roc, "metrics/roc.png")

    mlflow.sklearn.log_model(
        model,
        artifact_path=IDENTIFIER,
        registered_model_name=IDENTIFIER,
    )

    mlflow.end_run()


if __name__ == "__main__":
    fire.Fire(job)
