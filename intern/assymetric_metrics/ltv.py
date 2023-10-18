import numpy as np


def ltv_error(y_true: np.array, y_pred: np.array) -> float:
    error = np.maximum((y_true - y_pred) * 0.5, y_pred - y_true).mean()
    return error
