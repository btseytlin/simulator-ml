import numpy as np


def smape(y_true: np.array, y_pred: np.array) -> float:
    return np.mean(
        2
        * np.abs(y_true - y_pred)
        / ((np.abs(y_true) + np.abs(y_pred) + 1e-30))
    )
