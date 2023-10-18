import numpy as np


def turnover_error(y_true: np.array, y_pred: np.array) -> float:
    error = (((y_pred - y_true) / y_pred) ** 2).mean()
    return error


# e.g.
# Over predict
# y_true = 100
# y_pred = 150
# error = (((150 - 100) / 150) ** 2).mean() = 50/150**2 = 1/3**2 = 1/9

# Under predict
# y_true = 100
# y_pred = 50
# error = ((50 - 100)/50)**2 = -50/50**2 = 1
