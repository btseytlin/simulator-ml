import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


class GradientBoostingRegressor:
    def __init__(
        self,
        n_estimators=100,
        learning_rate=0.1,
        max_depth=3,
        min_samples_split=2,
        loss="mse",
        verbose=False,
    ):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.verbose = verbose

        self.loss = loss
        self._loss_fn = self.get_loss_function(loss)
        self.trees_ = []

    def get_loss_function(self, loss):
        if loss == "mse":
            return self._mse
        return loss

    def _mse(self, y_true, y_pred):
        loss = np.mean((y_true - y_pred) ** 2)
        grad = y_pred - y_true
        return loss, grad

    def fit(self, X, y):
        """
        Fit the model to the data.

        Args:
            X: array-like of shape (n_samples, n_features)
            y: array-like of shape (n_samples,)

        Returns:
            GradientBoostingRegressor: The fitted model.
        """
        self.base_pred_ = np.mean(y)
        self.trees_ = []

        for i in range(self.n_estimators):
            y_pred = self.predict(X)
            loss, grad = self._loss_fn(y, y_pred)
            if self.verbose:
                print(loss)
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
            )
            tree.fit(X, -grad)
            self.trees_.append(tree)
        return self

    def predict(self, X):
        """Predict the target of new data.

        Args:
            X: array-like of shape (n_samples, n_features)

        Returns:
            y: array-like of shape (n_samples,)
            The predict values.

        """

        predictions = np.full(X.shape[0], self.base_pred_)
        for tree in self.trees_:
            tree_predictions = tree.predict(X)
            predictions = predictions + self.learning_rate * tree_predictions
        return predictions
