"""Baseline for Kaggle AB."""

from copy import copy
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm


def prepare_dataset(DATA_PATH: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare dataset.
    Load data, split into X and y, one-hot encode categorical

    Parameters
    ----------
    DATA_PATH: str :
        path to the dataset

    Returns
    -------
    Tuple[np.ndarray, np.ndarray] :
        X and y
    """
    df = pd.read_csv(DATA_PATH)
    df = df.drop(["ID"], axis=1)
    y = df.pop("y").values

    # select only numeric columns
    X_num = df.select_dtypes(include="number")

    # select only categorical columns and one-hot encode them
    X_cat = df.select_dtypes(exclude="number")
    X_cat = pd.get_dummies(X_cat)

    # combine numeric and categorical
    X = pd.concat([X_num, X_cat], axis=1)
    X = X.fillna(0).values

    return X, y


def cross_val_score(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    cv: int,
    params_list: List[Dict],
    scoring: Callable,
    random_state: int = 42,
    show_progress: bool = False,
) -> np.ndarray:
    """
    Cross-validation score.

    Parameters
    ----------
    model: Callable :
        model to train (e.g. RandomForestRegressor)
    X: np.ndarray :

    y: np.ndarray :

    cv :
        number of folds fo cross-validation

    params_list: List[Dict] :
        list of model parameters

    scoring: Callable :
        scoring function (e.g. r2_score)

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    np.ndarray :
        cross-validation scores [n_models x n_folds]

    """
    scores_folds = []

    idx = np.arange(X.shape[0])
    for params_num, params in enumerate(params_list):
        scores_folds.append([])
        kfold = KFold(n_splits=cv, random_state=random_state, shuffle=True)
        for train_index, test_index in kfold.split(idx):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            fold_model = copy(model)
            fold_model.set_params(**params)
            fold_model.fit(X_train, np.log1p(y_train))
            y_pred = np.expm1(fold_model.predict(X_test))

            score = scoring(y_test, y_pred)
            scores_folds[params_num].append(score)
    metrics = np.array(scores_folds)
    return metrics


def compare_models(
    cv: int,
    model: Callable,
    params_list: List[Dict],
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    show_progress: bool = False,
) -> List[Dict]:
    """Compare models with Cross val.

    Parameters
    ----------
    cv: int :
        number of folds fo cross-validation

    model: Callable :
        model to train (e.g. RandomForestRegressor)

    params_list: List[Dict] :
        list of model parameters

    X: np.ndarray :

    y: np.ndarray :

    random_state: int :
        (Default value = 0)
        random state for cross-validation

    show_progress: bool :
        (Default value = False)

    Returns
    -------
    List[Dict] :
        list of dicts with model comparison results
        {
            model_index,
            avg_score,
            effect_sign
        }
    """
    cv_results = cross_val_score(
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        cv=cv,
        scoring=r2_score,
        random_state=random_state,
        show_progress=show_progress,
    )
    result = []
    for i, model in enumerate(params_list):
        result.append(
            {
                "model_index": i,
                "avg_score": cv_results[i].mean(),
            }
        )

    baseline_score = result[0]["avg_score"]
    for i, model_result in enumerate(result):
        if model_result["avg_score"] == baseline_score:
            model_result["effect_sign"] = 0
        elif model_result["avg_score"] > baseline_score:
            model_result["effect_sign"] = 1
        else:
            model_result["effect_sign"] = -1

    result = result[1:]

    result = sorted(result, key=lambda x: x["avg_score"], reverse=True)
    return result


def run() -> None:
    """Run."""

    data_path = "train.csv.zip"
    random_state = 42
    cv = 5
    params_list = [
        {"max_depth": 10},  # baseline
        {"max_depth": 2},
        {"max_depth": 3},
        {"max_depth": 4},
        {"max_depth": 5},
        {"max_depth": 9},
        {"max_depth": 11},
        {"max_depth": 12},
        {"max_depth": 15},
    ]

    X, y = prepare_dataset(data_path)
    model = RandomForestRegressor(
        n_estimators=200, n_jobs=-1, random_state=random_state
    )

    result = compare_models(
        cv=cv,
        model=model,
        params_list=params_list,
        X=X,
        y=y,
        random_state=random_state,
        show_progress=True,
    )
    print("KFold")
    print(pd.DataFrame(result))


if __name__ == "__main__":
    run()
