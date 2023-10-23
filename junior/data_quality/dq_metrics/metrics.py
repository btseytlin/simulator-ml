"""Metrics."""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import scipy


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        subdf = df[self.columns]
        n = len(df)
        k = subdf.isnull().agg(self.aggregation, axis=1).sum()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        subdf = df[self.columns]
        n = len(df)
        k = subdf.duplicated().sum()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        subdf = df[self.column]
        n = len(df)
        k = sum(subdf == self.value)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        series = df[self.column]
        n = len(series)
        if self.strict:
            k = sum(series < self.value)
        else:
            k = sum(series <= self.value)

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        series_x = df[self.column_x]
        series_y = df[self.column_y]
        n = len(series_x)
        if self.strict:
            k = sum(series_x < series_y)
        else:
            k = sum(series_x <= series_y)
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        series_x = df[self.column_x]
        series_y = df[self.column_y]
        series_z = df[self.column_z]
        n = len(df)
        if self.strict:
            k = sum((series_x / series_y) < series_z)
        else:
            k = sum((series_x / series_y) <= series_z)

        return {"total": n, "count": k, "delta": k / n}


def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2.0, n - 1)
    return m - h, m + h


@dataclass
class CountCB(Metric):
    """Calculate lower/upper bounds for N%-confidence interval"""

    column: str
    conf: float = 0.95

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        series = df[self.column]

        width = 1 - self.conf
        lcb = series.quantile(0 + width / 2)
        ucb = series.quantile(1 - width / 2)
        return {"lcb": lcb, "ucb": ucb}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"

    def __call__(self, df: pd.DataFrame) -> Dict[str, Any]:
        series = df[self.column]
        today = datetime.datetime.today()
        series_dt = pd.to_datetime(series, format=self.fmt)
        last_day = series_dt.max()
        lag = (today - last_day).days
        return {
            "today": today.strftime(self.fmt),
            "last_day": last_day.strftime(self.fmt),
            "lag": lag,
        }
