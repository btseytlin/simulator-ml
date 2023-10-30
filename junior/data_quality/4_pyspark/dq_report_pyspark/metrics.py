"""Metrics."""

import datetime
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
import pyspark.sql as ps
import scipy
from pyspark.sql.functions import col, count, isnan
from pyspark.sql.functions import sum as spark_sum
from pyspark.sql.functions import to_timestamp, when


@dataclass
class Metric:
    """Base class for Metric"""

    def __call__(
        self, df: Union[pd.DataFrame, ps.DataFrame]
    ) -> Dict[str, Any]:
        if isinstance(df, pd.DataFrame):
            return self._call_pandas(df)

        if isinstance(df, ps.DataFrame):
            return self._call_pyspark(df)

        msg = (
            f"Not supported type of arg 'df': {type(df)}. "
            "Supported types: pandas.DataFrame, "
            "pyspark.sql.dataframe.DataFrame"
        )
        raise NotImplementedError(msg)

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {}


@dataclass
class CountTotal(Metric):
    """Total number of rows in DataFrame"""

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        return {"total": len(df)}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        return {"total": df.count()}


@dataclass
class CountZeros(Metric):
    """Number of zeros in choosen column"""

    column: str

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        n = len(df)
        k = sum(df[self.column] == 0)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = df.filter(col(self.column) == 0).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountNull(Metric):
    """Number of empty values in choosen columns"""

    columns: List[str]
    aggregation: str = "any"  # either "all", or "any"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        subdf = df[self.columns]
        n = len(df)
        k = subdf.isnull().agg(self.aggregation, axis=1).sum()
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        checks = [
            (col(c).isNull() | isnan(col(c))).cast("int") for c in self.columns
        ]

        if self.aggregation == "any":
            condition = when(sum(checks) >= 1, 1).otherwise(0).cast("int")
        elif self.aggregation == "all":
            condition = (
                when(sum(checks) == len(self.columns), 1)
                .otherwise(0)
                .cast("int")
            )
        else:
            raise ValueError("aggregation must be either 'all' or 'any'")

        k = (
            df.withColumn("is_null_or_nan", condition)
            .select(spark_sum("is_null_or_nan"))
            .collect()[0][0]
        )

        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountDuplicates(Metric):
    """Number of duplicates in choosen columns"""

    columns: List[str]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        subdf = df[self.columns]
        n = len(df)
        k = subdf.duplicated().sum()
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = (
            df.groupBy(df.columns)
            .count()
            .where(col("count") > 1)
            .select(spark_sum("count"))
            .collect()[0]
            .asDict()["sum(count)"]
        )
        k = k or 0
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountValue(Metric):
    """Number of values in choosen column"""

    column: str
    value: Union[str, int, float]

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        subdf = df[self.column]
        n = len(df)
        k = sum(subdf == self.value)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        k = df.filter(col(self.column) == self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowValue(Metric):
    """Number of values below threshold"""

    column: str
    value: float
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        series = df[self.column]
        n = len(series)
        if self.strict:
            k = sum(series < self.value)
        else:
            k = sum(series <= self.value)

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        if self.strict:
            k = df.filter(col(self.column) < self.value).count()
        else:
            k = df.filter(col(self.column) <= self.value).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountBelowColumn(Metric):
    """Count how often column X below Y"""

    column_x: str
    column_y: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        series_x = df[self.column_x]
        series_y = df[self.column_y]
        n = len(series_x)
        if self.strict:
            k = sum(series_x < series_y)
        else:
            k = sum(series_x <= series_y)
        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        if self.strict:
            k = df.filter(col(self.column_x) < col(self.column_y)).count()
        else:
            k = df.filter(col(self.column_x) <= col(self.column_y)).count()
        return {"total": n, "count": k, "delta": k / n}


@dataclass
class CountRatioBelow(Metric):
    """Count how often X / Y below Z"""

    column_x: str
    column_y: str
    column_z: str
    strict: bool = False

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        series_x = df[self.column_x]
        series_y = df[self.column_y]
        series_z = df[self.column_z]
        n = len(df)
        if self.strict:
            k = sum((series_x / series_y) < series_z)
        else:
            k = sum((series_x / series_y) <= series_z)

        return {"total": n, "count": k, "delta": k / n}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        n = df.count()
        df = df.withColumn(
            "ratio_check_strict",
            (col(self.column_x) / col(self.column_y) < col(self.column_z)),
        )
        df = df.withColumn(
            "ratio_check_nonstrict",
            (col(self.column_x) / col(self.column_y) <= col(self.column_z)),
        )

        if self.strict:
            k = df.filter(col("ratio_check_strict").isNotNull()).count()
        else:
            k = df.filter(col("ratio_check_nonstrict").isNotNull()).count()
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

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
        series = df[self.column]

        width = 1 - self.conf
        lcb = series.quantile(0 + width / 2)
        ucb = series.quantile(1 - width / 2)
        return {"lcb": lcb, "ucb": ucb}

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        width = 1 - self.conf
        lcb, ucb = df.approxQuantile(
            self.column,
            [0 + width / 2, 1 - width / 2],
            0,
        )
        return {"lcb": lcb, "ucb": ucb}


@dataclass
class CountLag(Metric):
    """A lag between latest date and today"""

    column: str
    fmt: str = "%Y-%m-%d"
    spark_fmt = "yyyy-MM-dd"

    def _call_pandas(self, df: pd.DataFrame) -> Dict[str, Any]:
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

    def _call_pyspark(self, df: ps.DataFrame) -> Dict[str, Any]:
        from pyspark.sql.functions import max

        today = datetime.datetime.today()
        series = df.select(
            to_timestamp(col(self.column), self.spark_fmt).alias("dt")
        )
        last_day = series.select(max(col("dt"))).collect()[0][0]
        today = datetime.datetime.today()
        lag = (today - last_day).days
        return {
            "today": today.strftime(self.fmt),
            "last_day": last_day.strftime(self.fmt),
            "lag": lag,
        }
