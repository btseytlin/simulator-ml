"""DQ Report."""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import pandas as pd
import pyspark.sql as ps
from user_input.metrics import Metric

LimitType = Dict[str, Tuple[float, float]]
CheckType = Tuple[str, Metric, LimitType]


@dataclass
class Report:
    """DQ report class."""

    checklist: List[CheckType]
    engine: str = "pandas"

    def fit(
        self, tables: Dict[str, Union[pd.DataFrame, ps.DataFrame]]
    ) -> Dict:
        """Calculate DQ metrics and build report."""

        if self.engine == "pandas":
            return self._fit_pandas(tables)

        if self.engine == "pyspark":
            return self._fit_pyspark(tables)

        raise NotImplementedError(
            "Only pandas and pyspark APIs currently supported!"
        )

    def apply_check_to_table(
        self,
        check_class: Metric,
        check_limits: LimitType,
        table: pd.DataFrame,
        table_name: str,
    ):
        check_record = {
            "table_name": table_name,
            "metric": str(check_class),
            "limits": str(check_limits),
            "values": {},
            "status": ".",
            "error": "",
        }

        check_obj = check_class
        check_limits = check_limits
        check_result = None
        try:
            check_result = check_obj(table)
        except Exception as e:
            check_record["status"] = "E"
            check_record["error"] = str(e)

        if check_result:
            try:
                limit_vars = check_limits.keys()
                for limit_var in limit_vars:
                    limit_min = check_limits[limit_var][0]
                    limit_max = check_limits[limit_var][1]

                    assert check_result[limit_var] <= limit_max
                    assert check_result[limit_var] >= limit_min
            except AssertionError as e:
                check_record["status"] = "F"

        check_record["values"] = check_result
        return check_record

    def apply_checklist_to_tables(self, tables: Dict[str, pd.DataFrame]):
        records = []
        for check in self.checklist:
            check_table_name = check[0]
            table = tables[check_table_name]
            record = self.apply_check_to_table(
                check_class=check[1],
                check_limits=check[2],
                table=table,
                table_name=check_table_name,
            )
            records.append(record)
        return records

    def fill_report(self, result_df: pd.DataFrame):
        self.report_["result"] = result_df

        self.report_["total"] = len(result_df)
        self.report_["passed"] = sum(result_df["status"] == ".")
        self.report_["failed"] = sum(result_df["status"] == "F")
        self.report_["errors"] = sum(result_df["status"] == "E")

        self.report_["passed_pct"] = round(
            self.report_["passed"] / self.report_["total"] * 100, 2
        )
        self.report_["failed_pct"] = round(
            self.report_["failed"] / self.report_["total"] * 100, 2
        )
        self.report_["errors_pct"] = round(
            self.report_["errors"] / self.report_["total"] * 100, 2
        )

    def _fit_pandas(self, tables: Dict[str, pd.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: Pandas"""
        self.report_ = {}
        report = self.report_

        # Check if engine supported
        if self.engine != "pandas":
            raise NotImplementedError("Only pandas API currently supported!")

        self.report_[
            "title"
        ] = f"DQ Report for tables {list(sorted(tables.keys()))}"

        checks_records = self.apply_checklist_to_tables(tables)

        result_df = pd.DataFrame.from_records(checks_records)
        self.fill_report(result_df)
        return report

    def _fit_pyspark(self, tables: Dict[str, ps.DataFrame]) -> Dict:
        """Calculate DQ metrics and build report.  Engine: PySpark"""
        # self.report_ = {}
        # report = self.report_

        # ...
        self.engine = "pandas"
        report = self._fit_pandas(tables)
        return report

    def to_str(self) -> None:
        """Convert report to string format."""
        report = self.report_

        msg = (
            "This Report instance is not fitted yet. "
            "Call 'fit' before usong this method."
        )

        assert isinstance(report, dict), msg

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.max_colwidth", 20)
        pd.set_option("display.width", 1000)

        return (
            f"{report['title']}\n\n"
            f"{report['result']}\n\n"
            f"Passed: {report['passed']} ({report['passed_pct']}%)\n"
            f"Failed: {report['failed']} ({report['failed_pct']}%)\n"
            f"Errors: {report['errors']} ({report['errors_pct']}%)\n"
            "\n"
            f"Total: {report['total']}"
        )
