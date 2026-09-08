"""
Great Expectations — M5 Sales Data Validation Suite
Validates schema, null checks, value ranges, and business rules
before any model training or feature engineering runs.

Run:
    python great_expectations/expectations/m5_sales_suite.py
"""

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class M5SalesValidator:
    """
    Programmatic GE-style validator for M5 sales data.
    Runs expectation suites and returns pass/fail + human-readable report.
    """

    def __init__(self, df: pd.DataFrame, dataset_name: str = "m5_sales"):
        self.df = df
        self.dataset_name = dataset_name
        self.results = []
        self.passed = 0
        self.failed = 0

    def _check(self, name: str, condition: bool, details: str = "") -> bool:
        status = "PASS" if condition else "FAIL"
        self.results.append({
            "expectation": name,
            "status": status,
            "details": details,
        })
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            logger.warning(f"[FAIL] {name}: {details}")
        return condition

    # ---- Schema expectations ----

    def expect_columns_to_exist(self):
        required_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        missing = [c for c in required_cols if c not in self.df.columns]
        self._check(
            "expect_required_id_columns_to_exist",
            len(missing) == 0,
            f"Missing: {missing}" if missing else "All ID columns present",
        )

    def expect_day_columns_to_exist(self):
        day_cols = [c for c in self.df.columns if c.startswith("d_")]
        self._check(
            "expect_day_columns_to_exist",
            len(day_cols) >= 1000,
            f"Found {len(day_cols)} day columns (expected ≥1000)",
        )

    def expect_correct_categories(self):
        if "cat_id" in self.df.columns:
            valid_cats = {"HOBBIES", "HOUSEHOLD", "FOODS"}
            actual_cats = set(self.df["cat_id"].unique())
            unexpected = actual_cats - valid_cats
            self._check(
                "expect_valid_category_ids",
                len(unexpected) == 0,
                f"Valid: {actual_cats}" if not unexpected else f"Unexpected cats: {unexpected}",
            )

    def expect_valid_states(self):
        if "state_id" in self.df.columns:
            valid_states = {"CA", "TX", "WI"}
            actual = set(self.df["state_id"].unique())
            unexpected = actual - valid_states
            self._check(
                "expect_valid_state_ids",
                len(unexpected) == 0,
                f"States: {actual}" if not unexpected else f"Unexpected: {unexpected}",
            )

    # ---- Null checks ----

    def expect_no_null_ids(self):
        for col in ["id", "item_id", "store_id"]:
            if col in self.df.columns:
                null_count = self.df[col].isnull().sum()
                self._check(
                    f"expect_no_nulls_in_{col}",
                    null_count == 0,
                    f"{null_count} nulls found" if null_count > 0 else "No nulls",
                )

    def expect_unique_ids(self):
        if "id" in self.df.columns:
            dupes = self.df["id"].duplicated().sum()
            self._check(
                "expect_unique_row_ids",
                dupes == 0,
                f"{dupes} duplicate ids" if dupes > 0 else "All IDs unique",
            )

    # ---- Value range checks ----

    def expect_sales_non_negative(self):
        day_cols = [c for c in self.df.columns if c.startswith("d_")]
        if day_cols:
            sample_cols = day_cols[:50]  # check first 50 days
            sales_data = self.df[sample_cols].values.flatten()
            neg_count = (sales_data < 0).sum()
            self._check(
                "expect_sales_values_non_negative",
                neg_count == 0,
                f"{neg_count} negative sales values" if neg_count > 0 else "All sales ≥ 0",
            )

    def expect_reasonable_sales_range(self):
        day_cols = [c for c in self.df.columns if c.startswith("d_")]
        if day_cols:
            sample_cols = day_cols[:100]
            max_val = self.df[sample_cols].max().max()
            self._check(
                "expect_sales_max_below_threshold",
                max_val < 10000,
                f"Max sales value: {max_val} (threshold: 10000)",
            )

    def expect_min_row_count(self, min_rows: int = 100):
        self._check(
            "expect_minimum_row_count",
            len(self.df) >= min_rows,
            f"Row count: {len(self.df)} (min: {min_rows})",
        )

    # ---- Business logic checks ----

    def expect_store_item_combinations_valid(self):
        if "store_id" in self.df.columns and "item_id" in self.df.columns:
            n_stores = self.df["store_id"].nunique()
            n_items = self.df["item_id"].nunique()
            self._check(
                "expect_valid_store_item_diversity",
                n_stores >= 1 and n_items >= 1,
                f"Stores: {n_stores}, Items: {n_items}",
            )

    def expect_no_all_zero_items(self):
        day_cols = [c for c in self.df.columns if c.startswith("d_")]
        if day_cols:
            sample_cols = day_cols[:200]
            all_zero = (self.df[sample_cols].sum(axis=1) == 0).sum()
            pct = all_zero / len(self.df)
            self._check(
                "expect_items_not_all_zero",
                pct < 0.05,
                f"{all_zero} all-zero items ({pct:.1%}) — threshold: 5%",
            )

    def run_all(self) -> dict:
        """Run the full expectation suite."""
        logger.info(f"Running M5 validation suite on {len(self.df):,} rows...")

        self.expect_columns_to_exist()
        self.expect_day_columns_to_exist()
        self.expect_correct_categories()
        self.expect_valid_states()
        self.expect_no_null_ids()
        self.expect_unique_ids()
        self.expect_sales_non_negative()
        self.expect_reasonable_sales_range()
        self.expect_min_row_count()
        self.expect_store_item_combinations_valid()
        self.expect_no_all_zero_items()

        total = self.passed + self.failed
        success_rate = self.passed / total if total > 0 else 0

        report = {
            "dataset": self.dataset_name,
            "total_rows": len(self.df),
            "total_expectations": total,
            "passed": self.passed,
            "failed": self.failed,
            "success_rate": round(success_rate, 3),
            "overall_status": "PASS" if self.failed == 0 else "FAIL",
            "results": self.results,
        }

        logger.info(
            f"Validation complete: {self.passed}/{total} passed "
            f"({success_rate:.0%}) — {report['overall_status']}"
        )
        return report


class FeatureValidator:
    """Validates engineered features before model training."""

    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []
        self.failed = 0

    def _check(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        self.results.append({"expectation": name, "status": status, "details": details})
        if not condition:
            self.failed += 1
            logger.warning(f"[FAIL] {name}: {details}")

    def run_all(self) -> dict:
        lag_cols = ["lag_7", "lag_14", "lag_28"]
        for col in lag_cols:
            if col in self.df.columns:
                null_pct = self.df[col].isnull().mean()
                self._check(
                    f"expect_{col}_null_rate_below_30pct",
                    null_pct < 0.30,
                    f"Null rate: {null_pct:.1%}",
                )

        if "sales" in self.df.columns:
            neg = (self.df["sales"] < 0).sum()
            self._check("expect_target_non_negative", neg == 0, f"{neg} negative values")

        expected_features = [
            "lag_7", "lag_14", "lag_28",
            "rolling_mean_7", "rolling_mean_28",
            "dayofweek", "month", "is_weekend",
        ]
        for feat in expected_features:
            self._check(
                f"expect_feature_{feat}_exists",
                feat in self.df.columns,
                "present" if feat in self.df.columns else "MISSING",
            )

        total = len(self.results)
        passed = total - self.failed
        return {
            "total_expectations": total,
            "passed": passed,
            "failed": self.failed,
            "overall_status": "PASS" if self.failed == 0 else "FAIL",
            "results": self.results,
        }


def validate_m5_data(df: pd.DataFrame) -> dict:
    """Convenience: validate raw M5 data, raise on failure."""
    validator = M5SalesValidator(df)
    report = validator.run_all()
    if report["overall_status"] == "FAIL":
        raise ValueError(
            f"M5 data validation FAILED: {report['failed']} expectations failed. "
            "Fix the data before proceeding."
        )
    return report


def validate_features(df: pd.DataFrame) -> dict:
    """Convenience: validate engineered features, raise on failure."""
    validator = FeatureValidator(df)
    report = validator.run_all()
    if report["overall_status"] == "FAIL":
        raise ValueError(
            f"Feature validation FAILED: {report['failed']} expectations failed."
        )
    return report


if __name__ == "__main__":
    # Demo run with synthetic data
    logger.info("Running validation suite on synthetic demo data...")

    n_items = 200
    n_days = 50
    cols = {"id": [f"ITEM_{i}" for i in range(n_items)],
            "item_id": [f"FOODS_1_{i:03d}" for i in range(n_items)],
            "dept_id": ["FOODS_1"] * n_items,
            "cat_id": ["FOODS"] * n_items,
            "store_id": [f"CA_{(i % 10) + 1}" for i in range(n_items)],
            "state_id": ["CA"] * n_items}
    for d in range(1, n_days + 1):
        cols[f"d_{d}"] = np.random.randint(0, 50, n_items)

    demo_df = pd.DataFrame(cols)
    report = M5SalesValidator(demo_df).run_all()
    print(json.dumps(report, indent=2))
