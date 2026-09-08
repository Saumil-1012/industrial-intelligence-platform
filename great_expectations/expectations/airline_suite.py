"""
Great Expectations — DOT Airline Data Validation Suite
Validates schema, null checks, value ranges, and business rules
before any airline model training runs.
"""

import json
import logging
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AirlineSuiteValidator:
    def __init__(self, df: pd.DataFrame, dataset_name: str = "dot_airline"):
        self.df      = df
        self.dataset = dataset_name
        self.results = []
        self.passed  = 0
        self.failed  = 0

    def _check(self, name: str, condition: bool, details: str = ""):
        status = "PASS" if condition else "FAIL"
        self.results.append({"expectation": name, "status": status, "details": details})
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            logger.warning(f"[FAIL] {name}: {details}")

    def expect_required_columns(self):
        required = ["FL_DATE", "OP_CARRIER", "ORIGIN", "DEST",
                    "CRS_DEP_TIME", "DEP_DELAY", "ARR_DELAY"]
        missing  = [c for c in required if c not in self.df.columns]
        self._check("expect_required_columns", len(missing) == 0,
                    f"Missing: {missing}" if missing else "All required columns present")

    def expect_no_null_keys(self):
        for col in ["FL_DATE", "ORIGIN", "DEST", "OP_CARRIER"]:
            if col in self.df.columns:
                n = self.df[col].isnull().sum()
                self._check(f"expect_no_nulls_{col}", n == 0,
                            f"{n} nulls" if n > 0 else "No nulls")

    def expect_valid_delay_range(self):
        if "ARR_DELAY" in self.df.columns:
            delays = self.df["ARR_DELAY"].dropna()
            self._check("expect_arr_delay_range",
                        delays.between(-120, 1500).all(),
                        f"Range: [{delays.min():.0f}, {delays.max():.0f}]")

    def expect_valid_dep_time(self):
        if "CRS_DEP_TIME" in self.df.columns:
            times = self.df["CRS_DEP_TIME"].dropna()
            self._check("expect_dep_time_range",
                        times.between(0, 2359).all(),
                        f"Range: [{times.min()}, {times.max()}]")

    def expect_valid_carriers(self):
        if "OP_CARRIER" in self.df.columns:
            n = self.df["OP_CARRIER"].nunique()
            self._check("expect_multiple_carriers", n >= 2,
                        f"Found {n} unique carriers")

    def expect_non_negative_distance(self):
        if "DISTANCE" in self.df.columns:
            neg = (self.df["DISTANCE"].dropna() < 0).sum()
            self._check("expect_non_negative_distance", neg == 0,
                        f"{neg} negative values" if neg > 0 else "All distances ≥ 0")

    def expect_min_row_count(self, min_rows: int = 1000):
        self._check("expect_min_row_count", len(self.df) >= min_rows,
                    f"Rows: {len(self.df)} (min: {min_rows})")

    def expect_reasonable_delay_rate(self):
        if "ARR_DELAY" in self.df.columns:
            rate = (self.df["ARR_DELAY"].fillna(0) >= 15).mean()
            self._check("expect_realistic_delay_rate",
                        0.05 <= rate <= 0.60,
                        f"Delay rate: {rate:.1%} (expected 5%–60%)")

    def run_all(self) -> dict:
        logger.info(f"Running airline validation suite on {len(self.df):,} rows...")
        self.expect_required_columns()
        self.expect_no_null_keys()
        self.expect_valid_delay_range()
        self.expect_valid_dep_time()
        self.expect_valid_carriers()
        self.expect_non_negative_distance()
        self.expect_min_row_count()
        self.expect_reasonable_delay_rate()

        total        = self.passed + self.failed
        success_rate = self.passed / total if total > 0 else 0
        report = {
            "dataset": self.dataset, "total_rows": len(self.df),
            "total_expectations": total, "passed": self.passed,
            "failed": self.failed, "success_rate": round(success_rate, 3),
            "overall_status": "PASS" if self.failed == 0 else "FAIL",
            "results": self.results,
        }
        logger.info(f"Validation: {self.passed}/{total} passed — {report['overall_status']}")
        return report


def validate_airline_data(df: pd.DataFrame) -> dict:
    validator = AirlineSuiteValidator(df)
    report    = validator.run_all()
    if report["overall_status"] == "FAIL":
        raise ValueError(f"Airline data validation FAILED: {report['failed']} checks failed.")
    return report


if __name__ == "__main__":
    # Demo with synthetic DOT-shaped data
    n = 5000
    demo = pd.DataFrame({
        "FL_DATE":       pd.date_range("2015-01-01", periods=n, freq="1h").strftime("%Y-%m-%d"),
        "OP_CARRIER":    np.random.choice(["AA", "UA", "DL", "WN", "B6"], n),
        "TAIL_NUM":      [f"N{np.random.randint(100,999)}" for _ in range(n)],
        "FL_NUM":        np.random.randint(100, 9999, n),
        "ORIGIN":        np.random.choice(["JFK", "LAX", "ORD", "DEN", "ATL"], n),
        "DEST":          np.random.choice(["SFO", "MIA", "BOS", "SEA", "PHX"], n),
        "CRS_DEP_TIME":  np.random.randint(500, 2300, n),
        "DEP_DELAY":     np.random.normal(5, 25, n),
        "ARR_DELAY":     np.random.normal(5, 30, n),
        "CANCELLED":     np.zeros(n),
        "CARRIER_DELAY": np.random.exponential(5, n),
        "WEATHER_DELAY": np.random.exponential(2, n),
        "NAS_DELAY":     np.random.exponential(3, n),
        "SECURITY_DELAY": np.zeros(n),
        "LATE_AIRCRAFT_DELAY": np.random.exponential(8, n),
        "AIR_TIME":      np.random.randint(60, 360, n).astype(float),
        "DISTANCE":      np.random.randint(200, 3000, n).astype(float),
    })
    report = AirlineSuiteValidator(demo).run_all()
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2))
