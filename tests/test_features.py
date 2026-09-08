"""
Unit tests  Feature Engineering
Tests both supply chain and airline feature pipelines
using synthetic data (no Kaggle download required).
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))



# Supply Chain Feature Tests


from modules.supply_chain.features.feature_engineering import (
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    add_category_aggregations,
    encode_categoricals,
    melt_sales,
    FEATURE_COLS,
)


def make_long_df(n_items: int = 10, n_days: int = 60) -> pd.DataFrame:
    """Synthetic long-format supply chain DataFrame."""
    np.random.seed(42)
    rows = []
    for i in range(n_items):
        for d in range(n_days):
            rows.append({
                "id": f"ITEM_{i}", "item_id": i, "dept_id": 0,
                "cat_id": 0, "store_id": i % 3, "state_id": 0,
                "d": f"d_{d+1}", "sales": max(0, np.random.poisson(30)),
                "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=d),
                "wday": d % 7, "month": 1, "year": 2015,
                "event_name_1": None, "event_name_2": None,
            })
    return pd.DataFrame(rows)


class TestLagFeatures:
    def test_lag_columns_created(self):
        df   = make_long_df()
        df   = add_lag_features(df, lags=[7, 14, 28])
        assert "lag_7"  in df.columns
        assert "lag_14" in df.columns
        assert "lag_28" in df.columns

    def test_lag_values_correct(self):
        df  = make_long_df(n_items=1, n_days=40)
        df  = add_lag_features(df, lags=[7])
        row_30 = df[df["d"] == "d_30"].iloc[0]
        row_23 = df[df["d"] == "d_23"].iloc[0]
        assert row_30["lag_7"] == row_23["sales"]

    def test_first_rows_are_nan(self):
        df  = make_long_df(n_items=1, n_days=40)
        df  = add_lag_features(df, lags=[7])
        first_7 = df.head(7)
        assert first_7["lag_7"].isna().all()


class TestRollingFeatures:
    def test_rolling_columns_created(self):
        df = make_long_df()
        df = add_rolling_features(df, windows=[7])
        assert "rolling_mean_7" in df.columns
        assert "rolling_std_7"  in df.columns
        assert "rolling_max_7"  in df.columns

    def test_rolling_mean_non_negative(self):
        df = make_long_df()
        df = add_rolling_features(df)
        assert (df["rolling_mean_7"].dropna() >= 0).all()

    def test_rolling_std_non_negative(self):
        df = make_long_df()
        df = add_rolling_features(df)
        assert (df["rolling_std_7"].dropna() >= 0).all()


class TestCalendarFeatures:
    def test_calendar_columns_created(self):
        df = make_long_df()
        df = add_calendar_features(df)
        for col in ["dayofweek", "month", "year", "is_weekend",
                    "is_german_holiday", "has_event"]:
            assert col in df.columns, f"Missing: {col}"

    def test_is_weekend_binary(self):
        df = make_long_df()
        df = add_calendar_features(df)
        assert set(df["is_weekend"].unique()).issubset({0, 1})

    def test_dayofweek_range(self):
        df = make_long_df()
        df = add_calendar_features(df)
        assert df["dayofweek"].between(0, 6).all()

    def test_month_range(self):
        df = make_long_df()
        df = add_calendar_features(df)
        assert df["month"].between(1, 12).all()


class TestCategoryAggregations:
    def test_aggregation_columns_created(self):
        df = make_long_df()
        df = add_category_aggregations(df)
        assert "dept_mean_sales"  in df.columns
        assert "store_mean_sales" in df.columns
        assert "cat_mean_sales"   in df.columns

    def test_aggregations_non_negative(self):
        df = make_long_df()
        df = add_category_aggregations(df)
        assert (df["dept_mean_sales"]  >= 0).all()
        assert (df["store_mean_sales"] >= 0).all()


class TestEncodeCategoricals:
    def test_encoded_to_integers(self):
        df = make_long_df()
        df = encode_categoricals(df)
        for col in ["item_id", "dept_id", "cat_id", "store_id", "state_id"]:
            if col in df.columns:
                assert pd.api.types.is_integer_dtype(df[col]) or \
                       pd.api.types.is_float_dtype(df[col])



# Airline Feature Tests

from modules.airline.features.feature_engineering import (
    rename_and_clean,
    add_temporal_features,
    add_airport_congestion_features,
    add_weather_severity,
    add_carrier_features,
    AIRLINE_FEATURE_COLS,
)


def make_dot_df(n: int = 500) -> pd.DataFrame:
    """Synthetic DOT-shaped airline DataFrame."""
    np.random.seed(42)
    return pd.DataFrame({
        "FL_DATE":       pd.date_range("2015-01-01", periods=n, freq="1h")
                           .strftime("%Y-%m-%d"),
        "OP_CARRIER":    np.random.choice(["AA", "UA", "DL", "WN"], n),
        "TAIL_NUM":      [f"N{np.random.randint(100,999)}" for _ in range(n)],
        "FL_NUM":        np.random.randint(100, 9999, n),
        "ORIGIN":        np.random.choice(["JFK", "LAX", "ORD", "DEN"], n),
        "DEST":          np.random.choice(["SFO", "MIA", "BOS", "SEA"], n),
        "CRS_DEP_TIME":  np.random.randint(600, 2200, n),
        "DEP_TIME":      np.random.randint(600, 2200, n).astype(float),
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


class TestAirlineClean:
    def test_rename_applies(self):
        raw = make_dot_df()
        df  = rename_and_clean(raw)
        assert "carrier" in df.columns
        assert "origin"  in df.columns
        assert "arr_delay" in df.columns

    def test_cancelled_removed(self):
        raw = make_dot_df()
        raw.loc[:10, "CANCELLED"] = 1.0
        df  = rename_and_clean(raw)
        assert (df.get("cancelled", pd.Series([0])) != 1).all()

    def test_target_columns_created(self):
        df = rename_and_clean(make_dot_df())
        assert "is_delayed"    in df.columns
        assert "delay_minutes" in df.columns

    def test_is_delayed_binary(self):
        df = rename_and_clean(make_dot_df())
        assert set(df["is_delayed"].unique()).issubset({0, 1})

    def test_delay_minutes_non_negative(self):
        df = rename_and_clean(make_dot_df())
        assert (df["delay_minutes"] >= 0).all()


class TestAirlineTemporal:
    def test_temporal_columns_created(self):
        df = rename_and_clean(make_dot_df())
        df = add_temporal_features(df)
        for col in ["hour_of_day", "day_of_week", "month",
                    "is_weekend", "season", "is_peak_hour"]:
            assert col in df.columns, f"Missing: {col}"

    def test_hour_range(self):
        df = rename_and_clean(make_dot_df())
        df = add_temporal_features(df)
        assert df["hour_of_day"].between(0, 23).all()

    def test_season_range(self):
        df = rename_and_clean(make_dot_df())
        df = add_temporal_features(df)
        assert df["season"].between(0, 3).all()


class TestAirlineCongestion:
    def test_congestion_column_created(self):
        df = rename_and_clean(make_dot_df())
        df = add_temporal_features(df)
        df = add_airport_congestion_features(df)
        assert "origin_hourly_departures" in df.columns
        assert "congestion_tier"          in df.columns

    def test_congestion_non_negative(self):
        df = rename_and_clean(make_dot_df())
        df = add_temporal_features(df)
        df = add_airport_congestion_features(df)
        assert (df["origin_hourly_departures"] >= 0).all()


class TestWeatherSeverity:
    def test_weather_severity_range(self):
        df = rename_and_clean(make_dot_df())
        df = add_weather_severity(df)
        assert df["weather_severity"].between(0, 1).all()

    def test_has_weather_binary(self):
        df = rename_and_clean(make_dot_df())
        df = add_weather_severity(df)
        assert set(df["has_weather_delay"].unique()).issubset({0, 1})



# Cascade Detector Tests

from modules.airline.cascade.detector import CascadeDetector, build_demo_schedule


class TestCascadeDetector:
    def setup_method(self):
        self.det = CascadeDetector(buffer_minutes=30)
        self.det.build_graph(build_demo_schedule())

    def test_graph_built(self):
        summary = self.det.get_graph_summary()
        assert summary["total_flights"]     > 0
        assert summary["total_connections"] > 0

    def test_large_delay_propagates(self):
        result = self.det.propagate_delay("AA101", 90.0)
        assert result.total_affected > 0

    def test_small_delay_absorbed(self):
        # 10 min delay < 30 min buffer — should not propagate
        result = self.det.propagate_delay("AA101", 10.0)
        assert result.total_affected == 0

    def test_unknown_flight_returns_empty(self):
        result = self.det.propagate_delay("UNKNOWN999", 100.0)
        assert result.total_affected == 0

    def test_cascade_result_structure(self):
        result = self.det.propagate_delay("AA101", 90.0)
        for flight in result.affected_flights:
            assert "flight_id"              in flight
            assert "propagated_delay_min"   in flight
            assert "cascade_depth"          in flight
            assert flight["propagated_delay_min"] >= 0

    def test_max_propagated_less_than_source(self):
        result = self.det.propagate_delay("AA101", 90.0)
        assert result.max_propagated_delay <= 90.0

# GE Validation Tests
from great_expectations.expectations.m5_sales_suite import M5SalesValidator
from great_expectations.expectations.airline_suite import AirlineSuiteValidator


class TestGESupplyChain:
    def make_valid_m5(self) -> pd.DataFrame:
        n = 200
        cols = {
            "id":       [f"ITEM_{i}" for i in range(n)],
            "item_id":  [f"FOODS_1_{i:03d}" for i in range(n)],
            "dept_id":  ["FOODS_1"] * n,
            "cat_id":   ["FOODS"] * n,
            "store_id": [f"CA_{(i%10)+1}" for i in range(n)],
            "state_id": ["CA"] * n,
        }
        for d in range(1, 1942):   # real M5 has 1941 days
            cols[f"d_{d}"] = np.random.randint(0, 50, n)
        return pd.DataFrame(cols)

    def test_valid_data_passes(self):
        df     = self.make_valid_m5()
        report = M5SalesValidator(df).run_all()
        assert report["overall_status"] == "PASS"

    def test_negative_sales_fails(self):
        df = self.make_valid_m5()
        df["d_1"] = -1
        report = M5SalesValidator(df).run_all()
        assert report["failed"] > 0

    def test_missing_columns_fails(self):
        df = self.make_valid_m5().drop(columns=["id"])
        report = M5SalesValidator(df).run_all()
        assert report["failed"] > 0


class TestGEAirline:
    def make_valid_dot(self, n: int = 2000) -> pd.DataFrame:
        return make_dot_df(n)

    def test_valid_data_passes(self):
        df     = self.make_valid_dot()
        report = AirlineSuiteValidator(df).run_all()
        assert report["overall_status"] == "PASS"

    def test_missing_columns_fails(self):
        df = self.make_valid_dot().drop(columns=["FL_DATE"])
        report = AirlineSuiteValidator(df).run_all()
        assert report["failed"] > 0

# Drift Monitor Tests
from mlops.evidently.drift_monitor import DriftMonitor


class TestDriftMonitor:
    def test_stable_data_no_drift(self):
        np.random.seed(0)
        ref  = pd.DataFrame({"f1": np.random.normal(0, 1, 500),
                              "f2": np.random.normal(5, 2, 500)})
        cur  = pd.DataFrame({"f1": np.random.normal(0, 1, 500),
                              "f2": np.random.normal(5, 2, 500)})
        mon  = DriftMonitor(module="test")
        rep  = mon.compute_feature_drift(ref, cur, ["f1", "f2"])
        assert rep["overall_psi"] < 0.15

    def test_shifted_data_detects_drift(self):
        np.random.seed(0)
        ref  = pd.DataFrame({"f1": np.random.normal(0,  1, 500)})
        cur  = pd.DataFrame({"f1": np.random.normal(3,  1, 500)})  # 3-std shift
        mon  = DriftMonitor(module="test")
        rep  = mon.compute_feature_drift(ref, cur, ["f1"])
        assert rep["drift_detected"] is True

    def test_performance_degradation_detected(self):
        mon = DriftMonitor(module="test")
        rep = mon.compute_performance_drift(
            reference_metrics={"MAE": 5.0},
            current_metrics={"MAE": 8.0},   # 60% worse
        )
        assert rep["performance_drift_detected"] is True

    def test_stable_performance_no_alert(self):
        mon = DriftMonitor(module="test")
        rep = mon.compute_performance_drift(
            reference_metrics={"MAE": 5.0},
            current_metrics={"MAE": 5.2},   # 4% worse — under threshold
        )
        assert rep["performance_drift_detected"] is False
