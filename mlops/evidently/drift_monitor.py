"""
Evidently AI -Drift Monitoring
Runs weekly on both supply chain + airline modules.

Checks:
- Data drift: feature distribution shift (PSI / KL-divergence)
- Model performance drift: accuracy/MAE degrading over time
- Auto-triggers Airflow retraining DAG when drift > threshold

Usage:
    python mlops/evidently/drift_monitor.py --module supply_chain
    python mlops/evidently/drift_monitor.py --module airline
"""

import json
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

REPORTS_DIR  = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

DRIFT_THRESHOLD = 0.15   # PSI > 0.15 - trigger retrain

# Core drift computation (pure numpy/pandas -no Evidently dep
# at import time so the API stays fast)

def _psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.10 → no significant change
    PSI 0.10–0.25 → moderate change, monitor
    PSI > 0.25 → significant shift, retrain
    """
    expected = np.array(expected, dtype=float)
    actual   = np.array(actual,   dtype=float)

    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints  = np.unique(breakpoints)

    exp_counts = np.histogram(expected, bins=breakpoints)[0].astype(float)
    act_counts = np.histogram(actual,   bins=breakpoints)[0].astype(float)

    # Add small epsilon only to bins that have counts in either distribution
    # so non-overlapping distributions still show high PSI
    exp_counts = np.where(exp_counts + act_counts > 0, exp_counts + 1e-6, 1e-6)
    act_counts = np.where(exp_counts + act_counts > 0, act_counts + 1e-6, 1e-6)

    exp_pct = exp_counts / exp_counts.sum()
    act_pct = act_counts / act_counts.sum()

    psi = np.sum((act_pct - exp_pct) * np.log((act_pct + 1e-9) / (exp_pct + 1e-9)))
    return float(max(0.0, psi))


def _kl_divergence(p: np.ndarray, q: np.ndarray, bins: int = 20) -> float:
    """KL-divergence between two empirical distributions."""
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    all_data    = np.concatenate([p, q])
    edges       = np.histogram_bin_edges(all_data, bins=bins)
    p_hist      = np.histogram(p, bins=edges)[0].astype(float) + 1e-8
    q_hist      = np.histogram(q, bins=edges)[0].astype(float) + 1e-8
    p_hist     /= p_hist.sum()
    q_hist     /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


class DriftMonitor:
    """
    Computes feature drift + model performance drift between
    a reference dataset (training) and current production window.
    """

    def __init__(
        self,
        module: str,
        drift_threshold: float = DRIFT_THRESHOLD,
    ):
        self.module          = module
        self.drift_threshold = drift_threshold
        self.report_path     = REPORTS_DIR / f"{module}_drift_{datetime.now().strftime('%Y%m%d')}.json"

    def compute_feature_drift(
        self,
        reference_df: pd.DataFrame,
        current_df:   pd.DataFrame,
        feature_cols: list[str],
    ) -> dict:
        """
        Compute PSI per feature between reference and current windows.
        Returns per-feature PSI + overall drift score (mean PSI).
        """
        results  = {}
        drifted  = []

        for col in feature_cols:
            if col not in reference_df.columns or col not in current_df.columns:
                continue
            ref_vals = reference_df[col].dropna().values
            cur_vals = current_df[col].dropna().values
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue

            psi   = _psi(ref_vals, cur_vals)
            kl    = _kl_divergence(ref_vals, cur_vals)
            status = (
                "HIGH"     if psi > 0.25 else
                "MODERATE" if psi > 0.10 else
                "STABLE"
            )
            results[col] = {
                "psi":    round(psi, 4),
                "kl":     round(kl,  4),
                "status": status,
            }
            if psi > self.drift_threshold:
                drifted.append(col)

        overall_psi    = float(np.mean([v["psi"] for v in results.values()])) if results else 0.0
        drift_detected = overall_psi > self.drift_threshold

        return {
            "overall_psi":     round(overall_psi, 4),
            "drift_detected":  drift_detected,
            "drifted_features": drifted,
            "n_features_checked": len(results),
            "feature_scores":  results,
        }

    def compute_performance_drift(
        self,
        reference_metrics: dict,
        current_metrics:   dict,
    ) -> dict:
        """
        Compare model performance metrics between reference and current window.
        Flags degradation > 10% as drift.
        """
        results = {}
        for metric, ref_val in reference_metrics.items():
            if metric not in current_metrics:
                continue
            cur_val    = current_metrics[metric]
            pct_change = (cur_val - ref_val) / (abs(ref_val) + 1e-9)
            degraded   = pct_change > 0.10  # >10% worse

            results[metric] = {
                "reference": round(ref_val, 4),
                "current":   round(cur_val, 4),
                "pct_change": round(pct_change * 100, 2),
                "degraded":  degraded,
            }

        any_degraded = any(v["degraded"] for v in results.values())
        return {
            "performance_drift_detected": any_degraded,
            "metrics": results,
        }

    def run(
        self,
        reference_df:      pd.DataFrame,
        current_df:        pd.DataFrame,
        feature_cols:      list[str],
        reference_metrics: Optional[dict] = None,
        current_metrics:   Optional[dict] = None,
    ) -> dict:
        """
        Full drift report: feature drift + performance drift.
        Saves JSON report + returns structured result.
        """
        logger.info(f"Running drift monitor for module: {self.module}")
        logger.info(f"Reference: {len(reference_df):,} rows | Current: {len(current_df):,} rows")

        feature_report = self.compute_feature_drift(reference_df, current_df, feature_cols)

        perf_report = {}
        if reference_metrics and current_metrics:
            perf_report = self.compute_performance_drift(reference_metrics, current_metrics)

        retrain_triggered = (
            feature_report["drift_detected"] or
            perf_report.get("performance_drift_detected", False)
        )

        report = {
            "module":         self.module,
            "timestamp":      datetime.now().isoformat(),
            "drift_threshold": self.drift_threshold,
            "retrain_triggered": retrain_triggered,
            "feature_drift":  feature_report,
            "performance_drift": perf_report,
            "summary": {
                "overall_psi":     feature_report["overall_psi"],
                "drifted_features": feature_report["drifted_features"],
                "retrain_needed":  retrain_triggered,
            },
        }

        # Save JSON report
        with open(self.report_path, "w") as f:
            json.dump(report, f, indent=2)

        status = "⚠️  DRIFT DETECTED - retrain triggered" if retrain_triggered else "✅ Stable — no retrain needed"
        logger.info(f"Overall PSI: {feature_report['overall_psi']:.4f} | {status}")
        if feature_report["drifted_features"]:
            logger.info(f"Drifted features: {feature_report['drifted_features']}")

        return report

    def trigger_airflow_retrain(self, dag_id: str, api_url: str = "http://localhost:8080") -> bool:
        """
        Trigger an Airflow DAG run via REST API.
        Called automatically when drift > threshold.
        """
        import requests
        try:
            resp = requests.post(
                f"{api_url}/api/v1/dags/{dag_id}/dagRuns",
                json={"conf": {"triggered_by": "evidently_drift_monitor",
                               "module": self.module}},
                auth=("airflow", "airflow"),
                timeout=10,
            )
            if resp.status_code in (200, 201):
                logger.info(f"Airflow DAG '{dag_id}' triggered successfully")
                return True
            else:
                logger.warning(f"Airflow trigger failed: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.warning(f"Could not reach Airflow API: {e}")
            return False


# Convenience runners per module


def run_supply_chain_drift(
    parquet_path: Optional[Path] = None,
    trigger_airflow: bool = False,
) -> dict:
    """
    Run drift monitoring for supply chain module.
    Splits parquet into reference (older half) vs current (recent half).
    """
    from modules.supply_chain.features.feature_engineering import FEATURE_COLS

    parquet_path = parquet_path or Path("feature_store/data/supply_chain_features.parquet")

    if not parquet_path.exists():
        logger.warning(f"Parquet not found: {parquet_path}. Generating synthetic demo data.")
        df = _generate_synthetic_supply_chain(n=2000)
    else:
        df = pd.read_parquet(parquet_path)

    df = df.sort_values("event_timestamp") if "event_timestamp" in df.columns else df
    mid = len(df) // 2
    reference_df = df.iloc[:mid]
    current_df   = df.iloc[mid:]

    feature_cols = [f for f in FEATURE_COLS if f in df.columns]

    monitor = DriftMonitor(module="supply_chain")
    report  = monitor.run(reference_df, current_df, feature_cols)

    if report["retrain_triggered"] and trigger_airflow:
        monitor.trigger_airflow_retrain("supply_chain_weekly_pipeline")

    return report


def run_airline_drift(
    data_path: Optional[Path] = None,
    trigger_airflow: bool = False,
) -> dict:
    """Run drift monitoring for airline module."""
    from modules.airline.features.feature_engineering import AIRLINE_FEATURE_COLS

    if data_path and data_path.exists():
        df = pd.read_parquet(data_path)
    else:
        logger.warning("Airline parquet not found. Using synthetic demo data.")
        df = _generate_synthetic_airline(n=2000)

    mid          = len(df) // 2
    reference_df = df.iloc[:mid]
    current_df   = df.iloc[mid:]
    feature_cols = [f for f in AIRLINE_FEATURE_COLS if f in df.columns]

    monitor = DriftMonitor(module="airline")
    report  = monitor.run(reference_df, current_df, feature_cols)

    if report["retrain_triggered"] and trigger_airflow:
        monitor.trigger_airflow_retrain("airline_weekly_pipeline")

    return report
# Synthetic data generators for demo / CI
def _generate_synthetic_supply_chain(n: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    half = n // 2
    # Reference: stable distribution
    ref = {
        "lag_7":           np.random.normal(45, 8,  half),
        "lag_14":          np.random.normal(43, 7,  half),
        "lag_28":          np.random.normal(42, 9,  half),
        "rolling_mean_7":  np.random.normal(44, 5,  half),
        "rolling_std_7":   np.random.exponential(5, half),
        "rolling_mean_28": np.random.normal(43, 6,  half),
        "rolling_std_28":  np.random.exponential(7, half),
        "dayofweek":       np.random.randint(0, 7,  half),
        "month":           np.random.randint(1, 13, half),
        "is_weekend":      np.random.randint(0, 2,  half),
        "dept_mean_sales": np.random.normal(38, 6,  half),
        "store_mean_sales":np.random.normal(50, 8,  half),
        "event_timestamp": pd.date_range("2015-01-01", periods=half, freq="D"),
    }
    # Current: shifted distribution (simulating drift)
    cur = {
        "lag_7":           np.random.normal(55, 12, half),   # drifted up
        "lag_14":          np.random.normal(43,  7, half),
        "lag_28":          np.random.normal(42,  9, half),
        "rolling_mean_7":  np.random.normal(53,  8, half),   # drifted
        "rolling_std_7":   np.random.exponential(9, half),   # more volatile
        "rolling_mean_28": np.random.normal(43,  6, half),
        "rolling_std_28":  np.random.exponential(7, half),
        "dayofweek":       np.random.randint(0, 7,  half),
        "month":           np.random.randint(1, 13, half),
        "is_weekend":      np.random.randint(0, 2,  half),
        "dept_mean_sales": np.random.normal(38,  6, half),
        "store_mean_sales":np.random.normal(50,  8, half),
        "event_timestamp": pd.date_range("2016-01-01", periods=half, freq="D"),
    }
    ref_df = pd.DataFrame(ref)
    cur_df = pd.DataFrame(cur)
    return pd.concat([ref_df, cur_df], ignore_index=True)


def _generate_synthetic_airline(n: int = 2000) -> pd.DataFrame:
    np.random.seed(42)
    half = n // 2
    ref = {
        "hour_of_day":       np.random.randint(6, 22, half),
        "day_of_week":       np.random.randint(0, 7,  half),
        "rotation_time_min": np.random.normal(60, 20, half),
        "weather_severity":  np.random.beta(2, 8,     half),
        "congestion_tier":   np.random.randint(0, 3,  half).astype(float),
        "route_mean_delay":  np.random.exponential(10, half),
        "carrier_delay_rate":np.random.beta(3, 10,    half),
    }
    cur = {
        "hour_of_day":       np.random.randint(6, 22, half),
        "day_of_week":       np.random.randint(0, 7,  half),
        "rotation_time_min": np.random.normal(40, 25, half),   # tighter rotations
        "weather_severity":  np.random.beta(4, 6,     half),   # worse weather
        "congestion_tier":   np.random.randint(1, 3,  half).astype(float),
        "route_mean_delay":  np.random.exponential(18, half),  # higher delays
        "carrier_delay_rate":np.random.beta(5, 8,     half),
    }
    ref_df = pd.DataFrame(ref)
    cur_df = pd.DataFrame(cur)
    return pd.concat([ref_df, cur_df], ignore_index=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--module", choices=["supply_chain", "airline", "both"],
                        default="both")
    parser.add_argument("--trigger-airflow", action="store_true")
    args = parser.parse_args()

    if args.module in ("supply_chain", "both"):
        report = run_supply_chain_drift(trigger_airflow=args.trigger_airflow)
        print(f"\nSupply Chain | PSI: {report['summary']['overall_psi']} | "
              f"Retrain: {report['summary']['retrain_needed']}")

    if args.module in ("airline", "both"):
        report = run_airline_drift(trigger_airflow=args.trigger_airflow)
        print(f"Airline      | PSI: {report['summary']['overall_psi']} | "
              f"Retrain: {report['summary']['retrain_needed']}")
