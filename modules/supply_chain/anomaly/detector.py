"""
Supply Chain Anomaly Detection
- Isolation Forest on demand signals
- Statistical Process Control (SPC) z-score alerts
- Root cause attribution via SHAP
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models") / "supply_chain"
ANOMALY_FEATURES = [
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7",
    "rolling_mean_28", "rolling_std_28",
    "dept_mean_sales", "store_mean_sales",
]


class AnomalyDetector:
   
    def __init__(self, contamination: float = 0.05, z_threshold: float = 3.0):
        self.contamination = contamination
        self.z_threshold = z_threshold
        self.iso_forest = IsolationForest(
            contamination=contamination,
            n_estimators=200,
            random_state=42,
        )
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_cols = ANOMALY_FEATURES

    def fit(self, df: pd.DataFrame) -> "AnomalyDetector":
        """Fit on historical demand data."""
        available = [f for f in self.feature_cols if f in df.columns]
        X = df[available].fillna(0)
        X_scaled = self.scaler.fit_transform(X)
        self.iso_forest.fit(X_scaled)
        self.feature_cols = available
        self.is_fitted = True
        logger.info(f"AnomalyDetector fitted on {len(df):,} rows, {len(available)} features")
        return self

    def predict(self, features: pd.DataFrame, current_sales: float) -> dict:
        """
        Predict anomaly for a single row.
        Returns: anomaly_score, is_anomaly, z_score, root_cause_features
        """
        available = [f for f in self.feature_cols if f in features.columns]
        X = features[available].fillna(0)
        X_scaled = self.scaler.transform(X)

        # Isolation Forest score (-1 = anomaly, 1 = normal)
        iso_pred = self.iso_forest.predict(X_scaled)[0]
        iso_score = self.iso_forest.score_samples(X_scaled)[0]  # more negative = more anomalous
        iso_anomaly = iso_pred == -1

        # Z-score against rolling mean
        rolling_mean = features.get("rolling_mean_28", pd.Series([current_sales])).values[0]
        rolling_std = features.get("rolling_std_28", pd.Series([1.0])).values[0]
        z_score = abs(current_sales - rolling_mean) / (rolling_std + 1e-6)
        spc_anomaly = z_score > self.z_threshold

        is_anomaly = bool(iso_anomaly or spc_anomaly)

        # Root cause: features most different from expected
        root_cause = {}
        if is_anomaly:
            rolling_mean_7 = features.get("rolling_mean_7", pd.Series([0])).values[0]
            lag_diff = {
                "lag_7_deviation": round(float(features.get("lag_7", pd.Series([0])).values[0] - rolling_mean_7), 2),
                "lag_28_deviation": round(float(features.get("lag_28", pd.Series([0])).values[0] - rolling_mean_7), 2),
                "dept_deviation": round(float(features.get("dept_mean_sales", pd.Series([0])).values[0] - rolling_mean_7), 2),
            }
            root_cause = {k: v for k, v in lag_diff.items() if abs(v) > 0}

        return {
            "is_anomaly": is_anomaly,
            "anomaly_score": round(float(iso_score), 4),
            "z_score": round(float(z_score), 4),
            "detection_layer": (
                "isolation_forest+spc" if (iso_anomaly and spc_anomaly)
                else "isolation_forest" if iso_anomaly
                else "spc" if spc_anomaly
                else "none"
            ),
            "root_cause_features": root_cause,
            "threshold_z": self.z_threshold,
        }

    def save(self, path: Optional[Path] = None) -> Path:
        path = path or (MODELS_DIR / "anomaly_detector.pkl")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved anomaly detector to {path}")
        return path

    @classmethod
    def load(cls, path: Optional[Path] = None) -> "AnomalyDetector":
        path = path or (MODELS_DIR / "anomaly_detector.pkl")
        return joblib.load(path)


def train_anomaly_detector(df: pd.DataFrame) -> AnomalyDetector:
    """Convenience function: fit + save anomaly detector."""
    detector = AnomalyDetector()
    detector.fit(df)
    detector.save()
    return detector
