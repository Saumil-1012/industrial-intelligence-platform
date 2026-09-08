"""
Airline — Inference Module
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models") / "airline"

CAUSE_LABELS = {
    "carrier_delay":       "Carrier",
    "weather_delay":       "Weather",
    "nas_delay":           "NAS / ATC",
    "security_delay":      "Security",
    "late_aircraft_delay": "Late Aircraft",
}


def load_models() -> dict:
    import joblib
    models = {}
    paths = {
        "classifier":   MODELS_DIR / "delay_classifier.pkl",
        "regressor":    MODELS_DIR / "delay_regressor.pkl",
        "root_cause":   MODELS_DIR / "root_cause_model.pkl",
        "feature_cols": MODELS_DIR / "feature_cols.pkl",
        "cause_cols":   MODELS_DIR / "cause_cols.pkl",
    }
    for key, path in paths.items():
        if path.exists():
            models[key] = joblib.load(path)
        else:
            logger.warning(f"Model not found: {path}")
    if not models:
        raise FileNotFoundError(f"No airline models at {MODELS_DIR}. Run train.py first.")
    return models


def predict_delay(features: pd.DataFrame, models: dict) -> dict:
    feature_cols = models.get("feature_cols", [])
    clf          = models.get("classifier")
    reg          = models.get("regressor")

    if not feature_cols or clf is None:
        return _demo_delay_prediction()

    X = features[feature_cols]

    raw           = clf["base"].predict_proba(X)[:, 1].reshape(-1, 1)
    delay_prob    = float(clf["platt"].predict_proba(raw)[0, 1])
    delay_minutes = float(max(0, reg.predict(X)[0])) if reg else 0.0

    shap_explanation = {}
    try:
        fi    = clf["base"].feature_importances_
        pairs = sorted(zip(feature_cols, fi), key=lambda x: x[1], reverse=True)[:5]
        total = sum(v for _, v in pairs) + 1e-9
        shap_explanation = {feat: round(float(val / total), 4) for feat, val in pairs}
    except Exception as e:
        logger.warning(f"Feature importance failed: {e}")

    risk = "HIGH" if delay_prob >= 0.7 else "MEDIUM" if delay_prob >= 0.4 else "LOW"

    return {
        "delay_probability":  round(delay_prob, 4),
        "delay_minutes_pred": round(delay_minutes, 1),
        "risk_level":         risk,
        "shap_top5":          shap_explanation,
        "mode":               "model",
    }


def predict_root_cause(features: pd.DataFrame, models: dict) -> dict:
    rc_model     = models.get("root_cause")
    cause_cols   = models.get("cause_cols", [])
    feature_cols = models.get("feature_cols", [])

    if rc_model is None or not cause_cols:
        return _demo_root_cause()

    X     = features[[c for c in feature_cols if c in features.columns]]
    probs = rc_model.predict_proba(X)

    cause_probs = {}
    for i, col in enumerate(cause_cols):
        label = CAUSE_LABELS.get(col, col)
        prob  = float(probs[i][0][1]) if len(probs[i][0]) > 1 else 0.0
        cause_probs[label] = round(prob, 4)

    total            = sum(cause_probs.values()) + 1e-6
    cause_probs_norm = {k: round(v / total, 4) for k, v in cause_probs.items()}
    top_cause        = max(cause_probs, key=cause_probs.get)

    return {
        "cause_probabilities":      cause_probs,
        "cause_probabilities_norm": cause_probs_norm,
        "primary_cause":            top_cause,
        "mode":                     "model",
    }


def build_mock_features(feature_cols: list[str]) -> pd.DataFrame:
    mock = {col: 0.0 for col in feature_cols}
    overrides = {
        "hour_of_day": 8, "day_of_week": 4, "month": 12,
        "day_of_year": 355, "is_weekend": 0, "season": 0,
        "is_peak_hour": 1, "is_holiday_week": 1,
        "rotation_time_min": 35.0, "rotation_risk": 1,
        "origin_hourly_departures": 18, "congestion_tier": 2,
        "route_mean_delay": 12.5, "route_std_delay": 22.0,
        "carrier_delay_rate": 0.22, "carrier_mean_delay": 8.5,
        "weather_severity": 0.3, "has_weather_delay": 1, "has_nas_delay": 0,
        "carrier": 3, "origin": 12, "dest": 47,
        "distance": 850.0, "air_time": 115.0,
    }
    for k, v in overrides.items():
        if k in mock:
            mock[k] = v
    return pd.DataFrame([mock])


def _demo_delay_prediction() -> dict:
    return {
        "delay_probability":  0.72,
        "delay_minutes_pred": 34.5,
        "risk_level":         "HIGH",
        "shap_top5": {
            "rotation_risk": 0.31, "weather_severity": 0.22,
            "is_holiday_week": 0.18, "congestion_tier": 0.14,
            "carrier_delay_rate": 0.11,
        },
        "mode": "demo",
    }


def _demo_root_cause() -> dict:
    return {
        "cause_probabilities": {
            "Late Aircraft": 0.45, "Carrier": 0.25,
            "NAS / ATC": 0.20, "Weather": 0.08, "Security": 0.02,
        },
        "cause_probabilities_norm": {
            "Late Aircraft": 0.45, "Carrier": 0.25,
            "NAS / ATC": 0.20, "Weather": 0.08, "Security": 0.02,
        },
        "primary_cause": "Late Aircraft",
        "mode": "demo",
    }