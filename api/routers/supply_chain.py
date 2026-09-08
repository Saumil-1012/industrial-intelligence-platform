"""
Supply Chain API Router
POST /supply/forecast   — 4-week demand forecast with P10/P50/P90 + SHAP
POST /supply/anomaly    — Anomaly detection on recent sales
GET  /supply/drift      — Drift status (stub for now)
GET  /supply/model-info — Model metadata
"""

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter()

# Lazy-load models on first request
_models = None
_feature_cols = None
_anomaly_detector = None


def get_models():
    global _models, _feature_cols
    if _models is None:
        try:
            from modules.supply_chain.models.predict import load_models
            _models, _feature_cols = load_models()
        except FileNotFoundError:
            logger.warning("Models not found. Running in demo mode.")
            _models, _feature_cols = {}, []
    return _models, _feature_cols


def get_anomaly_detector():
    global _anomaly_detector
    if _anomaly_detector is None:
        try:
            from modules.supply_chain.anomaly.detector import AnomalyDetector
            _anomaly_detector = AnomalyDetector.load()
        except Exception:
            logger.warning("Anomaly detector not found. Using untrained instance.")
            from modules.supply_chain.anomaly.detector import AnomalyDetector
            _anomaly_detector = AnomalyDetector()
    return _anomaly_detector


# --- Request / Response Models ---

class ForecastRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "HOBBIES_1_001"})
    location_id: str = Field(..., json_schema_extra={"example": "CA_1"})
    horizon_days: int = Field(default=28, ge=1, le=90)
    # Optional raw features — if not provided, returns demo forecast
    lag_7: Optional[float] = Field(default=45.0)
    lag_14: Optional[float] = Field(default=42.0)
    lag_28: Optional[float] = Field(default=48.0)
    rolling_mean_28: Optional[float] = Field(default=43.0)
    rolling_std_28: Optional[float] = Field(default=7.0)
    is_weekend: Optional[int] = Field(default=0)
    is_german_holiday: Optional[int] = Field(default=0)
    has_event: Optional[int] = Field(default=0)
    price_rel_mean: Optional[float] = Field(default=1.0)


class AnomalyRequest(BaseModel):
    product_id: str = Field(..., json_schema_extra={"example": "HOBBIES_1_001"})
    current_sales: float = Field(..., ge=0, json_schema_extra={"example": 120.0})
    lag_7: Optional[float] = 45.0
    lag_14: Optional[float] = 42.0
    lag_28: Optional[float] = 48.0
    rolling_mean_7: Optional[float] = 44.0
    rolling_std_7: Optional[float] = 5.0
    rolling_mean_28: Optional[float] = 43.0
    rolling_std_28: Optional[float] = 7.0
    dept_mean_sales: Optional[float] = 38.0
    store_mean_sales: Optional[float] = 50.0


# --- Endpoints ---

@router.post("/forecast")
def forecast(req: ForecastRequest):
    """
    4-week demand forecast with P10/P50/P90 confidence bounds.
    Returns SHAP explanation for top features driving the prediction.
    """
    models, feature_cols = get_models()

    if not models:
        # Demo mode: return plausible mock response
        return {
            "product_id": req.product_id,
            "location_id": req.location_id,
            "horizon_days": req.horizon_days,
            "mode": "demo",
            "forecast": {"p10": 38.2, "p50": 45.7, "p90": 54.1},
            "shap_top5": {
                "lag_7": 3.4,
                "rolling_mean_28": 2.1,
                "is_german_holiday": -1.8,
                "has_event": 1.2,
                "rolling_std_28": -0.9,
            },
            "interpretation": "Demand driven primarily by recent week trend (+3.4) and 28-day average (+2.1)",
        }

    # Build feature row from request
    from modules.supply_chain.models.predict import predict_demand, build_mock_features
    features = build_mock_features(feature_cols)

    # Override with request values where provided
    for field in ["lag_7", "lag_14", "lag_28", "rolling_mean_28", "rolling_std_28",
                  "is_weekend", "is_german_holiday", "has_event", "price_rel_mean"]:
        val = getattr(req, field, None)
        if val is not None and field in features.columns:
            features[field] = val

    result = predict_demand(features, models, feature_cols)

    return {
        "product_id": req.product_id,
        "location_id": req.location_id,
        "horizon_days": req.horizon_days,
        "mode": "model",
        **result,
    }


@router.post("/anomaly")
def detect_anomaly(req: AnomalyRequest):
    """
    Detect demand anomalies using Isolation Forest + SPC z-score.
    Returns anomaly score, detection layer, and root cause features.
    """
    detector = get_anomaly_detector()

    # Build feature row
    feature_data = {
        "lag_7": req.lag_7, "lag_14": req.lag_14, "lag_28": req.lag_28,
        "rolling_mean_7": req.rolling_mean_7, "rolling_std_7": req.rolling_std_7,
        "rolling_mean_28": req.rolling_mean_28, "rolling_std_28": req.rolling_std_28,
        "dept_mean_sales": req.dept_mean_sales, "store_mean_sales": req.store_mean_sales,
    }
    features = pd.DataFrame([feature_data])

    if not detector.is_fitted:
        # Unfitted: return rule-based z-score only
        mean = req.rolling_mean_28 or 43.0
        std = req.rolling_std_28 or 7.0
        z = abs(req.current_sales - mean) / (std + 1e-6)
        return {
            "product_id": req.product_id,
            "mode": "spc_only",
            "is_anomaly": z > 3.0,
            "z_score": round(z, 4),
            "note": "Isolation Forest not trained yet. Run train.py first.",
        }

    result = detector.predict(features, req.current_sales)
    return {"product_id": req.product_id, **result}


@router.get("/drift")
def drift_status():
    """
    Live drift status from latest Evidently AI report.
    Run drift monitor: python -m mlops.evidently.drift_monitor --module supply_chain
    """
    from mlops.evidently.drift_api import drift_status_response
    return drift_status_response("supply_chain")


@router.get("/model-info")
def model_info():
    """Metadata about currently loaded models."""
    models, feature_cols = get_models()
    return {
        "quantile_models": list(models.keys()),
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "model_type": "LightGBM + Optuna",
        "quantiles": ["P10", "P50", "P90"],
    }
