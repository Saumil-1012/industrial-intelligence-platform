"""
Airline API Router
POST /airline/delay      — Delay probability + SHAP explanation
POST /airline/rootcause  — Root cause attribution (carrier/weather/ATC/aircraft)
POST /airline/cascade    — Cascading delay propagation via NetworkX
GET  /airline/drift      — Drift monitoring status
GET  /airline/model-info — Model metadata
"""

import logging
from typing import Optional

import pandas as pd
from fastapi import APIRouter
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

_models = None
_cascade_detector = None


def get_models():
    global _models
    if _models is None:
        try:
            from modules.airline.models.predict import load_models
            _models = load_models()
        except FileNotFoundError:
            logger.warning("Airline models not found. Running in demo mode.")
            _models = {}
    return _models


def get_cascade_detector():
    global _cascade_detector
    if _cascade_detector is None:
        from modules.airline.cascade.detector import CascadeDetector, build_demo_schedule
        _cascade_detector = CascadeDetector()
        demo_schedule = build_demo_schedule()
        _cascade_detector.build_graph(demo_schedule)
    return _cascade_detector


# --- Request / Response Models ---

class DelayRequest(BaseModel):
    flight_id:               str   = Field(..., json_schema_extra={"example": "AA101"})
    origin:                  str   = Field(..., json_schema_extra={"example": "JFK"})
    dest:                    str   = Field(..., json_schema_extra={"example": "LAX"})
    carrier:                 str   = Field(..., json_schema_extra={"example": "AA"})
    scheduled_dep:           float = Field(..., description="HHMM format e.g. 800 = 08:00", json_schema_extra={"example": 800.0})
    hour_of_day:             Optional[int]   = 8
    day_of_week:             Optional[int]   = 4
    month:                   Optional[int]   = 12
    is_weekend:              Optional[int]   = 0
    is_peak_hour:            Optional[int]   = 1
    is_holiday_week:         Optional[int]   = 0
    rotation_time_min:       Optional[float] = 45.0
    rotation_risk:           Optional[int]   = 0
    origin_hourly_departures: Optional[int]  = 10
    congestion_tier:         Optional[float] = 1.0
    route_mean_delay:        Optional[float] = 8.0
    route_std_delay:         Optional[float] = 15.0
    carrier_delay_rate:      Optional[float] = 0.20
    carrier_mean_delay:      Optional[float] = 7.0
    weather_severity:        Optional[float] = 0.1
    has_weather_delay:       Optional[int]   = 0
    has_nas_delay:           Optional[int]   = 0
    distance:                Optional[float] = 500.0
    air_time:                Optional[float] = 90.0


class RootCauseRequest(BaseModel):
    flight_id:         str   = Field(..., json_schema_extra={"example": "AA101"})
    delay_observed:    float = Field(..., ge=0, json_schema_extra={"example": 45.0})
    hour_of_day:       Optional[int]   = 8
    day_of_week:       Optional[int]   = 4
    month:             Optional[int]   = 12
    weather_severity:  Optional[float] = 0.3
    has_weather_delay: Optional[int]   = 1
    has_nas_delay:     Optional[int]   = 0
    rotation_risk:     Optional[int]   = 1
    carrier_delay_rate: Optional[float] = 0.22
    congestion_tier:   Optional[float] = 1.0


class CascadeRequest(BaseModel):
    delayed_flight_id: str   = Field(..., json_schema_extra={"example": "AA101"})
    delay_minutes:     float = Field(..., ge=0, json_schema_extra={"example": 75.0})


# --- Endpoints ---

@router.post("/delay")
def predict_delay(req: DelayRequest):
    """
    Predict flight delay probability (calibrated) and expected minutes.
    Returns SHAP explanation for top 5 driving features.
    """
    models = get_models()

    if not models:
        from modules.airline.models.predict import _demo_delay_prediction
        result = _demo_delay_prediction()
        return {"flight_id": req.flight_id, "origin": req.origin,
                "dest": req.dest, **result}

    from modules.airline.models.predict import predict_delay, build_mock_features
    feature_cols = models.get("feature_cols", [])
    features = build_mock_features(feature_cols)

    # Override with request values
    overrides = {
        "hour_of_day": req.hour_of_day, "day_of_week": req.day_of_week,
        "month": req.month, "is_weekend": req.is_weekend,
        "is_peak_hour": req.is_peak_hour, "is_holiday_week": req.is_holiday_week,
        "rotation_time_min": req.rotation_time_min, "rotation_risk": req.rotation_risk,
        "origin_hourly_departures": req.origin_hourly_departures,
        "congestion_tier": req.congestion_tier,
        "route_mean_delay": req.route_mean_delay, "route_std_delay": req.route_std_delay,
        "carrier_delay_rate": req.carrier_delay_rate, "carrier_mean_delay": req.carrier_mean_delay,
        "weather_severity": req.weather_severity,
        "has_weather_delay": req.has_weather_delay, "has_nas_delay": req.has_nas_delay,
        "distance": req.distance, "air_time": req.air_time,
    }
    for k, v in overrides.items():
        if v is not None and k in features.columns:
            features[k] = v

    result = predict_delay(features, models)
    return {"flight_id": req.flight_id, "origin": req.origin, "dest": req.dest, **result}


@router.post("/rootcause")
def root_cause(req: RootCauseRequest):
    """
    Attribute observed delay to one of 5 DOT cause categories:
    Carrier / Weather / NAS-ATC / Security / Late Aircraft.
    Returns probability per cause + primary cause.
    """
    models = get_models()

    if not models:
        from modules.airline.models.predict import _demo_root_cause
        result = _demo_root_cause()
        return {"flight_id": req.flight_id, "delay_observed_min": req.delay_observed, **result}

    from modules.airline.models.predict import predict_root_cause, build_mock_features
    feature_cols = models.get("feature_cols", [])
    features = build_mock_features(feature_cols)

    overrides = {
        "hour_of_day": req.hour_of_day, "day_of_week": req.day_of_week,
        "month": req.month, "weather_severity": req.weather_severity,
        "has_weather_delay": req.has_weather_delay, "has_nas_delay": req.has_nas_delay,
        "rotation_risk": req.rotation_risk, "carrier_delay_rate": req.carrier_delay_rate,
        "congestion_tier": req.congestion_tier,
    }
    for k, v in overrides.items():
        if v is not None and k in features.columns:
            features[k] = v

    result = predict_root_cause(features, models)
    return {"flight_id": req.flight_id, "delay_observed_min": req.delay_observed, **result}


@router.post("/cascade")
def cascade_delay(req: CascadeRequest):
    """
    Propagate a delay through the flight network graph.
    Returns all downstream flights affected + estimated new departure times.
    """
    detector = get_cascade_detector()
    result   = detector.propagate_delay(req.delayed_flight_id, req.delay_minutes)
    summary  = detector.get_graph_summary()

    return {
        "source_flight_id":       result.source_flight_id,
        "source_delay_min":       result.source_delay_min,
        "total_affected":         result.total_affected,
        "max_propagated_delay":   result.max_propagated_delay,
        "cascade_depth_reached":  result.cascade_depth_reached,
        "affected_flights":       result.affected_flights,
        "graph_summary":          summary,
    }


@router.get("/drift")
def drift_status():
    """
    Live drift status from latest Evidently AI report.
    Run drift monitor: python -m mlops.evidently.drift_monitor --module airline
    """
    from mlops.evidently.drift_api import drift_status_response
    return drift_status_response("airline")


@router.get("/model-info")
def model_info():
    models = get_models()
    return {
        "models":        list(models.keys()),
        "n_features":    len(models.get("feature_cols", [])),
        "classifier":    "XGBoost + Platt calibration",
        "regressor":     "XGBoost",
        "root_cause":    "MultiOutputClassifier (XGBoost)",
        "cascade":       "NetworkX DiGraph — aircraft rotation propagation",
    }
