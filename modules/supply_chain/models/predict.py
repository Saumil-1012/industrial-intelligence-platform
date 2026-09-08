"""
Supply Chain - Inference Module
Loads trained LightGBM models and returns:
- P10/P50/P90 demand forecasts
- SHAP explanations per prediction
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

logger = logging.getLogger(__name__)

MODELS_DIR = Path("models") / "supply_chain"


def load_models() -> tuple[dict, list[str]]:
    """Load saved quantile models + feature list."""
    import joblib

    quantile_models = {}
    for q_label, q_val in [("10", 0.1), ("50", 0.5), ("90", 0.9)]:
        path = MODELS_DIR / f"lgbm_q{q_label}.pkl"
        if path.exists():
            quantile_models[q_val] = joblib.load(path)

    feature_cols_path = MODELS_DIR / "feature_cols.pkl"
    feature_cols = joblib.load(feature_cols_path) if feature_cols_path.exists() else []

    if not quantile_models:
        raise FileNotFoundError(
            f"No models found at {MODELS_DIR}. Run train.py first."
        )

    logger.info(f"Loaded {len(quantile_models)} quantile models")
    return quantile_models, feature_cols


def predict_demand(
    features: pd.DataFrame,
    quantile_models: dict,
    feature_cols: list[str],
) -> dict:
    """
    Run P10/P50/P90 predictions on a feature row.
    Returns dict with forecasts + SHAP values.
    """
    X = features[feature_cols].values

    predictions = {}
    for q, model in quantile_models.items():
        predictions[f"p{int(q*100)}"] = float(max(0, model.predict(X)[0]))

    # SHAP explanation on P50 (point estimate)
    p50_model = quantile_models.get(0.5)
    shap_values = None
    shap_explanation = {}

    if p50_model is not None and SHAP_AVAILABLE:
        try:
            explainer = shap.TreeExplainer(p50_model)
            sv = explainer.shap_values(features[feature_cols])
            shap_values = sv[0] if sv.ndim > 1 else sv

            # Top 5 features by absolute SHAP value
            shap_pairs = sorted(
                zip(feature_cols, shap_values),
                key=lambda x: abs(x[1]),
                reverse=True,
            )[:5]
            shap_explanation = {
                feat: round(float(val), 4) for feat, val in shap_pairs
            }
        except Exception as e:
            logger.warning(f"SHAP failed: {e}")

    return {
        "forecast": predictions,
        "shap_top5": shap_explanation,
        "base_value": float(p50_model.predict(X)[0]) if p50_model else 0.0,
    }


def build_mock_features(feature_cols: list[str]) -> pd.DataFrame:
    """Generate realistic mock feature row for demo/testing."""
    mock = {col: 0.0 for col in feature_cols}

    # Set plausible values
    overrides = {
        "lag_7": 45.0, "lag_14": 42.0, "lag_28": 48.0,
        "rolling_mean_7": 44.0, "rolling_std_7": 5.2, "rolling_max_7": 55.0,
        "rolling_mean_14": 43.5, "rolling_std_14": 6.1, "rolling_max_14": 58.0,
        "rolling_mean_28": 43.0, "rolling_std_28": 7.0, "rolling_max_28": 60.0,
        "dayofweek": 2, "dayofmonth": 15, "weekofyear": 24,
        "month": 6, "year": 2015,
        "is_weekend": 0, "is_month_start": 0, "is_month_end": 0,
        "is_german_holiday": 0, "has_event": 0,
        "price_rel_mean": 1.02, "price_change": -0.01,
        "dept_mean_sales": 38.0, "store_mean_sales": 50.0, "cat_mean_sales": 42.0,
        "item_id": 5, "dept_id": 1, "cat_id": 0, "store_id": 3, "state_id": 1,
    }

    for k, v in overrides.items():
        if k in mock:
            mock[k] = v

    return pd.DataFrame([mock])
