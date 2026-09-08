"""
Airline Delay Prediction — XGBoost Training
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.multioutput import MultiOutputClassifier

from modules.airline.features.feature_engineering import (
    AIRLINE_FEATURE_COLS,
    DELAY_CAUSES,
    build_airline_features,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models") / "airline"


def get_available_features(df: pd.DataFrame) -> list[str]:
    return [f for f in AIRLINE_FEATURE_COLS if f in df.columns]


def time_based_split(df: pd.DataFrame, val_ratio: float = 0.15):
    df = df.sort_values("flight_date")
    cutoff = int(len(df) * (1 - val_ratio))
    return df.iloc[:cutoff].copy(), df.iloc[cutoff:].copy()


def tune_classifier(X_train, y_train, X_val, y_val, n_trials=30) -> dict:
    def objective(trial):
        params = {
            "n_estimators":          trial.suggest_int("n_estimators", 100, 800),
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":             trial.suggest_int("max_depth", 3, 10),
            "min_child_weight":      trial.suggest_int("min_child_weight", 1, 20),
            "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha":             trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda":            trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "early_stopping_rounds": 30,
            "eval_metric":           "logloss",
            "verbosity":             0,
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return -roc_auc_score(y_val, preds)

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info(f"Best AUC: {-study.best_value:.4f}")
    return study.best_params


def train_delay_classifier(X_train, y_train, X_val, y_val, best_params) -> dict:
    """Returns plain dict — always picklable."""
    params = {
        **best_params,
        "early_stopping_rounds": 50,
        "eval_metric":           "logloss",
        "verbosity":             0,
    }
    base_model = xgb.XGBClassifier(**params)
    base_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    raw_scores = base_model.predict_proba(X_val)[:, 1].reshape(-1, 1)
    platt = LogisticRegression()
    platt.fit(raw_scores, y_val)

    logger.info("Classifier trained + Platt calibrated")
    return {"base": base_model, "platt": platt}


def train_delay_regressor(X_train, y_train, X_val, y_val, best_params) -> xgb.XGBRegressor:
    params = {
        **{k: v for k, v in best_params.items()
           if k not in ["use_label_encoder", "eval_metric", "early_stopping_rounds"]},
        "objective":             "reg:squarederror",
        "early_stopping_rounds": 50,
        "verbosity":             0,
    }
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    logger.info("Delay regressor trained")
    return model


def train_root_cause_classifier(df, feature_cols):
    available_causes = [c for c in DELAY_CAUSES if c in df.columns]
    if not available_causes:
        logger.warning("No delay cause columns found — skipping root cause model")
        return None

    Y = (df[available_causes] > 0).astype(int)
    X = df[feature_cols]

    delayed_mask = df.get("is_delayed", pd.Series(np.ones(len(df)))) == 1
    X_delayed = X[delayed_mask]
    Y_delayed = Y[delayed_mask]

    if len(X_delayed) < 100:
        logger.warning("Too few delayed samples for root cause model")
        return None

    base = xgb.XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        eval_metric="logloss", verbosity=0,
    )
    model = MultiOutputClassifier(base, n_jobs=-1)
    model.fit(X_delayed, Y_delayed)
    logger.info(f"Root cause classifier trained on {len(X_delayed):,} delayed flights")
    return model, available_causes


def evaluate_classifier(model, X, y) -> dict:
    raw         = model["base"].predict_proba(X)[:, 1].reshape(-1, 1)
    preds_proba = model["platt"].predict_proba(raw)[:, 1]
    preds       = (preds_proba >= 0.5).astype(int)
    return {
        "AUC":       round(roc_auc_score(y, preds_proba), 4),
        "F1":        round(f1_score(y, preds, zero_division=0), 4),
        "Precision": round(precision_score(y, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y, preds, zero_division=0), 4),
        "AP":        round(average_precision_score(y, preds_proba), 4),
    }


def evaluate_regressor(model, X, y) -> dict:
    preds = model.predict(X).clip(0)
    return {
        "MAE":  round(mean_absolute_error(y, preds), 4),
        "RMSE": round(float(np.sqrt(np.mean((y - preds) ** 2))), 4),
    }


def train_pipeline(
    sample_frac: float = 1.0,
    years: Optional[list[int]] = None,
    n_trials: int = 30,
    experiment_name: str = "airline_delay_prediction",
):
    mlflow.set_experiment(experiment_name)

    logger.info("Building airline features...")
    df = build_airline_features(years=years, sample_frac=sample_frac)
    feature_cols = get_available_features(df)
    logger.info(f"Using {len(feature_cols)} features")

    train_df, val_df = time_based_split(df)
    logger.info(f"Train: {len(train_df):,} | Val: {len(val_df):,}")

    X_train     = train_df[feature_cols]
    y_cls_train = train_df["is_delayed"]
    y_reg_train = train_df["delay_minutes"]
    X_val       = val_df[feature_cols]
    y_cls_val   = val_df["is_delayed"]
    y_reg_val   = val_df["delay_minutes"]

    with mlflow.start_run(run_name="airline_xgboost"):
        mlflow.log_param("sample_frac", sample_frac)
        mlflow.log_param("n_features",  len(feature_cols))
        mlflow.log_param("train_size",  len(train_df))
        mlflow.log_param("val_size",    len(val_df))
        mlflow.log_param("delay_rate",  round(float(y_cls_train.mean()), 3))

        logger.info("Tuning classifier...")
        best_params = tune_classifier(X_train, y_cls_train, X_val, y_cls_val, n_trials)
        mlflow.log_params(best_params)

        classifier = train_delay_classifier(X_train, y_cls_train, X_val, y_cls_val, best_params)
        regressor  = train_delay_regressor(X_train, y_reg_train, X_val, y_reg_val, best_params)

        rc_result = train_root_cause_classifier(train_df, feature_cols)
        root_cause_model, cause_cols = (rc_result if rc_result else (None, []))

        cls_metrics = evaluate_classifier(classifier, X_val, y_cls_val)
        reg_metrics = evaluate_regressor(regressor, X_val, y_reg_val)

        for k, v in cls_metrics.items():
            mlflow.log_metric(f"classifier_{k}", v)
        for k, v in reg_metrics.items():
            mlflow.log_metric(f"regressor_{k}", v)

        logger.info(f"Classifier metrics: {cls_metrics}")
        logger.info(f"Regressor metrics:  {reg_metrics}")

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(classifier,   MODELS_DIR / "delay_classifier.pkl")
        joblib.dump(regressor,    MODELS_DIR / "delay_regressor.pkl")
        joblib.dump(feature_cols, MODELS_DIR / "feature_cols.pkl")
        if root_cause_model:
            joblib.dump(root_cause_model, MODELS_DIR / "root_cause_model.pkl")
            joblib.dump(cause_cols,       MODELS_DIR / "cause_cols.pkl")

        mlflow.xgboost.log_model(regressor, "delay_regressor")
        logger.info(f"MLflow run_id: {mlflow.active_run().info.run_id}")

    return classifier, regressor, root_cause_model, feature_cols


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=float, default=1.0)
    parser.add_argument("--years", nargs="+", type=int, default=None)
    parser.add_argument("--trials", type=int, default=30)
    args = parser.parse_args()
    train_pipeline(sample_frac=args.sample, years=args.years, n_trials=args.trials)