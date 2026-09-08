"""
Supply Chain Demand Forecasting - LightGBM Training
- Walk-forward cross validation (respects time order)
- Optuna Bayesian hyperparameter tuning
- Quantile regression for P10/P50/P90 uncertainty bounds
- MLflow experiment tracking
"""

import logging
import warnings
from pathlib import Path
from typing import Optional

import lightgbm as lgb
import mlflow
import mlflow.lightgbm
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from modules.supply_chain.features.feature_engineering import (
    FEATURE_COLS,
    TARGET_COL,
    build_features,
)

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODELS_DIR = Path(__file__).parents[3] / "mlruns"
QUANTILES = [0.1, 0.5, 0.9]


def walk_forward_split(df: pd.DataFrame, n_splits: int = 3, horizon: int = 28):
    """
    Time-series walk-forward splits.
    Each fold: train on all data before cutoff, validate on next `horizon` days.
    """
    dates = sorted(df["date"].unique())
    split_size = len(dates) // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        cutoff_idx = split_size * (i + 2)
        cutoff_date = dates[min(cutoff_idx, len(dates) - horizon - 1)]
        val_end = dates[min(cutoff_idx + horizon, len(dates) - 1)]

        train_mask = df["date"] <= cutoff_date
        val_mask = (df["date"] > cutoff_date) & (df["date"] <= val_end)

        splits.append((df[train_mask], df[val_mask]))
        logger.info(f"Split {i+1}: train until {cutoff_date}, val {cutoff_date} → {val_end}")

    return splits


def get_available_features(df: pd.DataFrame) -> list[str]:
    """Return only feature cols that actually exist in df."""
    return [f for f in FEATURE_COLS if f in df.columns]


def objective(trial: optuna.Trial, train_df: pd.DataFrame, val_df: pd.DataFrame) -> float:
    """Optuna objective: minimize MAE on validation set."""
    params = {
        "objective": "regression_l1",
        "metric": "mae",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 31, 255),
        "max_depth": trial.suggest_int("max_depth", 4, 12),
        "min_child_samples": trial.suggest_int("min_child_samples", 10, 100),
        "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
        "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
        "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
    }

    feature_cols = get_available_features(train_df)

    model = lgb.LGBMRegressor(**params)
    model.fit(
        train_df[feature_cols],
        train_df[TARGET_COL],
        eval_set=[(val_df[feature_cols], val_df[TARGET_COL])],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)],
    )

    preds = model.predict(val_df[feature_cols])
    return mean_absolute_error(val_df[TARGET_COL], preds)


def tune_hyperparameters(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    n_trials: int = 30,
) -> dict:
    """Run Optuna Bayesian search. Returns best params."""
    logger.info(f"Running Optuna tuning ({n_trials} trials)...")
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda trial: objective(trial, train_df, val_df),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    logger.info(f"Best MAE: {study.best_value:.4f} | Params: {study.best_params}")
    return study.best_params


def train_quantile_models(
    train_df: pd.DataFrame,
    best_params: dict,
    feature_cols: list[str],
) -> dict[float, lgb.LGBMRegressor]:
    """Train P10, P50, P90 quantile regression models."""
    models = {}
    for q in QUANTILES:
        logger.info(f"Training quantile model q={q}...")
        params = {
            **best_params,
            "objective": "quantile",
            "alpha": q,
            "metric": "quantile",
            "verbosity": -1,
        }
        model = lgb.LGBMRegressor(**params)
        model.fit(train_df[feature_cols], train_df[TARGET_COL])
        models[q] = model
    return models


def evaluate(model: lgb.LGBMRegressor, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    """Compute MAE, RMSE, MAPE."""
    preds = model.predict(df[feature_cols])
    actuals = df[TARGET_COL].values
    mae = mean_absolute_error(actuals, preds)
    rmse = np.sqrt(mean_squared_error(actuals, preds))
    mape = np.mean(np.abs((actuals - preds) / (actuals + 1e-6))) * 100
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "MAPE": round(mape, 4)}


def train_pipeline(
    sample: bool = False,
    n_trials: int = 30,
    n_cv_splits: int = 3,
    experiment_name: str = "supply_chain_forecasting",
):
    """
    Full training pipeline:
    1. Load + feature engineer
    2. Walk-forward CV
    3. Optuna tuning on last CV split
    4. Train P10/P50/P90 models on full train set
    5. Log everything to MLflow
    """
    mlflow.set_experiment(experiment_name)

    logger.info("Building features...")
    df = build_features(sample=sample)
    feature_cols = get_available_features(df)
    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    splits = walk_forward_split(df, n_splits=n_cv_splits)

    # Use last split for tuning
    train_df, val_df = splits[-1]
    logger.info(f"Train size: {len(train_df):,} | Val size: {len(val_df):,}")

    with mlflow.start_run(run_name="supply_chain_lgbm"):
        # Hyperparameter tuning
        best_params = tune_hyperparameters(train_df, val_df, n_trials=n_trials)
        mlflow.log_params(best_params)
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_param("train_size", len(train_df))
        mlflow.log_param("val_size", len(val_df))
        mlflow.log_param("sample_mode", sample)

        # Walk-forward CV metrics
        cv_maes = []
        for i, (tr, vl) in enumerate(splits):
            params = {**best_params, "objective": "regression_l1", "verbosity": -1}
            m = lgb.LGBMRegressor(**params)
            m.fit(tr[feature_cols], tr[TARGET_COL])
            metrics = evaluate(m, vl, feature_cols)
            cv_maes.append(metrics["MAE"])
            for k, v in metrics.items():
                mlflow.log_metric(f"cv_split_{i+1}_{k}", v)

        mlflow.log_metric("cv_mean_MAE", round(np.mean(cv_maes), 4))
        logger.info(f"CV MAE across {n_cv_splits} splits: {cv_maes} | Mean: {np.mean(cv_maes):.4f}")

        # Train final quantile models on full data
        logger.info("Training final quantile models on full dataset...")
        quantile_models = train_quantile_models(df, best_params, feature_cols)

        # Final eval metrics (P50 = point estimate)
        final_metrics = evaluate(quantile_models[0.5], val_df, feature_cols)
        for k, v in final_metrics.items():
            mlflow.log_metric(f"final_{k}", v)
        logger.info(f"Final val metrics (P50): {final_metrics}")

        # Log models to MLflow
        for q, model in quantile_models.items():
            mlflow.lightgbm.log_model(model, f"model_q{int(q*100)}")

        # Save models locally too
        models_dir = Path("models") / "supply_chain"
        models_dir.mkdir(parents=True, exist_ok=True)
        import joblib
        for q, model in quantile_models.items():
            joblib.dump(model, models_dir / f"lgbm_q{int(q*100)}.pkl")
        joblib.dump(feature_cols, models_dir / "feature_cols.pkl")

        logger.info(f"Models saved to {models_dir}")
        run_id = mlflow.active_run().info.run_id
        logger.info(f"MLflow run_id: {run_id}")

    return quantile_models, feature_cols, final_metrics


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="Use sample data for fast dev")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials")
    args = parser.parse_args()

    train_pipeline(sample=args.sample, n_trials=args.trials)
