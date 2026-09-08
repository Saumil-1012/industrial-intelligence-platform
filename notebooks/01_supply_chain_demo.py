"""
Supply Chain Module - Demo Walkthrough
Run with: python notebooks/01_supply_chain_demo.py

Demonstrates the full pipeline end-to-end using synthetic data
(no Kaggle download required). Shows every component output.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

print("=" * 60)
print("  Industrial Intelligence Platform")
print("  Supply Chain Module - Demo")
print("=" * 60)

#  1. Generate synthetic M5-shaped data\
print("\n[1/6] Generating synthetic supply chain data...")
np.random.seed(42)
n_items, n_days = 50, 100
long_rows = []
for i in range(n_items):
    for d in range(n_days):
        long_rows.append({
            "id": f"FOODS_1_{i:03d}_CA_1",
            "item_id": i % 10, "dept_id": 0, "cat_id": 0,
            "store_id": i % 3, "state_id": 0,
            "d": f"d_{d+1}",
            "sales": max(0, int(np.random.poisson(30 + 10 * np.sin(d / 7)))),
            "date": pd.Timestamp("2015-01-01") + pd.Timedelta(days=d),
            "wday": d % 7, "month": 1, "year": 2015,
            "event_name_1": None, "event_name_2": None,
        })
df = pd.DataFrame(long_rows)
print(f"   Shape: {df.shape} | Items: {df['id'].nunique()} | Days: {n_days}")

#  2. Feature engineering
print("\n[2/6] Feature engineering...")
from modules.supply_chain.features.feature_engineering import (
    add_lag_features, add_rolling_features,
    add_calendar_features, add_category_aggregations, encode_categoricals,
)
df = add_lag_features(df)
df = add_rolling_features(df)
df = add_calendar_features(df)
df = add_category_aggregations(df)
df = encode_categoricals(df)
df = df.dropna(subset=["lag_7", "lag_14", "lag_28"])
print(f"   Feature shape: {df.shape}")
print(f"   Features: lag_7={df['lag_7'].mean():.1f}, "
      f"rolling_mean_28={df['rolling_mean_28'].mean():.1f}, "
      f"is_weekend={df['is_weekend'].mean():.2f}")
#  3. Data validation 
print("\n[3/6] Great Expectations validation...")
from great_expectations.expectations.m5_sales_suite import FeatureValidator
report = FeatureValidator(df).run_all()
print(f"   {report['passed']}/{report['total_expectations']} checks passed - {report['overall_status']}")

#  4. Train LightGBM 
print("\n[4/6] Training LightGBM (quick demo - 5 trials)...")
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error
import optuna; optuna.logging.set_verbosity(optuna.logging.WARNING)

FEATURES = [c for c in [
    "lag_7","lag_14","lag_28","rolling_mean_7","rolling_std_7",
    "rolling_mean_28","rolling_std_28","dayofweek","month",
    "is_weekend","is_german_holiday","has_event",
    "dept_mean_sales","store_mean_sales","item_id","store_id",
] if c in df.columns]

df_s = df.sort_values("date")
split = int(len(df_s) * 0.8)
X_tr, X_vl = df_s[FEATURES].iloc[:split], df_s[FEATURES].iloc[split:]
y_tr, y_vl = df_s["sales"].iloc[:split],  df_s["sales"].iloc[split:]

# Quick Optuna tune (5 trials for demo speed)
def objective(trial):
    p = {"n_estimators": trial.suggest_int("n_estimators", 100, 300),
         "learning_rate": trial.suggest_float("lr", 0.05, 0.2),
         "num_leaves": trial.suggest_int("leaves", 20, 60),
         "verbosity": -1}
    m = lgb.LGBMRegressor(**p)
    m.fit(X_tr, y_tr, eval_set=[(X_vl, y_vl)],
          callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(-1)])
    return mean_absolute_error(y_vl, m.predict(X_vl))

study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=5, show_progress_bar=False)

best = lgb.LGBMRegressor(**study.best_params, verbosity=-1)
best.fit(X_tr, y_tr)
preds = best.predict(X_vl)
mae  = mean_absolute_error(y_vl, preds)
mape = (abs(y_vl - preds) / (y_vl + 1e-6)).mean() * 100
print(f"   Best trial MAE: {study.best_value:.3f}")
print(f"   Val  MAE: {mae:.3f} | MAPE: {mape:.1f}%")

# Quantile regression P10/P50/P90
p10 = lgb.LGBMRegressor(**study.best_params, objective="quantile", alpha=0.1, verbosity=-1)
p90 = lgb.LGBMRegressor(**study.best_params, objective="quantile", alpha=0.9, verbosity=-1)
p10.fit(X_tr, y_tr); p90.fit(X_tr, y_tr)

sample = X_vl.iloc[:3]
print(f"\n   Example forecast (3 items):")
print(f"   {'Item':>4}  {'P10':>6}  {'P50 (pred)':>10}  {'P90':>6}  {'Actual':>8}")
for i in range(3):
    row = sample.iloc[[i]]
    print(f"   {i+1:>4}  {max(0,p10.predict(row)[0]):>6.1f}  "
          f"{max(0,best.predict(row)[0]):>10.1f}  "
          f"{max(0,p90.predict(row)[0]):>6.1f}  "
          f"{y_vl.iloc[i]:>8.1f}")

# 5. Anomaly detection 
print("\n[5/6] Anomaly detection (Isolation Forest + SPC)...")
from modules.supply_chain.anomaly.detector import AnomalyDetector
detector = AnomalyDetector()
detector.fit(df)

# Test normal and anomalous values
test_cases = [
    ("Normal sales (30)",  30.0),
    ("Spike (200)",       200.0),
    ("Zero sales (0)",     0.0),
]
feat_row = df[detector.feature_cols].dropna().iloc[[0]]
for label, sales_val in test_cases:
    result = detector.predict(feat_row, sales_val)
    flag = "⚠️  ANOMALY" if result["is_anomaly"] else "✅  Normal"
    print(f"   {label:25s} → {flag}  (z={result['z_score']:.2f})")

#  6. SHAP explanation
print("\n[6/6] SHAP feature importance (top 5)...")
try:
    import shap
    explainer = shap.TreeExplainer(best)
    sv = explainer.shap_values(X_vl.iloc[:50])
    mean_abs = pd.Series(
        abs(sv).mean(axis=0), index=FEATURES
    ).sort_values(ascending=False)
    print("   Feature            |  Mean |SHAP|")
    print("   " + "─"*35)
    for feat, val in mean_abs.head(5).items():
        bar = "█" * int(val / mean_abs.max() * 20)
        print(f"   {feat:<20} | {val:6.3f}  {bar}")
except ImportError:
    fi = pd.Series(best.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("   (SHAP not installed — using built-in feature importance)")
    for feat, val in fi.head(5).items():
        bar = "█" * int(val / fi.max() * 20)
        print(f"   {feat:<20} | {val:6.0f}  {bar}")

print("\n" + "=" * 60)
print("  Supply Chain demo complete.")
print("  To train on real M5 data:")
print("    make download-data && make train-supply")
print("=" * 60)
