"""
Airline Module — Demo Walkthrough
Run with: python notebooks/02_airline_demo.py

Demonstrates XGBoost delay prediction, root cause attribution,
and NetworkX cascade detection using synthetic data.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
import pandas as pd

print("=" * 60)
print("  Industrial Intelligence Platform")
print("  Airline Module — Demo")
print("=" * 60)

#  1. Synthetic DOT data 
print("\n[1/5] Generating synthetic airline data...")
np.random.seed(42)
N = 3000
raw = pd.DataFrame({
    "FL_DATE":       pd.date_range("2015-01-01", periods=N, freq="30min")
                       .strftime("%Y-%m-%d"),
    "OP_CARRIER":    np.random.choice(["AA","UA","DL","WN","B6"], N),
    "TAIL_NUM":      [f"N{np.random.randint(100,999)}" for _ in range(N)],
    "FL_NUM":        np.random.randint(100, 9999, N),
    "ORIGIN":        np.random.choice(["JFK","LAX","ORD","DEN","ATL","SFO"], N),
    "DEST":          np.random.choice(["MIA","BOS","SEA","PHX","LAS","IAH"], N),
    "CRS_DEP_TIME":  np.random.randint(600, 2200, N),
    "DEP_TIME":      np.random.randint(600, 2200, N).astype(float),
    "DEP_DELAY":     np.random.normal(5, 25, N),
    "ARR_DELAY":     np.random.normal(5, 30, N),
    "CANCELLED":     np.zeros(N),
    "CARRIER_DELAY": np.random.exponential(5, N),
    "WEATHER_DELAY": np.random.exponential(3, N),
    "NAS_DELAY":     np.random.exponential(4, N),
    "SECURITY_DELAY":np.zeros(N),
    "LATE_AIRCRAFT_DELAY": np.random.exponential(10, N),
    "AIR_TIME":      np.random.randint(60, 360, N).astype(float),
    "DISTANCE":      np.random.randint(200, 3000, N).astype(float),
})
print(f"   Records: {len(raw):,} | Carriers: {raw['OP_CARRIER'].nunique()}")

#  2. Feature engineering
print("\n[2/5] Feature engineering...")
from modules.airline.features.feature_engineering import (
    rename_and_clean, add_temporal_features,
    add_airport_congestion_features, add_route_historical_features,
    add_carrier_features, add_weather_severity, encode_categoricals,
    AIRLINE_FEATURE_COLS,
)
df = rename_and_clean(raw)
df = add_temporal_features(df)
df = add_airport_congestion_features(df)
df = add_route_historical_features(df)
df = add_carrier_features(df)
df = add_weather_severity(df)
df = encode_categoricals(df)

FEATURES = [f for f in AIRLINE_FEATURE_COLS if f in df.columns]
print(f"   Shape: {df.shape} | Features: {len(FEATURES)}")
print(f"   Delay rate: {df['is_delayed'].mean():.1%}")

# 3. Train XGBoost 
print("\n[3/5] Training XGBoost (delay classifier + regressor)...")
import xgboost as xgb
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_auc_score, f1_score, mean_absolute_error

df_s  = df.sort_values("flight_date")
split = int(len(df_s) * 0.8)
X_tr  = df_s[FEATURES].iloc[:split]
X_vl  = df_s[FEATURES].iloc[split:]
y_cls_tr = df_s["is_delayed"].iloc[:split]
y_cls_vl = df_s["is_delayed"].iloc[split:]
y_reg_tr = df_s["delay_minutes"].iloc[:split]
y_reg_vl = df_s["delay_minutes"].iloc[split:]

# Classifier
clf_base = xgb.XGBClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    early_stopping_rounds=20,
    use_label_encoder=False, eval_metric="logloss", verbosity=0,
)
clf_base.fit(X_tr, y_cls_tr, eval_set=[(X_vl, y_cls_vl)], verbose=False)
from sklearn.calibration import calibration_curve
# Direct Platt scaling via logistic regression on validation scores
from sklearn.linear_model import LogisticRegression
raw_scores = clf_base.predict_proba(X_vl)[:, 1].reshape(-1, 1)
platt = LogisticRegression()
platt.fit(raw_scores, y_cls_vl)
class CalibratedWrapper:
    def predict_proba(self, X):
        raw = clf_base.predict_proba(X)[:, 1].reshape(-1, 1)
        cal = platt.predict_proba(raw)
        return cal
clf = CalibratedWrapper()

proba = clf.predict_proba(X_vl)[:, 1]
preds = (proba >= 0.5).astype(int)
auc   = roc_auc_score(y_cls_vl, proba)
f1    = f1_score(y_cls_vl, preds, zero_division=0)
print(f"   Classifier  → AUC: {auc:.4f} | F1: {f1:.4f}")

# Regressor
reg = xgb.XGBRegressor(
    n_estimators=200, max_depth=5, learning_rate=0.1,
    early_stopping_rounds=20, verbosity=0,
)
reg.fit(X_tr, y_reg_tr, eval_set=[(X_vl, y_reg_vl)], verbose=False)
reg_mae = mean_absolute_error(y_reg_vl, reg.predict(X_vl).clip(0))
print(f"   Regressor   → MAE: {reg_mae:.2f} min")

# Demo predictions
print(f"\n   Example predictions (5 flights):")
print(f"   {'#':>2}  {'Prob':>6}  {'Risk':>6}  {'Est. Delay':>10}  {'Actual':>8}")
print("   " + "─"*40)
for i in range(5):
    row   = X_vl.iloc[[i]]
    p     = float(clf.predict_proba(row)[0, 1])
    dm    = float(max(0, reg.predict(row)[0]))
    act   = float(y_cls_vl.iloc[i])
    risk  = "HIGH" if p > 0.6 else "MED" if p > 0.3 else "LOW"
    flag  = "✅" if int(p >= 0.5) == int(act) else "❌"
    print(f"   {i+1:>2}  {p:>6.1%}  {risk:>6}  {dm:>8.1f}min  {flag}")

#  4. Cascade detection
print("\n[4/5] Cascade delay propagation (NetworkX)...")
from modules.airline.cascade.detector import CascadeDetector, build_demo_schedule

schedule = build_demo_schedule()
det = CascadeDetector(buffer_minutes=30)
det.build_graph(schedule)
summary = det.get_graph_summary()
print(f"   Graph: {summary['total_flights']} flights | "
      f"{summary['total_connections']} connections")

for source, delay in [("AA101", 45.0), ("AA101", 90.0), ("AA101", 15.0)]:
    result = det.propagate_delay(source, delay)
    print(f"   {source} +{delay:.0f}min → "
          f"{result.total_affected} downstream affected, "
          f"max propagated: {result.max_propagated_delay:.0f}min")
    for f in result.affected_flights:
        print(f"      └─ {f['flight_id']} ({f['origin']}→{f['dest']}) "
              f"+{f['propagated_delay_min']:.0f}min [depth {f['cascade_depth']}]")

#  5. Drift check 
print("\n[5/5] Drift monitoring (Evidently AI)...")
from mlops.evidently.drift_monitor import DriftMonitor

mid  = len(df) // 2
ref  = df.iloc[:mid]
cur  = df.iloc[mid:]
mon  = DriftMonitor(module="airline_demo")
rep  = mon.compute_feature_drift(ref, cur, FEATURES[:8])
print(f"   Overall PSI:       {rep['overall_psi']:.4f}")
print(f"   Drift detected:    {rep['drift_detected']}")
print(f"   Features checked:  {rep['n_features_checked']}")
if rep["drifted_features"]:
    print(f"   Drifted features:  {rep['drifted_features']}")

print("\n" + "=" * 60)
print("  Airline demo complete.")
print("  To train on real DOT data:")
print("    make download-data && make train-airline")
print("=" * 60)
