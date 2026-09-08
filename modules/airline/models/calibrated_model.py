"""
CalibratedModel — defined in its own module so joblib can pickle/unpickle it.
"""
from sklearn.linear_model import LogisticRegression


class CalibratedModel:
    """XGBoost + Platt scaling wrapper."""

    def __init__(self, base, calibrator):
        self.base       = base
        self.calibrator = calibrator

    def predict_proba(self, X):
        raw = self.base.predict_proba(X)[:, 1].reshape(-1, 1)
        return self.calibrator.predict_proba(raw)

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(int)