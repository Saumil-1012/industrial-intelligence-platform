"""
Drift API Helper — reads the latest saved drift report for a module.
Used by FastAPI /drift endpoints to serve live drift status.
"""

import json
from pathlib import Path
from typing import Optional

REPORTS_DIR = Path(__file__).parent / "reports"


def get_latest_drift_report(module: str) -> Optional[dict]:
    """Return the most recent drift report JSON for a given module."""
    pattern = f"{module}_drift_*.json"
    reports = sorted(REPORTS_DIR.glob(pattern), reverse=True)
    if not reports:
        return None
    with open(reports[0]) as f:
        return json.load(f)


def drift_status_response(module: str) -> dict:
    """
    Build the /drift API response.
    Returns live data if a report exists, otherwise stub.
    """
    report = get_latest_drift_report(module)

    if report is None:
        return {
            "status":           "no_report_yet",
            "message":          "Run drift monitor first: python -m mlops.evidently.drift_monitor",
            "drift_score":      None,
            "drift_detected":   None,
            "retrain_triggered": None,
            "drifted_features": [],
            "threshold":        0.15,
            "last_run":         None,
        }

    summary = report.get("summary", {})
    return {
        "status":            "monitored",
        "drift_score":       summary.get("overall_psi"),
        "drift_detected":    report.get("feature_drift", {}).get("drift_detected"),
        "retrain_triggered": summary.get("retrain_needed"),
        "drifted_features":  summary.get("drifted_features", []),
        "threshold":         report.get("drift_threshold", 0.15),
        "last_run":          report.get("timestamp"),
        "n_features_checked": report.get("feature_drift", {}).get("n_features_checked"),
    }
