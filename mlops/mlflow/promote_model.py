"""
MLflow Model Registry — Staging → Production Promotion Script

Promotes the best-performing model version from Staging to Production
for both supply chain and airline modules.

Usage:
    python mlops/mlflow/promote_model.py                    # promote both
    python mlops/mlflow/promote_model.py --module supply_chain
    python mlops/mlflow/promote_model.py --module airline --metric AUC
"""

import argparse
import logging

import mlflow
from mlflow.tracking import MlflowClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MLFLOW_URI = "http://localhost:5000"

# Model names as registered in MLflow
MODEL_REGISTRY = {
    "supply_chain": {
        "model_name":    "supply_chain_lgbm_p50",
        "metric":        "final_MAE",
        "lower_is_better": True,
    },
    "airline": {
        "model_name":    "airline_xgboost_classifier",
        "metric":        "classifier_AUC",
        "lower_is_better": False,
    },
}


def get_best_staging_version(
    client: MlflowClient,
    model_name: str,
    metric: str,
    lower_is_better: bool,
) -> str | None:
    """
    Find the best model version in Staging by comparing metric values.
    Returns the version string of the best model, or None if no staging models.
    """
    try:
        staging_versions = client.get_latest_versions(model_name, stages=["Staging"])
    except Exception as e:
        logger.warning(f"Could not fetch staging versions for {model_name}: {e}")
        return None

    if not staging_versions:
        logger.info(f"No Staging versions found for {model_name}")
        return None

    best_version = None
    best_metric  = None

    for mv in staging_versions:
        run = client.get_run(mv.run_id)
        metric_val = run.data.metrics.get(metric)

        if metric_val is None:
            logger.warning(f"Metric '{metric}' not found in run {mv.run_id}")
            continue

        if best_metric is None:
            best_metric  = metric_val
            best_version = mv.version
        else:
            if lower_is_better and metric_val < best_metric:
                best_metric  = metric_val
                best_version = mv.version
            elif not lower_is_better and metric_val > best_metric:
                best_metric  = metric_val
                best_version = mv.version

    if best_version:
        logger.info(f"Best staging version: {best_version} ({metric}={best_metric:.4f})")

    return best_version


def archive_current_production(client: MlflowClient, model_name: str):
    """Move current Production versions to Archived before promoting new one."""
    try:
        prod_versions = client.get_latest_versions(model_name, stages=["Production"])
        for mv in prod_versions:
            client.transition_model_version_stage(
                name=model_name, version=mv.version, stage="Archived"
            )
            logger.info(f"Archived previous production version {mv.version}")
    except Exception as e:
        logger.warning(f"Could not archive production versions: {e}")


def promote_module(client: MlflowClient, module: str):
    """Promote best Staging model to Production for a given module."""
    config = MODEL_REGISTRY[module]
    model_name      = config["model_name"]
    metric          = config["metric"]
    lower_is_better = config["lower_is_better"]

    logger.info(f"\n{'─'*50}")
    logger.info(f"Module: {module} | Model: {model_name} | Metric: {metric}")

    best_version = get_best_staging_version(client, model_name, metric, lower_is_better)

    if best_version is None:
        logger.warning(f"No eligible Staging model found for {module}. Skipping.")
        return

    # Archive current production
    archive_current_production(client, model_name)

    # Promote to production
    client.transition_model_version_stage(
        name=model_name,
        version=best_version,
        stage="Production",
        archive_existing_versions=False,
    )

    # Add description tag
    client.update_model_version(
        name=model_name,
        version=best_version,
        description=f"Promoted to Production by promote_model.py | {metric} optimised",
    )

    logger.info(f"✅  {model_name} v{best_version} → Production")


def main(modules: list[str]):
    mlflow.set_tracking_uri(MLFLOW_URI)
    client = MlflowClient()

    for module in modules:
        promote_module(client, module)

    logger.info("\n✅  Promotion complete")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--module",
        choices=["supply_chain", "airline", "both"],
        default="both",
    )
    args = parser.parse_args()

    modules = (
        ["supply_chain", "airline"] if args.module == "both"
        else [args.module]
    )
    main(modules)
