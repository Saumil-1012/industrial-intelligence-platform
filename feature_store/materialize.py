"""
Feature Store Materialization Script
Converts engineered features → Parquet for Feast + applies GE validation.

Usage:
    python feature_store/materialize.py --sample      # dev mode
    python feature_store/materialize.py               # full M5 dataset
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Add project root to path
sys.path.insert(0, str(Path(__file__).parents[1]))

from modules.supply_chain.features.feature_engineering import build_features, FEATURE_COLS
from great_expectations.expectations.m5_sales_suite import validate_features

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).parent / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def materialize_supply_chain(sample: bool = False):
    """
    1. Build features via feature_engineering.py
    2. Validate with Great Expectations
    3. Add Feast required columns (entity key + timestamps)
    4. Write to Parquet
    """
    logger.info("Starting feature materialization...")

    df = build_features(sample=sample)

    # GE validation — fail fast on bad features
    logger.info("Running feature validation...")
    try:
        report = validate_features(df)
        logger.info(f"Validation: {report['passed']}/{report['total_expectations']} passed")
    except ValueError as e:
        logger.error(f"Validation failed: {e}")
        sys.exit(1)

    # Add Feast required columns
    df["supply_chain_item"] = df["item_id"].astype(str) + "_" + df["store_id"].astype(str)
    df["event_timestamp"]   = pd.to_datetime(df["date"]).dt.tz_localize("UTC")
    df["created_timestamp"] = datetime.now(tz=timezone.utc)

    # Select columns for feature store
    feast_cols = (
        ["supply_chain_item", "event_timestamp", "created_timestamp"]
        + [f for f in FEATURE_COLS if f in df.columns]
        + ["sales"]  # keep target for offline retrieval
    )
    feast_df = df[[c for c in feast_cols if c in df.columns]]

    output_path = OUTPUT_DIR / "supply_chain_features.parquet"
    feast_df.to_parquet(output_path, index=False)
    logger.info(f"Wrote {len(feast_df):,} rows → {output_path}")
    logger.info(f"Columns: {list(feast_df.columns)}")

    return feast_df


def apply_feast_registry():
    """Run `feast apply` to register feature views in the registry."""
    import subprocess
    feature_repo = Path(__file__).parent / "feature_repo"
    logger.info("Running `feast apply`...")
    result = subprocess.run(
        ["feast", "apply"],
        cwd=feature_repo,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        logger.info("feast apply successful")
        logger.info(result.stdout)
    else:
        logger.warning(f"feast apply output: {result.stderr}")
    return result.returncode == 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--apply", action="store_true", help="Run feast apply after materialization")
    args = parser.parse_args()

    df = materialize_supply_chain(sample=args.sample)

    if args.apply:
        apply_feast_registry()
