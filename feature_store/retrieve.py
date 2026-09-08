"""
Feast Feature Retrieval Helpers
Used by the API to fetch online features for a given item-store pair.

Usage:
    from feature_store.retrieve import get_online_features, get_training_features
"""

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

FEATURE_REPO = Path(__file__).parent / "feature_repo"
PARQUET_PATH = Path(__file__).parent / "data" / "supply_chain_features.parquet"


def get_store():
    """Initialize Feast FeatureStore."""
    try:
        from feast import FeatureStore
        return FeatureStore(repo_path=str(FEATURE_REPO))
    except Exception as e:
        logger.warning(f"Feast store init failed: {e}")
        return None


def get_online_features(
    item_ids: list[str],
    store_ids: list[str],
    feature_refs: Optional[list[str]] = None,
) -> pd.DataFrame:
    """
    Retrieve online features for inference.
    Falls back to parquet offline read if online store not populated.

    Args:
        item_ids:  list of encoded item IDs (as strings)
        store_ids: list of encoded store IDs (as strings)
        feature_refs: Feast feature references, e.g.
                      ["supply_chain_lag_features:lag_7", ...]

    Returns:
        DataFrame with one row per (item_id, store_id) pair
    """
    entity_keys = [f"{i}_{s}" for i, s in zip(item_ids, store_ids)]

    if feature_refs is None:
        feature_refs = [
            "supply_chain_lag_features:lag_7",
            "supply_chain_lag_features:lag_14",
            "supply_chain_lag_features:lag_28",
            "supply_chain_rolling_features:rolling_mean_7",
            "supply_chain_rolling_features:rolling_std_7",
            "supply_chain_rolling_features:rolling_mean_28",
            "supply_chain_rolling_features:rolling_std_28",
            "supply_chain_calendar_features:dayofweek",
            "supply_chain_calendar_features:is_weekend",
            "supply_chain_calendar_features:is_german_holiday",
            "supply_chain_calendar_features:has_event",
            "supply_chain_price_features:price_rel_mean",
            "supply_chain_category_features:dept_mean_sales",
            "supply_chain_category_features:store_mean_sales",
        ]

    store = get_store()
    if store is not None:
        try:
            entity_df = pd.DataFrame({
                "supply_chain_item": entity_keys,
                "event_timestamp": [datetime.now(tz=timezone.utc)] * len(entity_keys),
            })
            features = store.get_online_features(
                features=feature_refs,
                entity_rows=[{"supply_chain_item": k} for k in entity_keys],
            ).to_df()
            return features
        except Exception as e:
            logger.warning(f"Online store retrieval failed, falling back to parquet: {e}")

    # Offline fallback — read from parquet
    return _offline_fallback(entity_keys)


def _offline_fallback(entity_keys: list[str]) -> pd.DataFrame:
    """Read most recent row per entity from the parquet offline store."""
    if not PARQUET_PATH.exists():
        logger.error(f"Parquet store not found at {PARQUET_PATH}. Run materialize.py first.")
        return pd.DataFrame()

    df = pd.read_parquet(PARQUET_PATH)
    df = df[df["supply_chain_item"].isin(entity_keys)]
    df = df.sort_values("event_timestamp").groupby("supply_chain_item").last().reset_index()
    return df


def get_training_features(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """
    Point-in-time correct historical feature retrieval for training.
    Uses Feast offline store (parquet).
    """
    store = get_store()
    if store is None or not PARQUET_PATH.exists():
        logger.warning("Feast store unavailable — loading raw parquet for training")
        if PARQUET_PATH.exists():
            df = pd.read_parquet(PARQUET_PATH)
            if start_date:
                df = df[df["event_timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
            if end_date:
                df = df[df["event_timestamp"] <= pd.Timestamp(end_date, tz="UTC")]
            return df
        return pd.DataFrame()

    # Build entity + timestamp combinations for point-in-time join
    df = pd.read_parquet(PARQUET_PATH)
    entity_df = df[["supply_chain_item", "event_timestamp", "sales"]].copy()

    if start_date:
        entity_df = entity_df[entity_df["event_timestamp"] >= pd.Timestamp(start_date, tz="UTC")]
    if end_date:
        entity_df = entity_df[entity_df["event_timestamp"] <= pd.Timestamp(end_date, tz="UTC")]

    try:
        training_df = store.get_historical_features(
            entity_df=entity_df,
            features=[
                "supply_chain_lag_features:lag_7",
                "supply_chain_lag_features:lag_14",
                "supply_chain_lag_features:lag_28",
                "supply_chain_rolling_features:rolling_mean_7",
                "supply_chain_rolling_features:rolling_std_7",
                "supply_chain_rolling_features:rolling_mean_28",
                "supply_chain_rolling_features:rolling_std_28",
                "supply_chain_calendar_features:dayofweek",
                "supply_chain_calendar_features:month",
                "supply_chain_calendar_features:is_weekend",
                "supply_chain_calendar_features:is_german_holiday",
                "supply_chain_price_features:price_rel_mean",
                "supply_chain_category_features:dept_mean_sales",
                "supply_chain_category_features:store_mean_sales",
            ],
        ).to_df()
        return training_df
    except Exception as e:
        logger.warning(f"Historical retrieval failed: {e}. Using raw parquet.")
        return df
