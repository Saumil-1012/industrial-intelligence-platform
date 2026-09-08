"""
Supply Chain Feature Engineering
Transforms M5 raw sales data into ML-ready features.

Features:
- Lag features (7, 14, 28 days)
- Rolling stats (mean, std, max)
- Calendar effects (weekday, month, holidays)
- Price elasticity features
- Category-level aggregations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[3] / "data" / "supply_chain"

# German public holidays (static for reproducibility)
GERMAN_HOLIDAYS = {
    "2011-01-01", "2011-04-22", "2011-05-01", "2011-10-03", "2011-12-25", "2011-12-26",
    "2012-01-01", "2012-04-06", "2012-05-01", "2012-10-03", "2012-12-25", "2012-12-26",
    "2013-01-01", "2013-03-29", "2013-05-01", "2013-10-03", "2013-12-25", "2013-12-26",
    "2014-01-01", "2014-04-18", "2014-05-01", "2014-10-03", "2014-12-25", "2014-12-26",
    "2015-01-01", "2015-04-03", "2015-05-01", "2015-10-03", "2015-12-25", "2015-12-26",
    "2016-01-01", "2016-03-25", "2016-05-01", "2016-10-03", "2016-12-25", "2016-12-26",
}


def load_m5_data(
    sales_path: Optional[Path] = None,
    calendar_path: Optional[Path] = None,
    prices_path: Optional[Path] = None,
    sample: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load M5 raw CSVs.
    If sample=True, loads only 500 items for fast dev iteration.
    """
    sales_path = sales_path or DATA_DIR / "sales_train_validation.csv"
    calendar_path = calendar_path or DATA_DIR / "calendar.csv"
    prices_path = prices_path or DATA_DIR / "sell_prices.csv"

    logger.info("Loading M5 dataset...")
    sales = pd.read_csv(sales_path)
    calendar = pd.read_csv(calendar_path, parse_dates=["date"])
    prices = pd.read_csv(prices_path)

    if sample:
        sales = sales.sample(n=min(500, len(sales)), random_state=42)
        logger.info(f"Sampled to {len(sales)} items for dev mode")

    logger.info(f"Loaded {len(sales)} items, {len(calendar)} calendar days")
    return sales, calendar, prices


def melt_sales(sales: pd.DataFrame, calendar: pd.DataFrame) -> pd.DataFrame:
    """
    Pivot M5 from wide (one row per item) to long (one row per item-day).
    Merges in calendar dates.
    """
    id_cols = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_cols = [c for c in sales.columns if c.startswith("d_")]

    df = sales.melt(id_vars=id_cols, value_vars=day_cols, var_name="d", value_name="sales")
    df = df.merge(calendar[["d", "date", "wday", "month", "year", "event_name_1", "event_name_2"]],
                  on="d", how="left")
    df = df.sort_values(["id", "date"]).reset_index(drop=True)
    return df


def add_lag_features(df: pd.DataFrame, lags: list[int] = [7, 14, 28]) -> pd.DataFrame:
    """Add lag sales features per item."""
    for lag in lags:
        df[f"lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, windows: list[int] = [7, 14, 28]) -> pd.DataFrame:
    """Add rolling mean, std, max per item."""
    for window in windows:
        grp = df.groupby("id")["sales"]
        df[f"rolling_mean_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0)
        )
        df[f"rolling_max_{window}"] = grp.transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).max()
        )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal and holiday features."""
    df["date"] = pd.to_datetime(df["date"])
    df["dayofweek"] = df["date"].dt.dayofweek          # 0=Mon
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    df["is_german_holiday"] = df["date"].dt.strftime("%Y-%m-%d").isin(GERMAN_HOLIDAYS).astype(int)
    df["has_event"] = (df["event_name_1"].notna() | df["event_name_2"].notna()).astype(int)
    return df


def add_price_features(df: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sell prices and compute price elasticity proxy:
    price relative to item's historical mean.
    """
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"] if "wm_yr_wk" in df.columns else ["store_id", "item_id"],
                  how="left")

    if "sell_price" in df.columns:
        item_mean_price = df.groupby("item_id")["sell_price"].transform("mean")
        df["price_rel_mean"] = df["sell_price"] / (item_mean_price + 1e-6)
        df["price_change"] = df.groupby("item_id")["sell_price"].pct_change().fillna(0)
    return df


def add_category_aggregations(df: pd.DataFrame) -> pd.DataFrame:
    """Category-level mean sales (store + dept level aggregations)."""
    df["dept_mean_sales"] = df.groupby(["dept_id", "date"])["sales"].transform("mean")
    df["store_mean_sales"] = df.groupby(["store_id", "date"])["sales"].transform("mean")
    df["cat_mean_sales"] = df.groupby(["cat_id", "date"])["sales"].transform("mean")
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode categorical columns."""
    cat_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def build_features(
    sales_path: Optional[Path] = None,
    calendar_path: Optional[Path] = None,
    prices_path: Optional[Path] = None,
    sample: bool = False,
) -> pd.DataFrame:
    """
    Full feature pipeline: load → melt → engineer → return ready-to-train DataFrame.
    """
    sales, calendar, prices = load_m5_data(sales_path, calendar_path, prices_path, sample=sample)

    logger.info("Melting to long format...")
    df = melt_sales(sales, calendar)

    logger.info("Adding lag features...")
    df = add_lag_features(df)

    logger.info("Adding rolling features...")
    df = add_rolling_features(df)

    logger.info("Adding calendar features...")
    df = add_calendar_features(df)

    logger.info("Adding price features...")
    # prices needs wm_yr_wk — merge calendar first
    if "wm_yr_wk" not in df.columns:
        cal_wk = pd.read_csv(calendar_path or DATA_DIR / "calendar.csv")[["d", "wm_yr_wk"]]
        df = df.merge(cal_wk, on="d", how="left")
    df = add_price_features(df, prices)

    logger.info("Adding category aggregations...")
    df = add_category_aggregations(df)

    logger.info("Encoding categoricals...")
    df = encode_categoricals(df)

    # Drop rows with NaN lags (first 28 days per item)
    df = df.dropna(subset=["lag_7", "lag_14", "lag_28"])
    df = df.reset_index(drop=True)

    logger.info(f"Feature engineering complete. Shape: {df.shape}")
    return df


FEATURE_COLS = [
    "lag_7", "lag_14", "lag_28",
    "rolling_mean_7", "rolling_std_7", "rolling_max_7",
    "rolling_mean_14", "rolling_std_14", "rolling_max_14",
    "rolling_mean_28", "rolling_std_28", "rolling_max_28",
    "dayofweek", "dayofmonth", "weekofyear", "month", "year",
    "is_weekend", "is_month_start", "is_month_end",
    "is_german_holiday", "has_event",
    "price_rel_mean", "price_change",
    "dept_mean_sales", "store_mean_sales", "cat_mean_sales",
    "item_id", "dept_id", "cat_id", "store_id", "state_id",
]

TARGET_COL = "sales"
