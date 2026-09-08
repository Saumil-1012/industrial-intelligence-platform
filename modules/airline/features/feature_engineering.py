"""
Airline Feature Engineering
Transforms DOT On-Time Performance CSV into ML-ready features.

Features:
- Aircraft rotation time (time since last flight for same tail number)
- Airport congestion score (departures per hour at origin)
- Weather severity index (from DOT weather delay fields)
- ATC delay history (rolling NAS delay per airport)
- Route-level historical delay statistics
- Temporal features (hour, day, season, holiday proximity)
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[3] / "data" / "airline"

# DOT column mapping — dataset uses these exact names
DOT_COLS = {
    "FL_DATE":           "flight_date",
    "OP_CARRIER":        "carrier",
    "TAIL_NUM":          "tail_number",
    "FL_NUM":            "flight_num",
    "ORIGIN":            "origin",
    "DEST":              "dest",
    "CRS_DEP_TIME":      "scheduled_dep",
    "DEP_TIME":          "actual_dep",
    "DEP_DELAY":         "dep_delay",
    "ARR_DELAY":         "arr_delay",
    "CANCELLED":         "cancelled",
    "CARRIER_DELAY":     "carrier_delay",
    "WEATHER_DELAY":     "weather_delay",
    "NAS_DELAY":         "nas_delay",
    "SECURITY_DELAY":    "security_delay",
    "LATE_AIRCRAFT_DELAY": "late_aircraft_delay",
    "AIR_TIME":          "air_time",
    "DISTANCE":          "distance",
}


def load_dot_data(
    data_dir: Optional[Path] = None,
    years: Optional[list[int]] = None,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Load DOT CSVs. Supports multiple years (2009-2018).
    sample_frac < 1.0 for dev iteration.
    """
    data_dir = data_dir or DATA_DIR
    years    = years or list(range(2009, 2019))

    dfs = []
    for year in years:
        path = data_dir / f"{year}.csv"
        if not path.exists():
            logger.warning(f"Missing: {path} - skipping")
            continue
        logger.info(f"Loading {path.name}...")
        df = pd.read_csv(path, low_memory=False)
        if sample_frac < 1.0:
            df = df.sample(frac=sample_frac, random_state=42)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError(
            f"No DOT CSV files found in {data_dir}. Run data/airline/download.py first."
        )

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Loaded {len(combined):,} flight records")
    return combined


def rename_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DOT columns to clean names, drop cancelled flights, cast types."""
    available = {k: v for k, v in DOT_COLS.items() if k in df.columns}
    df = df.rename(columns=available)

    # Drop cancelled flights — not predicting cancellations here
    if "cancelled" in df.columns:
        df = df[df["cancelled"] != 1.0].copy()

    # Parse date
    if "flight_date" in df.columns:
        df["flight_date"] = pd.to_datetime(df["flight_date"], errors="coerce")

    # Numeric casts
    numeric_cols = [
        "dep_delay", "arr_delay", "carrier_delay", "weather_delay",
        "nas_delay", "security_delay", "late_aircraft_delay",
        "air_time", "distance", "scheduled_dep",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Target: binary delay (>= 15 min arrival delay = DOT definition)
    if "arr_delay" in df.columns:
        df["is_delayed"]    = (df["arr_delay"] >= 15).astype(int)
        df["delay_minutes"] = df["arr_delay"].clip(lower=0).fillna(0)

    df = df.dropna(subset=["flight_date", "origin", "dest"]).reset_index(drop=True)
    logger.info(f"After cleaning: {len(df):,} flights")
    return df


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour of day, day of week, month, season, holiday proximity."""
    df["hour_of_day"]  = (df["scheduled_dep"] // 100).clip(0, 23).astype(int)
    df["day_of_week"]  = df["flight_date"].dt.dayofweek      # 0=Mon
    df["month"]        = df["flight_date"].dt.month
    df["day_of_year"]  = df["flight_date"].dt.dayofyear
    df["is_weekend"]   = (df["day_of_week"] >= 5).astype(int)

    # Season: 0=winter, 1=spring, 2=summer, 3=fall
    df["season"] = ((df["month"] % 12) // 3).astype(int)

    # Peak travel hours: 6-9am and 4-8pm
    df["is_peak_hour"] = (
        ((df["hour_of_day"] >= 6)  & (df["hour_of_day"] <= 9)) |
        ((df["hour_of_day"] >= 16) & (df["hour_of_day"] <= 20))
    ).astype(int)

    # Holiday proximity (US major holidays — simple rule-based)
    df["is_holiday_week"] = (
        (df["month"].isin([11, 12])) &
        (df["flight_date"].dt.day.between(20, 31))
    ).astype(int)

    return df


def add_aircraft_rotation_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aircraft rotation time: minutes between an aircraft's last arrival
    and this departure. High rotation time = less pressure from previous delay.
    Low rotation time = cascading delay risk.
    """
    if "tail_number" not in df.columns:
        df["rotation_time_min"] = np.nan
        df["rotation_risk"]     = 0
        return df

    df = df.sort_values(["tail_number", "flight_date", "scheduled_dep"])

    # Approximate actual departure time as decimal hour
    df["dep_hour_decimal"] = df["hour_of_day"] + (df["scheduled_dep"] % 100) / 60.0

    # Previous flight's arrival time per tail number
    df["prev_arr_hour"] = df.groupby("tail_number")["dep_hour_decimal"].shift(1)
    df["rotation_time_min"] = (df["dep_hour_decimal"] - df["prev_arr_hour"]) * 60

    # Clip to sane range (overnight turns show as negative)
    df["rotation_time_min"] = df["rotation_time_min"].clip(-60, 300)

    # Rotation risk: < 45 min is tight
    df["rotation_risk"] = (df["rotation_time_min"] < 45).astype(int)
    df["rotation_risk"] = df["rotation_risk"].fillna(0)

    return df


def add_airport_congestion_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Airport congestion score: number of departures from the same origin
    in the same hour window. High count = more delays expected.
    """
    if "origin" not in df.columns:
        df["origin_hourly_departures"] = 0
        return df

    df["hour_bucket"] = df["flight_date"].astype(str) + "_" + df["hour_of_day"].astype(str)
    congestion = (
        df.groupby(["origin", "hour_bucket"])["flight_date"]
        .transform("count")
        .rename("origin_hourly_departures")
    )
    df["origin_hourly_departures"] = congestion

    # Congestion tier: low/medium/high
    df["congestion_tier"] = pd.cut(
        df["origin_hourly_departures"],
        bins=[0, 5, 15, np.inf],
        labels=[0, 1, 2],
    ).astype(float).fillna(0)

    return df


def add_route_historical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Route-level historical delay statistics.
    Mean/std delay per (origin, dest) pair — computed on training set.
    Note: must be computed only on training data to avoid leakage.
    """
    if "arr_delay" not in df.columns:
        return df

    route_stats = (
        df.groupby(["origin", "dest"])["arr_delay"]
        .agg(["mean", "std"])
        .rename(columns={"mean": "route_mean_delay", "std": "route_std_delay"})
        .reset_index()
    )
    df = df.merge(route_stats, on=["origin", "dest"], how="left")
    df["route_mean_delay"] = df["route_mean_delay"].fillna(df["arr_delay"].mean())
    df["route_std_delay"]  = df["route_std_delay"].fillna(df["arr_delay"].std())
    return df


def add_carrier_features(df: pd.DataFrame) -> pd.DataFrame:
    """Carrier-level historical delay rate and mean delay."""
    if "carrier" not in df.columns or "arr_delay" not in df.columns:
        return df

    carrier_stats = (
        df.groupby("carrier").agg(
            carrier_delay_rate=("is_delayed", "mean"),
            carrier_mean_delay=("arr_delay", "mean"),
        ).reset_index()
    )
    df = df.merge(carrier_stats, on="carrier", how="left")
    df["carrier_delay_rate"] = df["carrier_delay_rate"].fillna(0.2)
    df["carrier_mean_delay"] = df["carrier_mean_delay"].fillna(0.0)
    return df


def add_weather_severity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Weather severity index: composite of DOT weather_delay + nas_delay.
    Normalized to 0-1.
    """
    weather_raw = df.get("weather_delay", pd.Series(np.zeros(len(df)))).fillna(0)
    nas_raw     = df.get("nas_delay",     pd.Series(np.zeros(len(df)))).fillna(0)

    raw = weather_raw + 0.5 * nas_raw
    max_val = raw.quantile(0.99) + 1e-6
    df["weather_severity"] = (raw / max_val).clip(0, 1)

    df["has_weather_delay"] = (weather_raw > 0).astype(int)
    df["has_nas_delay"]     = (nas_raw > 0).astype(int)
    return df


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode carrier, origin, dest."""
    for col in ["carrier", "origin", "dest"]:
        if col in df.columns:
            df[col] = df[col].astype("category").cat.codes
    return df


def build_airline_features(
    data_dir: Optional[Path] = None,
    years: Optional[list[int]] = None,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Full airline feature pipeline.
    Returns ML-ready DataFrame with is_delayed + delay_minutes targets.
    """
    df = load_dot_data(data_dir, years, sample_frac)
    df = rename_and_clean(df)

    logger.info("Adding temporal features...")
    df = add_temporal_features(df)

    logger.info("Adding aircraft rotation features...")
    df = add_aircraft_rotation_features(df)

    logger.info("Adding airport congestion features...")
    df = add_airport_congestion_features(df)

    logger.info("Adding route historical features...")
    df = add_route_historical_features(df)

    logger.info("Adding carrier features...")
    df = add_carrier_features(df)

    logger.info("Adding weather severity...")
    df = add_weather_severity(df)

    logger.info("Encoding categoricals...")
    df = encode_categoricals(df)

    df = df.reset_index(drop=True)
    logger.info(f"Airline feature engineering complete. Shape: {df.shape}")
    return df


AIRLINE_FEATURE_COLS = [
    # Temporal
    "hour_of_day", "day_of_week", "month", "day_of_year",
    "is_weekend", "season", "is_peak_hour", "is_holiday_week",
    # Aircraft
    "rotation_time_min", "rotation_risk",
    # Airport
    "origin_hourly_departures", "congestion_tier",
    # Route history
    "route_mean_delay", "route_std_delay",
    # Carrier
    "carrier_delay_rate", "carrier_mean_delay",
    # Weather
    "weather_severity", "has_weather_delay", "has_nas_delay",
    # Encoded categoricals
    "carrier", "origin", "dest",
    # Flight metadata
    "distance", "air_time",
]

DELAY_CAUSES = ["carrier_delay", "weather_delay", "nas_delay", "security_delay", "late_aircraft_delay"]
