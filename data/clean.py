"""
Clean and preprocess the Chicago Crimes dataset (2001–2025).

Implements Issue #49 tasks:
- drop rows with missing latitude/longitude
- extract temporal features: Year, Month, Day, Hour, Day of Week
- convert suitable columns to categorical (best preserved in Parquet)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


RAW_DEFAULT = "data/raw/chicago_crimes_2001_2025_raw.csv"
OUT_DEFAULT_CSV = "data/cleaned/chicago_crimes_cleaned.csv"
OUT_DEFAULT_PARQUET = "data/cleaned/chicago_crimes_cleaned.parquet"


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # Normalize column names to expected lowercase snake-ish naming used across repo.
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Parse timestamp.
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"], errors="coerce", utc=False)
    else:
        raise ValueError("Expected a 'date' column in the raw dataset.")

    # Filter to 2001–2025 using timestamp (more robust than relying on 'year' column).
    df = df[(dt >= "2001-01-01") & (dt < "2026-01-01")]
    dt = dt.loc[df.index]

    # Drop rows with missing coordinates.
    for col in ("latitude", "longitude"):
        if col not in df.columns:
            raise ValueError(f"Expected '{col}' column in the raw dataset.")
    df = df.dropna(subset=["latitude", "longitude"])
    dt = dt.loc[df.index]

    # Temporal features (capitalized names requested in Issue #49).
    df["Year"] = dt.dt.year.astype("int16")
    df["Month"] = dt.dt.month.astype("int8")
    df["Day"] = dt.dt.day.astype("int8")
    df["Hour"] = dt.dt.hour.astype("int8")
    df["Day of Week"] = dt.dt.day_name()

    # Standardize types for numeric codes where appropriate.
    for col in ("district", "ward", "community_area"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Convert "code-like" and low-cardinality strings to categorical to reduce memory.
    for col in ("primary_type", "description", "location_description", "iucr", "fbi_code", "Day of Week"):
        if col in df.columns:
            df[col] = df[col].astype("category")

    # Remove redundant structured column if present.
    if "location" in df.columns:
        df = df.drop(columns=["location"])

    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Clean/preprocess the Chicago Crimes dataset.")
    parser.add_argument("--raw", default=RAW_DEFAULT, help="Path to raw CSV.")
    parser.add_argument(
        "--format",
        choices=["csv", "parquet"],
        default="parquet",
        help="Output format (parquet recommended).",
    )
    parser.add_argument("--out", default=None, help="Output file path override.")
    args = parser.parse_args()

    raw_path = Path(args.raw)
    if not raw_path.exists():
        raise SystemExit(f"Raw data not found at {raw_path}. Run download_data.py first.")

    out_path = Path(
        args.out
        or (OUT_DEFAULT_PARQUET if args.format == "parquet" else OUT_DEFAULT_CSV)
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(raw_path, low_memory=False)
    cleaned = preprocess(df)

    if args.format == "parquet":
        cleaned.to_parquet(out_path, index=False)
    else:
        cleaned.to_csv(out_path, index=False)

    print(f"Saved cleaned dataset to {out_path} (rows: {len(cleaned)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
