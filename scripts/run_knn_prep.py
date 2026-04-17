"""
Prepare a compact monthly community-area table for US-2.4 KNN training/testing.

This reduces incident-level data into:
  - community_area, month, month_index
  - crime_count (target)
  - lat/lon (mean location proxy)

Run from repo root (after cleaned data exists):
    PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplot uv run python scripts/run_knn_prep.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root))

from utils import load_data  # noqa: E402
from src.algorithms.data_prep import (  # noqa: E402
    build_monthly_ca_table,
    MonthlyCommunityAreaTable,
    split_train_test_by_year,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare monthly CA table for KNN")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca.parquet",
        help="Output path (parquet).",
    )
    parser.add_argument(
        "--write-split",
        action="store_true",
        help="Also write train/test split parquet files next to --output.",
    )
    args = parser.parse_args()

    cols = ["date", "community_area", "latitude", "longitude"]
    print("Loading cleaned data (subset columns)…", flush=True)
    df = load_data(columns=cols)
    print(f"Loaded rows: {len(df):,}", flush=True)

    print("Building monthly community_area table…", flush=True)
    monthly = build_monthly_ca_table(
        df,
        date_col="date",
        ca_col="community_area",
        lat_col="latitude",
        lon_col="longitude",
    ).df

    print(f"Monthly table rows: {len(monthly):,}", flush=True)
    print(monthly.head(3).to_string(index=False), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_parquet(args.output, index=False)
    print(f"Saved: {args.output}", flush=True)

    if args.write_split:
        train_df, test_df = split_train_test_by_year(
            monthly=MonthlyCommunityAreaTable(df=monthly)
        )
        # Default split per US-2.4: train 2015–2022, test 2023–2024
        train_path = args.output.with_name("monthly_ca_train_2015_2022.parquet")
        test_path = args.output.with_name("monthly_ca_test_2023_2024.parquet")
        train_df.to_parquet(train_path, index=False)
        test_df.to_parquet(test_path, index=False)
        print(f"Saved split train: {train_path} rows={len(train_df):,}", flush=True)
        print(f"Saved split test:  {test_path} rows={len(test_df):,}", flush=True)


if __name__ == "__main__":
    main()

