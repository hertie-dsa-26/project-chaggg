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
from src.algorithms.data_prep import build_monthly_ca_table  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare monthly CA table for KNN")
    parser.add_argument(
        "--output",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca.parquet",
        help="Output path (parquet).",
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


if __name__ == "__main__":
    main()

