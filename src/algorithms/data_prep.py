from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MonthlyCommunityAreaTable:
    """
    Compact table for KNN training/testing:
      - one row per (community_area, month)
      - target: crime_count
      - features: lat, lon (centroid proxy) + month_index
    """

    df: pd.DataFrame


def month_index(series: pd.Series, origin: str = "2015-01-01") -> pd.Series:
    """
    Convert timestamps to integer month index relative to origin.
    """
    ts = pd.to_datetime(series, errors="coerce")
    origin_ts = pd.Timestamp(origin)
    return (ts.dt.year - origin_ts.year) * 12 + (ts.dt.month - origin_ts.month)


def build_monthly_ca_table(
    crimes: pd.DataFrame,
    date_col: str = "date",
    ca_col: str = "community_area",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> MonthlyCommunityAreaTable:
    """
    Aggregate incident-level crimes into monthly counts per community area.
    Uses mean lat/lon of incidents as a simple location proxy for that CA+month.
    """
    d = crimes[[date_col, ca_col, lat_col, lon_col]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col, ca_col])

    # Month bucket
    d["month"] = d[date_col].dt.to_period("M").dt.to_timestamp()
    out = (
        d.groupby([ca_col, "month"], dropna=False)
        .agg(
            crime_count=(date_col, "size"),
            lat=(lat_col, "mean"),
            lon=(lon_col, "mean"),
        )
        .reset_index()
    )
    out["month_index"] = month_index(out["month"])
    return MonthlyCommunityAreaTable(df=out)


def split_train_test_by_year(
    monthly: MonthlyCommunityAreaTable,
    train_years: tuple[int, int] = (2015, 2022),
    test_years: tuple[int, int] = (2023, 2024),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = monthly.df.copy()
    year = pd.to_datetime(df["month"], errors="coerce").dt.year
    train = df[year.between(train_years[0], train_years[1])].reset_index(drop=True)
    test = df[year.between(test_years[0], test_years[1])].reset_index(drop=True)
    return train, test
