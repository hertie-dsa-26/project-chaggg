"""
Naive arrest-probability estimator based on empirical grouped frequencies.

Usage:
    from scripts.estimators import compute_arrest_probability
    probs = compute_arrest_probability(df, crime_type="THEFT", area_col="district")
"""

import pandas as pd

_SUPPORTED_AREA_COLS = {"block", "district"}


def compute_arrest_probability(df: pd.DataFrame, crime_type: str, area_col: str) -> pd.Series:
    """Return P(arrest | crime_type, area) for each area value as a pd.Series."""
    if area_col not in _SUPPORTED_AREA_COLS:
        raise ValueError(
            f"area_col must be one of {_SUPPORTED_AREA_COLS!r}, got {area_col!r}"
        )

    subset = df[df["primary_type"] == crime_type]

    if subset.empty:
        return pd.Series(dtype=float, name=area_col)

    grouped = subset.groupby(area_col)["arrest"]
    totals = grouped.count()
    arrests = grouped.sum()

    probabilities = arrests / totals
    probabilities.name = "arrest_probability"
    return probabilities
