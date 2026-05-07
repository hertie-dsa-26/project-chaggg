"""Precompute per-crime-type KNN artifacts.

For each crime type with sufficient data, standardize features, augment with
intercept, and save to data/precomputed/knn/<normalized_type>.npz.

Run with: uv run python -m scripts.precompute_knn_arrays
"""

import numpy as np
from pathlib import Path

from src.preprocess_data import preprocess_data
from scripts.utils import normalise_crime_type

FEATURE_COLS = [
    "latitude", "longitude",
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
]
LABEL_COL = "arrest"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "precomputed" / "knn"


def precompute_one(df, crime_type: str) -> dict:
    """Build the KNN artifact arrays for a single crime type subset."""
    subset = df[df["primary_type"] == crime_type].reset_index(drop=True)

    if len(subset) == 0:
        raise ValueError(f"No rows for crime_type {crime_type!r}")

    features = subset[FEATURE_COLS].to_numpy().astype(np.float64)
    label = subset[LABEL_COL].to_numpy().astype(np.int8)

    # Standardize (matches knn_lrr exactly, epsilon baked into std)
    features_mean = features.mean(axis=0)
    features_std = features.std(axis=0) + 1e-8
    features_scaled = (features - features_mean) / features_std

    # Augment with intercept column
    n = features_scaled.shape[0]
    features_aug = np.hstack([np.ones((n, 1)), features_scaled])

    return {
        "features_aug": features_aug.astype(np.float32),
        "label": label,
        "features_mean": features_mean.astype(np.float32),
        "features_std": features_std.astype(np.float32),
    }


def main():
    print("Loading and preprocessing data...")
    df = preprocess_data()

    crime_types = sorted(df["primary_type"].unique())
    print(f"Found {len(crime_types)} crime types meeting the 5k threshold.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for crime_type in crime_types:
        slug = normalise_crime_type(crime_type)
        artifact = precompute_one(df, crime_type)
        out_path = OUTPUT_DIR / f"{slug}.npz"
        np.savez(out_path, **artifact)
        n = artifact["label"].shape[0]
        print(f"  {slug:<40} n={n:>7}  -> {out_path}")

    print(f"\nDone. {len(crime_types)} artifacts written to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()