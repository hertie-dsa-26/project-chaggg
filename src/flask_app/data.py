from __future__ import annotations

import os
from pathlib import Path

import pandas as pd


def load_crime_data() -> pd.DataFrame:
    """
    Load the dataset for the Flask app.

    The app is intentionally decoupled from pipeline scripts. Provide a local
    file path via CHAGGG_DATA_PATH (CSV or Parquet). If the file is not present,
    the app remains bootable and returns an empty DataFrame.
    """
    data_path = os.environ.get("CHAGGG_DATA_PATH")
    if not data_path:
        return pd.DataFrame()

    path = Path(data_path)
    if not path.exists() or not path.is_file():
        return pd.DataFrame()

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    return pd.DataFrame()

