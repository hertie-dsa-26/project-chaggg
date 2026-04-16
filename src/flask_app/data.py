from __future__ import annotations

import pandas as pd
from scripts.utils import load_data


DEFAULT_COLUMNS = [
    "ID",
    "Date",
    "Primary Type",
    "Description",
    "Arrest",
    "Domestic",
    "District",
    "Ward",
    "Community Area",
    "Latitude",
    "Longitude",
]


def load_crime_data() -> pd.DataFrame:
    """
    Load the Chicago crime dataset for the web app.
    
    Uses the cleaned data from the preprocessing pipeline.
    Falls back to empty DataFrame if data is not available.
    """
    try:
        return load_data(prefer_parquet=True)
    except FileNotFoundError:
        # Keep the app bootable even if cleaned data doesn't exist
        return pd.DataFrame(columns=DEFAULT_COLUMNS)