from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

# Ensure scripts/ is importable (load_data/config live there).
project_root = Path(__file__).resolve().parents[2]
scripts_dir = project_root / "scripts"
if str(scripts_dir) not in sys.path:
    sys.path.insert(0, str(scripts_dir))

from utils import load_data  # noqa: E402


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
    if os.environ.get("CHAGGG_SKIP_DATA_LOAD") == "1":
        return pd.DataFrame(columns=DEFAULT_COLUMNS)
    try:
        return load_data(prefer_parquet=True)
    except FileNotFoundError:
        # Keep the app bootable even if cleaned data doesn't exist
        return pd.DataFrame(columns=DEFAULT_COLUMNS)