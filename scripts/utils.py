"""
Utility functions for loading and working with Chicago crime data.

Common usage:
    from scripts.utils import load_data
    df = load_data()
"""
import pandas as pd
import os
try:
    from .config import *  # When imported as a module (Flask)
except ImportError:
    from config import * # When run directly from scripts/

def filter_valid_coordinates(df, lat_col="latitude", lon_col="longitude"):
    """
    Keep rows with non-null latitude/longitude inside Chicago bounds from config.

    Use this before spatial clustering (DBSCAN) or KNN on geographic coordinates.
    """
    lat_ok = df[lat_col].between(*VALID_LAT_RANGE)
    lon_ok = df[lon_col].between(*VALID_LON_RANGE)
    mask = df[lat_col].notna() & df[lon_col].notna() & lat_ok & lon_ok
    return df.loc[mask].copy()


def load_spatial_subset(
    columns=None,
    prefer_parquet=True,
    n_sample=None,
    random_state=42,
):
    """
    Load cleaned data with valid coordinates only.

    Parameters
    ----------
    columns : list[str] | None
        If set, return only these columns (plus lat/lon if omitted but needed).
    prefer_parquet : bool
        Passed to load_data.
    n_sample : int | None
        If set, draw this many rows after filtering (for heavy DBSCAN experiments).
    random_state : int
        RNG seed when n_sample is used.
    """
    read_cols = None
    if columns is not None:
        read_cols = list(dict.fromkeys(list(columns) + ["latitude", "longitude"]))
    df = load_data(prefer_parquet=prefer_parquet, columns=read_cols)
    df = filter_valid_coordinates(df)
    if columns is not None:
        cols = list(dict.fromkeys(list(columns) + ["latitude", "longitude"]))
        cols = [c for c in cols if c in df.columns]
        df = df[cols]
    if n_sample is not None and len(df) > n_sample:
        df = df.sample(n=n_sample, random_state=random_state)
    return df.reset_index(drop=True)


def load_data(prefer_parquet=True, columns=None):
    """
    Load cleaned Chicago crime data.

    Args:
        prefer_parquet: If True and parquet exists, load that. Otherwise CSV.
        columns: Optional list of columns to read (much faster for huge CSV).

    Returns:
        pandas.DataFrame containing cleaned crime data
    """
    if prefer_parquet and os.path.exists(CLEANED_PARQUET):
        print(f"Loading from {CLEANED_PARQUET}...")
        return pd.read_parquet(CLEANED_PARQUET, columns=columns)

    if os.path.exists(CLEANED_CSV):
        print(f"Loading from {CLEANED_CSV}...")
        read_kwargs = {"low_memory": False}
        if columns is not None:
            read_kwargs["usecols"] = columns
        return pd.read_csv(CLEANED_CSV, **read_kwargs)
    
    raise FileNotFoundError(
        f"No cleaned data found. Please run clean.py first.\n"
        f"Expected: {CLEANED_CSV} or {CLEANED_PARQUET}"
    )

def convert_to_parquet(force=False):
    """Convert cleaned CSV to parquet format for faster loading.
    
    Args:
        force: If True, regenerate parquet even if it already exists
    """
    if os.path.exists(CLEANED_PARQUET) and not force:
        print(f"Parquet already exists: {CLEANED_PARQUET}")
        print(f"Size: {os.path.getsize(CLEANED_PARQUET) / 1024**2:.1f} MB")
        return
    
    if not os.path.exists(CLEANED_CSV):
        raise FileNotFoundError(
            f"Cleaned CSV not found at {CLEANED_CSV}. Run clean.py first."
        )
    
    print(f"Converting {CLEANED_CSV} to parquet...")
    print(f"Original CSV size: {os.path.getsize(CLEANED_CSV) / 1024**2:.1f} MB")
    
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df.to_parquet(CLEANED_PARQUET, index=False)
    
    print(f"Saved to {CLEANED_PARQUET}")
    print(f"Parquet size: {os.path.getsize(CLEANED_PARQUET) / 1024**2:.1f} MB")
    print(f"Space saved: {(1 - os.path.getsize(CLEANED_PARQUET) / os.path.getsize(CLEANED_CSV)) * 100:.1f}%")

def get_data_info():
    """Print information about available data files."""
    print("Chicago Crime Data Files:")
    print("-" * 60)
    
    files = [
        ("Raw CSV", RAW_CSV),
        ("Cleaned CSV", CLEANED_CSV),
        ("Cleaned Parquet", CLEANED_PARQUET)
    ]
    
    for name, path in files:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024**2
            print(f"✓ {name:20s} {size_mb:8.1f} MB  {path}")
        else:
            print(f"✗ {name:20s} {'(not found)':>10s}  {path}")

if __name__ == "__main__":
    print("=" * 60)
    print("CHICAGO CRIME DATA UTILITIES")
    print("=" * 60)
    print()
    
    get_data_info()
    print()
    
    if os.path.exists(CLEANED_CSV) and not os.path.exists(CLEANED_PARQUET):
        print("Converting CSV to parquet...")
        print()
        convert_to_parquet()
    elif os.path.exists(CLEANED_PARQUET):
        print("Parquet file already exists. Nothing to do.")
    else:
        print("No cleaned data found. Run clean.py first.")
