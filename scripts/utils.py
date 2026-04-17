"""
Utility functions for loading and working with Chicago crime data.

Common usage:
    from scripts.utils import load_data
    df = load_data()
"""

import pandas as pd
import os
import math

try:
    from .config import *  # When imported as a module (Flask)
except ImportError:
    from config import * # When run directly from scripts/

def filter_valid_coordinates(df, lat_col="latitude", lon_col="longitude"):
    """
    Keep rows with non-null latitude/longitude inside Chicago bounds from config.

    Use this before spatial clustering or KNN on geographic coordinates.
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
        If set, draw this many rows after filtering (for heavy clustering experiments).
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

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Parameters:
        lat1, lon1: Latitude and longitude of point 1 (in decimal degrees)
        lat2, lon2: Latitude and longitude of point 2 (in decimal degrees)
    
    Returns:
        Distance in kilometers
    """
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371  # Earth's radius in kilometers
    return R * c


def get_spatial_bounds(df):
    """
    Calculate spatial bounds and maximum distance for normalization.
    
    Parameters:
        df: DataFrame with 'latitude' and 'longitude' columns
    
    Returns:
        dict with 'max_distance' (km) and 'bounds' (lat/lon ranges)
    """
    max_lat, min_lat = df['latitude'].max(), df['latitude'].min()
    max_lon, min_lon = df['longitude'].max(), df['longitude'].min()
    max_distance = haversine(min_lat, min_lon, max_lat, max_lon)
    
    return {
        'max_distance': max_distance,
        'bounds': {
            'lat': (min_lat, max_lat),
            'lon': (min_lon, max_lon)
        }
    }

def cyclical_distance(value1, value2, max_value):
    """
    Calculate distance between two cyclical values using sine-cosine encoding.
    
    Parameters:
        value1, value2: The cyclical values (e.g., hour=23, hour=1)
        max_value: Maximum value in the cycle (e.g., 24 for hours, 7 for day_of_week)
    
    Returns:
        Distance between 0 and 1
    """
    # Convert to angles on unit circle
    angle1 = 2 * math.pi * value1 / max_value
    angle2 = 2 * math.pi * value2 / max_value
    
    # Compute sin/cos coordinates
    sin1, cos1 = math.sin(angle1), math.cos(angle1)
    sin2, cos2 = math.sin(angle2), math.cos(angle2)
    
    # Euclidean distance on unit circle
    distance = math.sqrt((sin1 - sin2)**2 + (cos1 - cos2)**2)
    
    # Normalize to [0, 1] (max distance on unit circle is 2)
    return distance / 2.0

def temporal_distance(month1, hour1, dow1, month2, hour2, dow2):
    """
    Calculate normalized temporal distance between two crimes.
    Uses cyclical encoding for month, hour, and day_of_week.
    
    Parameters:
        month1, hour1, dow1: Time components of crime 1
        month2, hour2, dow2: Time components of crime 2
        (month: 1-12, dow = day_of_week: 0=Monday, 6=Sunday)
    
    Returns:
        Temporal distance score between 0 and 1
    """
    # Cyclical features
    month_dist = cyclical_distance(month1, month2, 12)     # months range 1-12
    hour_dist = cyclical_distance(hour1, hour2, 24)        # hours range 0-23
    dow_dist = cyclical_distance(dow1, dow2, 7)            # day_of_week range 0-6

    # Average the four normalized differences
    return (month_dist + hour_dist + dow_dist) / 3

def combined_distance(crime1, crime2, max_distance, alpha=0.5, beta=0.5):
    """
    Combined spatial and temporal distance between two crimes.
    
    Parameters:
        crime1, crime2: dicts or rows with keys 'latitude', 'longitude', 
                        'month', 'hour', 'day_of_week'
        alpha: weight for spatial distance
        beta: weight for temporal distance
    
    Returns:
        Combined distance score
    """
    spatial = haversine(crime1['latitude'], crime1['longitude'], 
                        crime2['latitude'], crime2['longitude'])
    temporal = temporal_distance(crime1['month'], crime1['hour'], crime1['day_of_week'],
                                 crime2['month'], crime2['hour'], crime2['day_of_week'])
    
    # Normalise spatial to [0, 1] scale using max_distance in dataset
    spatial_normalized = spatial / max_distance 
    
    return alpha * spatial_normalized + beta * temporal

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
