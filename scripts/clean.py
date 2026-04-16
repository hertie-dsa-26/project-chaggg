"""
Clean raw Chicago crime data.

Input:  data/raw/chicago_crimes_2001_2025_raw.csv
Output: data/cleaned/chicago_crimes_cleaned.csv

Usage:
    python clean.py
"""
import pandas as pd
import os
from config import *

# ── Loading ───────────────────────────────────────────────────────────────────
def load_raw_data():
    """Load raw data with error handling."""
    if not os.path.exists(RAW_CSV):
        raise FileNotFoundError(
            f"Raw data not found at {RAW_CSV}. "
            "Please run fetch.py first."
        )
    print(f"Loading raw data from {RAW_CSV}...")
    return pd.read_csv(RAW_CSV, low_memory=False)

# ── Cleaning Functions ────────────────────────────────────────────────────────
def filter_years(df):
    """Filter to valid year range."""
    print(f"\nFiltering to years {YEAR_RANGE[0]}-{YEAR_RANGE[1]}...")
    initial_count = len(df)
    df = df[df['year'].between(*YEAR_RANGE)]
    final_count = len(df)
    print(f"Records: {initial_count:,} → {final_count:,} ({initial_count - final_count:,} removed)")
    return df

def convert_types(df):
    """Convert date/time and categorical columns to appropriate types."""
    print("\nConverting data types...")
    
    # Split datetime into separate date and time columns
    print("  - Splitting date and time...")
    date_series = df['date'].str[:10]
    time_series = df['date'].str[11:19]
    
    df['date'] = pd.to_datetime(date_series, format="%Y-%m-%d")
    df['time'] = pd.to_datetime(time_series, format="%H:%M:%S").dt.time
    
    # Convert to nullable integers (Int64 can store NaN, int64 cannot)
    print("  - Converting district, ward, community_area to Int64...")
    df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")
    df["ward"] = pd.to_numeric(df["ward"], errors="coerce").astype("Int64")
    df["community_area"] = pd.to_numeric(df["community_area"], errors="coerce").astype("Int64")
    
    # Categorical encoding for memory efficiency
    print("  - Encoding categorical columns...")
    df["primary_type"] = df["primary_type"].astype("category")
    df["description"] = df["description"].astype("category")
    df["location_description"] = df["location_description"].astype("category")
    
    return df

def extract_temporal_features(df):
    """Extract year, month, day, hour, and day of week from date/time columns."""
    print("\nExtracting temporal features...")
    
    # From date column
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    
    # From time column (convert back to datetime to extract hour)
    df['hour'] = pd.to_datetime(df['time'], format='%H:%M:%S').dt.hour
    
    print(f"  - Extracted: year, month, day, hour, day_of_week")
    
    return df

def remove_na_coordinates(df):
    print("Removing rows with NaN values in longitude and latitude...")
    return df.dropna(subset=['longitude', 'latitude'])

def drop_redundant_columns(df):
    """Remove columns that duplicate information."""
    print("\nDropping redundant columns...")
    if 'location' in df.columns:
        print("  - Removing 'location' column (redundant with x/y coordinates)")
        df = df.drop(columns=['location'])
    return df

def remove_invalid_coordinates(df):
    """Remove rows with coordinates outside of valid Chicago bounds."""
    print("\nRemoving rows with invalid coordinates...")
    
    initial_count = len(df)
    valid_lat = df['latitude'].between(*VALID_LAT_RANGE)
    valid_lon = df['longitude'].between(*VALID_LON_RANGE)
    
    df = df[valid_lat & valid_lon]
    
    final_count = len(df)
    print(f"Records: {initial_count:,} → {final_count:,} ({initial_count - final_count:,} removed)")
    return df

# ── Validation & Reporting ────────────────────────────────────────────────────
def print_missing_summary(df, label=""):
    """Print missing value summary."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    summary = pd.DataFrame({
        "missing_count": missing,
        "missing_pct": missing_pct
    }).sort_values("missing_pct", ascending=False)
    
    print(f"\n{label}")
    print("-" * 60)
    display_summary = summary[summary['missing_count'] > 0]
    if len(display_summary) == 0:
        print("No missing values!")
    else:
        print(display_summary.to_string())
    
    return summary

def print_data_overview(df):
    """Print basic data overview."""
    print("\nData Overview:")
    print("-" * 60)
    print(f"Total records: {len(df):,}")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"Columns: {len(df.columns)}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")

# ── Save ──────────────────────────────────────────────────────────────────────
def save_cleaned_data(df):
    """Save cleaned data to CSV."""
    print(f"\nSaving cleaned data to {CLEANED_CSV}...")
    df.to_csv(CLEANED_CSV, index=False)
    print(f"Done! {len(df):,} records saved.")

# ── Main Pipeline ─────────────────────────────────────────────────────────────
def main():
    """Run full cleaning pipeline."""
    print("=" * 60)
    print("CHICAGO CRIME DATA CLEANING PIPELINE")
    print("=" * 60)
    
    # Load
    df = load_raw_data()
    print(f"Initial shape: {df.shape}")
    
    # Clean
    df = filter_years(df)
    df = convert_types(df)
    df = extract_temporal_features(df)
    df = remove_na_coordinates(df)
    df = remove_invalid_coordinates(df)
    df = drop_redundant_columns(df)
    
    # Validate & Report
    print_missing_summary(df, "Missing Values After Type Conversion:")
    print_data_overview(df)
    
    # Save
    save_cleaned_data(df)
    
    print("\n" + "=" * 60)
    print("CLEANING COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    main()
