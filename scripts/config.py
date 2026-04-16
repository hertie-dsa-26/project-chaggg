"""
Configuration file for Chicago crime data pipeline.
Contains all paths, API settings, and constants used across scripts.
"""
import os

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Go up one level
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
BATCH_DIR = os.path.join(RAW_DIR, "batches")

RAW_CSV = os.path.join(RAW_DIR, "chicago_crimes_2001_2025_raw.csv")
CLEANED_CSV = os.path.join(CLEANED_DIR, "chicago_crimes_cleaned.csv")
CLEANED_PARQUET = os.path.join(CLEANED_DIR, "chicago_crimes_cleaned.parquet")
PROGRESS_FILE = os.path.join(RAW_DIR, "download_progress.json")

# ── API Settings ──────────────────────────────────────────────────────────────
APP_TOKEN = "YOUR_APP_TOKEN_HERE"  # Get your token from Chicago Data portal
BASE_URL = "https://data.cityofchicago.org/resource/ijzp-q8t2.json"
BATCH_SIZE = 50000
END_DATE = "2026-01-01T00:00:00.000"  # Fetch records only up to the end of 2025

# ── Schema ────────────────────────────────────────────────────────────────────
EXPECTED_COLUMNS = [
    'id', 'case_number', 'date', 'block', 'iucr', 'primary_type',
    'description', 'location_description', 'arrest', 'domestic', 'beat',
    'district', 'ward', 'community_area', 'fbi_code', 'year', 'updated_on',
    'x_coordinate', 'y_coordinate', 'latitude', 'longitude', 'location'
]

# ── Data Filters ──────────────────────────────────────────────────────────────
VALID_LAT_RANGE = (41.624851, 42.07436)
VALID_LON_RANGE = (-87.968437, -87.397217)
YEAR_RANGE = (2002, 2025)

# ── Create Directories ────────────────────────────────────────────────────────
for directory in [DATA_DIR, RAW_DIR, CLEANED_DIR, BATCH_DIR]:
    os.makedirs(directory, exist_ok=True)
