"""
Run the full Chicago crime data pipeline: fetch → clean → convert to parquet.

Usage from project root:
    uv run python scripts/main.py --skip-download
"""
import os
import sys

# Add scripts directory to path so we can import the modules
sys.path.insert(0, os.path.dirname(__file__))

# Import pipeline modules
import fetch
import clean
from utils import convert_to_parquet, get_data_info
from config import RAW_CSV, CLEANED_CSV, CLEANED_PARQUET

def run_pipeline(skip_download=False):
    """
    Run the full data pipeline.
    
    Args:
        skip_download: If True, skip the download step (useful if you already have raw data)
    """
    print("=" * 70)
    print(" " * 15 + "CHICAGO CRIME DATA PIPELINE")
    print("=" * 70)
    print()
    
    # Step 1: Download
    if skip_download:
        print("STEP 1: Download - SKIPPED")
        print("-" * 70)
        if not os.path.exists(RAW_CSV):
            print(f"ERROR: Raw data not found at {RAW_CSV}")
            print("Cannot skip download step without existing raw data.")
            sys.exit(1)
        print(f"Using existing raw data: {RAW_CSV}")
        print()
    else:
        print("STEP 1: Downloading raw data")
        print("-" * 70)
        if os.path.exists(RAW_CSV):
            print(f"Raw data already exists: {RAW_CSV}")
            print("Skipping download. Delete file to re-download.")
        else:
            fetch.download_batches()
            fetch.combine_batches()
        print()
    
    # Step 2: Clean
    print("STEP 2: Cleaning data")
    print("-" * 70)
    cleaned_data_updated = False
    if os.path.exists(CLEANED_CSV):
        print(f"Cleaned data already exists: {CLEANED_CSV}")
        response = input("Do you want to re-run cleaning? (y/n): ").lower().strip()
        if response != 'y':
            print("Skipping cleaning step.")
        else:
            clean.main()
            cleaned_data_updated = True
    else:
        clean.main()
        cleaned_data_updated = True
    print()

    # Step 3: Convert to Parquet
    print("STEP 3: Converting to parquet")
    print("-" * 70)
    if cleaned_data_updated:
        # If we just cleaned the data, always regenerate parquet
        print("Regenerating parquet from updated CSV...")
        convert_to_parquet(force=True)
    elif os.path.exists(CLEANED_PARQUET):
        print(f"Parquet file already exists: {CLEANED_PARQUET}")
        print("Skipping conversion (cleaned CSV unchanged).")
    else:
        convert_to_parquet()
    print()
    
    # Summary
    print("=" * 70)
    print(" " * 20 + "PIPELINE COMPLETE!")
    print("=" * 70)
    print()
    get_data_info()
    print()
    print("Next steps:")
    print("  - Import data in your analysis with: from utils import load_data")
    print("  - Load the data with: df = load_data()")
    print()

if __name__ == "__main__":
    # Check if user wants to skip download (useful if raw data already exists)
    skip_download = False
    
    if len(sys.argv) > 1 and sys.argv[1] == "--skip-download":
        skip_download = True
    
    run_pipeline(skip_download=skip_download)