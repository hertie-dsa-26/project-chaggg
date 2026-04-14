"""
Download Chicago crime data from the data portal.
Saves raw data to data/raw/chicago_crimes_2001_2025_raw.csv

Usage:
    uv run python scripts/fetch.py
"""
import requests
import pandas as pd
import time
import json
import os
from config import *

# ── Progress Tracking ─────────────────────────────────────────────────────────
def load_progress():
    """Load download progress from file."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"offset": 0, "batch_num": 1, "total_records": 0}

def save_progress(offset, batch_num, total):
    """Save download progress to file."""
    with open(PROGRESS_FILE, "w") as f:
        json.dump({"offset": offset, "batch_num": batch_num, "total_records": total}, f)

# ── API Fetch ─────────────────────────────────────────────────────────────────
def fetch_batch(offset, retries=3):
    """
    Fetch a single batch of records from the Chicago Data Portal API.
    Only fetches records up to END_DATE specified in config.
    
    Args:
        offset: Starting record number
        retries: Number of retry attempts on failure
    
    Returns:
        List of records (dicts)
    """
    params = {
        "$limit": BATCH_SIZE,
        "$offset": offset,
        "$order": "date ASC",
        "$where": f"date < '{END_DATE}'",  # Filter download to records until end of 2025
        "$$app_token": "H9ETY0ZCsP5vHUqNI6NE2FKG5"
    }
    
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(BASE_URL, params=params, timeout=60)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"  [Attempt {attempt}/{retries}] Failed: {e}")
            if attempt < retries:
                time.sleep(5 * attempt)
            else:
                raise

def align_columns(df):
    """Ensure dataframe has all expected columns in correct order."""
    for col in EXPECTED_COLUMNS:
        if col not in df.columns:
            df[col] = None
    return df[EXPECTED_COLUMNS]

# ── Download Functions ────────────────────────────────────────────────────────
def download_batches():
    """
    Download data in batches to the batches/ directory.
    Supports resuming if interrupted.
    """
    progress = load_progress()
    offset = progress["offset"]
    batch_num = progress["batch_num"]
    total = progress["total_records"]

    print(f"{'Resuming' if offset > 0 else 'Starting'} download "
          f"(batch={batch_num}, offset={offset}, saved={total})\n")

    while True:
        batch_file = os.path.join(BATCH_DIR, f"batch_{batch_num:04d}.csv")

        if os.path.exists(batch_file):
            print(f"Batch {batch_num} already exists, skipping...")
            offset += BATCH_SIZE
            batch_num += 1
            continue

        print(f"Fetching batch {batch_num} (offset={offset})...")
        batch = fetch_batch(offset)

        if not batch:
            print("No more data. Download complete.")
            break

        df = align_columns(pd.DataFrame(batch))
        df.to_csv(batch_file, index=False)

        total += len(batch)
        offset += BATCH_SIZE
        batch_num += 1
        save_progress(offset, batch_num, total)
        print(f"  -> {len(batch)} records. Total: {total}")
        time.sleep(0.5)

    if os.path.exists(PROGRESS_FILE):
        os.remove(PROGRESS_FILE)

def combine_batches():
    """
    Combine all batch files into a single raw CSV file.
    """
    if os.path.exists(RAW_CSV):
        print(f"Raw file already exists: {RAW_CSV}")
        return

    batch_files = sorted([
        os.path.join(BATCH_DIR, f)
        for f in os.listdir(BATCH_DIR) if f.endswith(".csv")
    ])
    
    if not batch_files:
        raise FileNotFoundError("No batch files found. Run download_batches() first.")

    print(f"\nMerging {len(batch_files)} batches into {RAW_CSV}...")

    with open(RAW_CSV, "w", encoding="utf-8") as out:
        for i, path in enumerate(batch_files):
            df = pd.read_csv(path, low_memory=False)
            df = align_columns(df)
            df.to_csv(out, index=False, header=(i == 0))
            print(f"  Merged {path} ({len(df)} rows)")

    print(f"\nDone! Raw data saved to: {RAW_CSV}")

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if os.path.exists(RAW_CSV):
        print(f"Raw data already exists at {RAW_CSV}")
        print("Delete it if you want to re-download.")
    else:
        download_batches()
        combine_batches()
