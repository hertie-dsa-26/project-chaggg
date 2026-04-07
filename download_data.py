"""
Download a deterministic Chicago Crimes snapshot (2001–2025) into this repo.

Issue #32: ensure everyone downloads the same raw data (do NOT download "up to today").

This script downloads a shared snapshot from Google Drive and performs a quick
sanity check that the dataset does not include records beyond 2025-12-31.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gdown
import pandas as pd


RAW_FILE_ID_DEFAULT = "1HGIQVus5LLVN8ONsKPhN6pX1ViHQdrdJ"
RAW_OUTPUT_DEFAULT = "data/raw/chicago_crimes_2001_2025_raw.csv"


def ensure_dirs() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)


def download_from_drive(*, file_id: str, out_path: str, force: bool) -> None:
    if os.path.exists(out_path) and not force:
        print(f"Raw dataset already exists at {out_path}, skipping.")
        return

    print("Downloading raw dataset snapshot from Google Drive...")
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)
    print("Done.")


def assert_cutoff_2025(*, csv_path: str, date_col: str = "date", sample_rows: int = 200_000) -> None:
    """
    Quick check: fail if any date is >= 2026-01-01.

    Uses chunked reading to avoid loading the whole file into memory.
    """
    cutoff = pd.Timestamp("2026-01-01")
    for chunk in pd.read_csv(csv_path, usecols=[date_col], parse_dates=[date_col], chunksize=sample_rows):
        if (chunk[date_col] >= cutoff).any():
            raise SystemExit(
                f"Dataset appears to include records from 2026 or later. "
                f"Expected a 2001–2025 snapshot. Please re-check the Drive file_id."
            )


def main() -> int:
    parser = argparse.ArgumentParser(description="Download raw Chicago crimes snapshot (2001–2025).")
    parser.add_argument("--raw-file-id", default=RAW_FILE_ID_DEFAULT, help="Google Drive file id.")
    parser.add_argument("--out", default=RAW_OUTPUT_DEFAULT, help="Output path for raw CSV.")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    parser.add_argument(
        "--skip-cutoff-check",
        action="store_true",
        help="Skip the 2025 cutoff sanity check (not recommended).",
    )
    args = parser.parse_args()

    ensure_dirs()
    download_from_drive(file_id=args.raw_file_id, out_path=args.out, force=args.force)

    if not args.skip_cutoff_check:
        assert_cutoff_2025(csv_path=args.out)
        print("Cutoff check passed (no records from 2026+).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
