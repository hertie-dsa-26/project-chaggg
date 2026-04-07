"""
Download shared dataset snapshots from Google Drive.

For team consistency, we distribute fixed snapshots (2001–2025) via Drive rather
than downloading "up to today" from the live portal.

Issue #31 / #33:
- provide a fast Parquet option for consistent access across machines
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import gdown


RAW_FILE_ID_DEFAULT = "1HGIQVus5LLVN8ONsKPhN6pX1ViHQdrdJ"
RAW_OUTPUT_DEFAULT = "data/raw/chicago_crimes_2001_2025_raw.csv"

# Set this once the cleaned parquet is uploaded to the shared drive.
# Can also be provided via env var CHAGGG_CLEANED_PARQUET_FILE_ID.
CLEANED_PARQUET_FILE_ID_DEFAULT = None
CLEANED_PARQUET_OUTPUT_DEFAULT = "data/cleaned/chicago_crimes_cleaned.parquet"


def ensure_dirs() -> None:
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    Path("data/cleaned").mkdir(parents=True, exist_ok=True)


def download_drive_file(*, file_id: str, out_path: str, force: bool) -> None:
    if os.path.exists(out_path) and not force:
        print(f"File already exists at {out_path}, skipping.")
        return
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, out_path, quiet=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Download shared dataset snapshots from Drive.")
    parser.add_argument(
        "--raw-file-id",
        default=os.environ.get("CHAGGG_RAW_FILE_ID", RAW_FILE_ID_DEFAULT),
        help="Google Drive file id for the raw CSV snapshot.",
    )
    parser.add_argument(
        "--raw-out",
        default=RAW_OUTPUT_DEFAULT,
        help="Output path for the raw CSV.",
    )
    parser.add_argument(
        "--cleaned-parquet-file-id",
        default=os.environ.get("CHAGGG_CLEANED_PARQUET_FILE_ID", CLEANED_PARQUET_FILE_ID_DEFAULT),
        help="Google Drive file id for the cleaned Parquet snapshot (optional).",
    )
    parser.add_argument(
        "--cleaned-parquet-out",
        default=CLEANED_PARQUET_OUTPUT_DEFAULT,
        help="Output path for the cleaned Parquet.",
    )
    parser.add_argument(
        "--skip-raw",
        action="store_true",
        help="Skip downloading the raw CSV snapshot.",
    )
    parser.add_argument(
        "--skip-cleaned-parquet",
        action="store_true",
        help="Skip downloading the cleaned Parquet snapshot.",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist.")
    args = parser.parse_args()

    ensure_dirs()

    if not args.skip_raw:
        print("Downloading raw dataset snapshot (CSV)...")
        download_drive_file(file_id=args.raw_file_id, out_path=args.raw_out, force=args.force)
        print("Done.")

    if not args.skip_cleaned_parquet:
        if not args.cleaned_parquet_file_id:
            print(
                "No cleaned parquet file id configured yet. "
                "Set CHAGGG_CLEANED_PARQUET_FILE_ID or pass --cleaned-parquet-file-id."
            )
        else:
            print("Downloading cleaned dataset snapshot (Parquet)...")
            download_drive_file(
                file_id=args.cleaned_parquet_file_id,
                out_path=args.cleaned_parquet_out,
                force=args.force,
            )
            print("Done.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
