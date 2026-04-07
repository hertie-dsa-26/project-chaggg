from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert cleaned CSV to Parquet for faster distribution.")
    parser.add_argument(
        "--in",
        dest="in_path",
        default="data/cleaned/chicago_crimes_cleaned.csv",
        help="Input cleaned CSV path.",
    )
    parser.add_argument(
        "--out",
        dest="out_path",
        default="data/cleaned/chicago_crimes_cleaned.parquet",
        help="Output Parquet path.",
    )
    args = parser.parse_args()

    in_path = Path(args.in_path)
    out_path = Path(args.out_path)
    if not in_path.exists():
        raise SystemExit(f"Input CSV not found at {in_path}")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path, low_memory=False)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {out_path} (rows: {len(df)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

