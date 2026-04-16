"""
Run US-2.4 KNN forecasting end-to-end from prepared monthly CA parquet files.

Inputs: parquet produced by scripts/run_knn_prep.py --write-split
Outputs:
  - outputs/knn/forecast_predictions_k{K}.csv
  - outputs/knn/forecast_metrics.json

Run from repo root:
  PYTHONUNBUFFERED=1 uv run python scripts/run_knn_forecast.py
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.knn_scratch import SpatiotemporalKNN  # noqa: E402
from src.algorithms.data_prep import (  # noqa: E402
    MonthlyCommunityAreaTable,
    split_train_test_by_year,
)
from src.algorithms.metrics import mae, rmse  # noqa: E402


def _require_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


def _predict_df(model: SpatiotemporalKNN, df: pd.DataFrame) -> np.ndarray:
    # Keep as simple loop: test set is typically small (e.g., ~1–2k rows).
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    t = df["month_index"].to_numpy(dtype=float)
    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        out[i] = model.predict((lat[i], lon[i]), t[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Run from-scratch spatiotemporal KNN forecast")
    parser.add_argument(
        "--monthly",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca.parquet",
        help="Monthly parquet path (used as fallback if --train/--test are missing).",
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca_train_2015_2022.parquet",
        help="Train parquet path",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca_test_2023_2024.parquet",
        help="Test parquet path",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=project_root / "outputs" / "knn",
        help="Output directory",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="5,10,20",
        help="Comma-separated K values (e.g. '5,10,20')",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="meters",
        choices=["degrees", "meters"],
        help="Feature space for distance computations",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=0.0,
        help="Scale applied to month_index feature (units depend on --space)",
    )
    parser.add_argument(
        "--limit-train",
        type=int,
        default=0,
        help="Optional cap for train rows (0 = no cap)",
    )
    parser.add_argument(
        "--limit-test",
        type=int,
        default=0,
        help="Optional cap for test rows (0 = no cap)",
    )
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    if not ks:
        raise ValueError("No valid K values parsed from --k")

    if args.train.exists() and args.test.exists():
        print("Loading train/test parquet…", flush=True)
        train_df = pd.read_parquet(args.train)
        test_df = pd.read_parquet(args.test)
    elif args.monthly.exists():
        print("Train/test parquet not found; loading monthly parquet and splitting…", flush=True)
        monthly_df = pd.read_parquet(args.monthly)
        train_df, test_df = split_train_test_by_year(
            monthly=MonthlyCommunityAreaTable(df=monthly_df)
        )
    else:
        raise FileNotFoundError(
            "Could not find inputs. Expected either:\n"
            f" - train+test: {args.train} and {args.test}\n"
            f" - monthly:   {args.monthly}\n\n"
            "Create them with:\n"
            "  uv run python scripts/run_knn_prep.py --write-split\n"
        )

    required = ["lat", "lon", "month_index", "crime_count"]
    _require_cols(train_df, required, "train")
    _require_cols(test_df, required, "test")

    if args.limit_train and args.limit_train > 0:
        train_df = train_df.head(args.limit_train).reset_index(drop=True)
    if args.limit_test and args.limit_test > 0:
        test_df = test_df.head(args.limit_test).reset_index(drop=True)

    print(f"Train rows: {len(train_df):,}  Test rows: {len(test_df):,}", flush=True)

    args.outdir.mkdir(parents=True, exist_ok=True)

    y_true = test_df["crime_count"].to_numpy(dtype=float)
    results: dict[str, object] = {
        "run_config": {
            "train_path": str(args.train),
            "test_path": str(args.test),
            "k_values": ks,
            "space": args.space,
            "time_scale": args.time_scale,
            "limit_train": int(args.limit_train),
            "limit_test": int(args.limit_test),
        },
        "metrics_by_k": {},
    }

    for k in ks:
        print(f"\nTraining SpatiotemporalKNN(k={k}, space={args.space}, time_scale={args.time_scale})…", flush=True)
        model = SpatiotemporalKNN(k=k, space=args.space, time_scale=args.time_scale)
        t0 = time()
        model.train(train_df)
        t_train = time() - t0

        print("Predicting test set…", flush=True)
        t1 = time()
        y_pred = _predict_df(model, test_df)
        t_pred = time() - t1

        k_mae = mae(y_true, y_pred)
        k_rmse = rmse(y_true, y_pred)
        print(f"MAE={k_mae:.4f} RMSE={k_rmse:.4f} (train {t_train:.2f}s, predict {t_pred:.2f}s)", flush=True)

        pred_out = test_df.copy()
        pred_out["predicted_crime_count"] = y_pred
        pred_path = args.outdir / f"forecast_predictions_k{k}.csv"
        pred_out.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}", flush=True)

        results["metrics_by_k"][str(k)] = {
            "mae": float(k_mae),
            "rmse": float(k_rmse),
            "train_seconds": float(t_train),
            "predict_seconds": float(t_pred),
        }

    metrics_path = args.outdir / "forecast_metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"\nSaved metrics: {metrics_path}", flush=True)

    best_k = min(ks, key=lambda kk: float(results["metrics_by_k"][str(kk)]["rmse"]))
    combined = args.outdir / "predictions_knn.csv"
    shutil.copy(args.outdir / f"forecast_predictions_k{best_k}.csv", combined)
    print(f"Best K by RMSE: {best_k} → workflow alias {combined}", flush=True)


if __name__ == "__main__":
    main()

