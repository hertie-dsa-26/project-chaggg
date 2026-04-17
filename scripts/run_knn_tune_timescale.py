"""
Grid search time_scale × K for spatiotemporal KNN (US-2.4).

Reuses the same train/test parquet as scripts/run_knn_forecast.py.
Writes: outputs/knn/forecast_metrics_timescale_sweep.json

Run from repo root:
  PYTHONUNBUFFERED=1 uv run python scripts/run_knn_tune_timescale.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import time

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.algorithms.knn_scratch import SpatiotemporalKNN  # noqa: E402
from src.algorithms.metrics import mae, rmse  # noqa: E402


def _predict_df(model: SpatiotemporalKNN, df: pd.DataFrame) -> np.ndarray:
    lat = df["lat"].to_numpy(dtype=float)
    lon = df["lon"].to_numpy(dtype=float)
    t = df["month_index"].to_numpy(dtype=float)
    out = np.empty(len(df), dtype=float)
    for i in range(len(df)):
        out[i] = model.predict((lat[i], lon[i]), t[i])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune time_scale for KNN forecast")
    parser.add_argument(
        "--train",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca_train_2015_2022.parquet",
    )
    parser.add_argument(
        "--test",
        type=Path,
        default=project_root / "outputs" / "knn" / "monthly_ca_test_2023_2024.parquet",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=project_root / "outputs" / "knn" / "forecast_metrics_timescale_sweep.json",
    )
    parser.add_argument(
        "--k",
        type=str,
        default="5,10,20",
        help="Comma-separated K values",
    )
    parser.add_argument(
        "--time-scales",
        type=str,
        default="0,500,1000,2000,5000,10000",
        help="Comma-separated time_scale values (meters × month when --space meters)",
    )
    parser.add_argument(
        "--space",
        type=str,
        default="meters",
        choices=["degrees", "meters"],
    )
    args = parser.parse_args()

    ks = [int(x.strip()) for x in args.k.split(",") if x.strip()]
    scales = [float(x.strip()) for x in args.time_scales.split(",") if x.strip()]
    if not ks or not scales:
        raise ValueError("Need at least one K and one time_scale")

    train_df = pd.read_parquet(args.train)
    test_df = pd.read_parquet(args.test)
    y_true = test_df["crime_count"].to_numpy(dtype=float)

    results: list[dict[str, object]] = []
    best: dict[str, object] = {"rmse": float("inf")}

    for ts in scales:
        for k in ks:
            print(f"time_scale={ts} K={k} …", flush=True)
            model = SpatiotemporalKNN(k=k, space=args.space, time_scale=ts)
            t0 = time()
            model.train(train_df)
            t_train = time() - t0
            t1 = time()
            y_pred = _predict_df(model, test_df)
            t_pred = time() - t1
            err_mae = mae(y_true, y_pred)
            err_rmse = rmse(y_true, y_pred)
            row = {
                "time_scale": ts,
                "k": k,
                "mae": float(err_mae),
                "rmse": float(err_rmse),
                "train_seconds": float(t_train),
                "predict_seconds": float(t_pred),
            }
            results.append(row)
            if err_rmse < float(best["rmse"]):
                best = {**row, "rmse": float(err_rmse)}
            print(f"  MAE={err_mae:.4f} RMSE={err_rmse:.4f}", flush=True)

    payload = {
        "run_config": {
            "train_path": str(args.train),
            "test_path": str(args.test),
            "space": args.space,
            "k_values": ks,
            "time_scales": scales,
        },
        "results": results,
        "best_by_rmse": best,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Saved {args.out}", flush=True)
    print(f"Best (RMSE): {best}", flush=True)


if __name__ == "__main__":
    main()
