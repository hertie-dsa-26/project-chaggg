"""Paths and small helpers for KNN forecast artifacts under outputs/knn/."""

from __future__ import annotations

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNN_DIR = PROJECT_ROOT / "outputs" / "knn"


def knn_artifacts_present() -> bool:
    return (KNN_DIR / "forecast_predictions_k10.csv").is_file() and (
        KNN_DIR / "monthly_ca_train_2015_2022.parquet"
    ).is_file()


def metrics_path() -> Path:
    return KNN_DIR / "forecast_metrics.json"


def load_forecast_metrics() -> dict:
    p = metrics_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def predictions_csv_path(k: int) -> Path:
    return KNN_DIR / f"forecast_predictions_k{k}.csv"
