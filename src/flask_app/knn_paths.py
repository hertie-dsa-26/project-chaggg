"""Paths and small helpers for KNN forecast artifacts under outputs/knn/."""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def knn_dir() -> Path:
    """
    Directory for KNN artifacts.

    Overridable for tests/CI via CHAGGG_KNN_DIR.
    """
    override = os.environ.get("CHAGGG_KNN_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return PROJECT_ROOT / "outputs" / "knn"


def knn_artifacts_present() -> bool:
    d = knn_dir()
    return (d / "forecast_predictions_k10.csv").is_file() and (
        d / "monthly_ca_train_2015_2022.parquet"
    ).is_file()


def metrics_path() -> Path:
    return knn_dir() / "forecast_metrics.json"


def load_forecast_metrics() -> dict:
    p = metrics_path()
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def predictions_csv_path(k: int) -> Path:
    return knn_dir() / f"forecast_predictions_k{k}.csv"
