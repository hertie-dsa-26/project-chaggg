#!/usr/bin/env bash
# US-2.4 KNN end-to-end demo (workflow Adım 10).
# Requires cleaned data: data/cleaned/chicago_crimes_cleaned.parquet (or .csv).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== KNN demo: monthly prep + train/test split + forecast =="
uv run python scripts/run_knn_prep.py --write-split
uv run python scripts/run_knn_forecast.py --k 5,10,20 --space meters --time-scale 0

echo ""
echo "Artifacts under outputs/knn/:"
echo "  - predictions_knn.csv   (copy of best-K forecast by RMSE)"
echo "  - forecast_metrics.json"
echo ""
echo "Optional: time_scale grid → uv run python scripts/run_knn_tune_timescale.py"
echo "Flask (example): CHAGGG_SKIP_DATA_LOAD=1 uv run flask --app src.flask_app:create_app run"
echo "  then open http://127.0.0.1:5000/viz/knn"
