"""
DBSCAN hotspot pipeline: tune epsilon/min_samples, binary metrics, save artifacts.

Run from repository root (after clean data exists):
    uv run python scripts/run_dbscan_hotspots.py

Improvement (default): negatives are drawn from lat/lon grid cells with no
test-period crime, instead of only "far from a crime subsample" — fairer
binary evaluation. Use --negative-strategy distance for the older behavior.

Optional caps keep runtime reasonable on large Chicago extracts; raise them
for final reporting.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt

from config import VALID_LAT_RANGE, VALID_LON_RANGE
from src.algorithms.dbscan_hotspots import (
    extract_coordinates,
    grid_search_dbscan,
    k_distance_curve,
    plot_hotspots,
    plot_k_distance_curve,
    predict_hotspot_labels,
    project_latlon_to_meters,
    sample_negative_points,
    sample_negative_points_sparse_grid,
    save_cluster_boundaries,
    save_metrics_json,
)
from utils import filter_valid_coordinates, load_data


def main() -> None:
    parser = argparse.ArgumentParser(description="DBSCAN crime hotspots (sklearn)")
    parser.add_argument(
        "--space",
        choices=("degrees", "meters"),
        default="degrees",
        help="Run DBSCAN in lat/lon degrees (default) or projected meters (UTM16N).",
    )
    parser.add_argument(
        "--k-distance-plot",
        action="store_true",
        help="Save a k-distance curve plot (helps justify eps selection).",
    )
    parser.add_argument(
        "--k-distance-k",
        type=int,
        default=50,
        help="k for k-distance curve (often equals min_samples).",
    )
    parser.add_argument(
        "--max-train",
        type=int,
        default=45_000,
        help="Max training incidents (subsampled). Raise e.g. 120000 for final report.",
    )
    parser.add_argument(
        "--test-crimes-max",
        type=int,
        default=4_000,
        help="Max positive test points (2023–2024 crimes)",
    )
    parser.add_argument(
        "--n-neg",
        type=int,
        default=4_000,
        help="Number of negative test points (strategy: sparse_grid or distance)",
    )
    parser.add_argument(
        "--negative-strategy",
        choices=("sparse_grid", "distance"),
        default="sparse_grid",
        help="How to sample y=0 locations (default: sparse_grid)",
    )
    parser.add_argument(
        "--grid-lat-bins",
        type=int,
        default=32,
        help="Lat bins for sparse_grid histogram",
    )
    parser.add_argument(
        "--grid-lon-bins",
        type=int,
        default=32,
        help="Lon bins for sparse_grid histogram",
    )
    parser.add_argument(
        "--max-crimes-per-cell",
        type=int,
        default=0,
        help="Sparse grid: allow negatives in cells with at most this many test crimes (0=empty only)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress messages during grid search",
    )
    parser.add_argument(
        "--min-sep-deg",
        type=float,
        default=0.015,
        help="distance strategy: min separation in degrees vs all 2023–2024 crime points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <repo>/outputs/dbscan",
    )
    parser.add_argument(
        "--export-flask-static",
        action="store_true",
        help="Copy resulting cluster_boundaries.geojson into Flask static/geo for demo.",
    )
    args = parser.parse_args()
    out_dir = args.output_dir or (project_root / "outputs" / "dbscan")

    # Only needed columns: avoids loading the full ~1.7GB-wide table into memory.
    print("Loading cleaned rows (latitude, longitude, year, date)…", flush=True)
    df = load_data(columns=["latitude", "longitude", "year", "date"])
    print(f"  loaded {len(df):,} rows; filtering valid coordinates…", flush=True)
    df = filter_valid_coordinates(df)
    print(f"  valid coords: {len(df):,} rows", flush=True)

    if "year" not in df.columns:
        df = df.copy()
        df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year

    train_df = df[df["year"].between(2015, 2022)]
    test_crimes_df = df[df["year"].between(2023, 2024)]

    train_latlon = extract_coordinates(train_df)
    print(f"Train window 2015–2022: {len(train_latlon):,} points", flush=True)
    if len(train_latlon) > args.max_train:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(train_latlon), size=args.max_train, replace=False)
        train_latlon = train_latlon[idx]
        print(f"  subsampled to {args.max_train:,} for DBSCAN speed", flush=True)

    test_period_latlon_full = extract_coordinates(test_crimes_df)
    crime_test_latlon = test_period_latlon_full
    if len(crime_test_latlon) > args.test_crimes_max:
        rng = np.random.default_rng(43)
        idx = rng.choice(len(crime_test_latlon), size=args.test_crimes_max, replace=False)
        crime_test_latlon = crime_test_latlon[idx]

    if args.negative_strategy == "sparse_grid":
        print(
            f"Negatives: sparse grid ({args.grid_lat_bins}×{args.grid_lon_bins}), "
            f"max {args.max_crimes_per_cell} crimes/cell…",
            flush=True,
        )
        neg_xy = sample_negative_points_sparse_grid(
            test_period_latlon_full,
            n_sample=args.n_neg,
            lat_range=VALID_LAT_RANGE,
            lon_range=VALID_LON_RANGE,
            n_lat_bins=args.grid_lat_bins,
            n_lon_bins=args.grid_lon_bins,
            max_count_in_cell=args.max_crimes_per_cell,
            random_state=44,
        )
    else:
        print("Negatives: distance from full test-period crime set…", flush=True)
        neg_xy = sample_negative_points(
            test_period_latlon_full,
            n_sample=args.n_neg,
            lat_range=VALID_LAT_RANGE,
            lon_range=VALID_LON_RANGE,
            min_sep_deg=args.min_sep_deg,
            random_state=44,
        )

    xy_test = np.vstack([crime_test_latlon, neg_xy])
    y_test = np.concatenate(
        [
            np.ones(len(crime_test_latlon), dtype=np.int64),
            np.zeros(len(neg_xy), dtype=np.int64),
        ]
    )

    print("Parameter grid (9 combos) — first DBSCAN fit may take 1–3 min…", flush=True)
    if args.space == "degrees":
        train_xy_for_fit = train_latlon
        train_latlon_for_bounds = None
        eps_values = [0.01, 0.05, 0.1]
    else:
        # Cluster in meters for more meaningful eps; keep boundaries in WGS84 for mapping.
        print("Projecting training points to meters (UTM16N) for DBSCAN…", flush=True)
        train_xy_for_fit = project_latlon_to_meters(train_latlon, dst_crs="EPSG:32616")
        train_latlon_for_bounds = train_latlon
        eps_values = [500, 1500, 3000]

    if args.k_distance_plot:
        print(
            f"Computing k-distance curve (k={args.k_distance_k})…",
            flush=True,
        )
        curve = k_distance_curve(train_xy_for_fit, k=args.k_distance_k)
        unit = "meters" if args.space == "meters" else "degrees"
        plot_k_distance_curve(
            curve,
            out_dir / "k_distance_plot.png",
            title=f"k-distance curve (k={args.k_distance_k}, space={args.space}, unit={unit})",
        )

    result = grid_search_dbscan(
        train_xy_for_fit,
        train_latlon_for_bounds,
        xy_test,
        y_test,
        eps_values=eps_values,
        min_samples_values=[10, 50, 100],
        verbose=not args.quiet,
    )
    y_pred = predict_hotspot_labels(xy_test, result.boundaries)

    out_dir.mkdir(parents=True, exist_ok=True)
    boundaries_path = out_dir / "cluster_boundaries.geojson"
    save_cluster_boundaries(result.boundaries, boundaries_path)

    if args.export_flask_static:
        flask_geo_dir = (
            project_root / "src" / "flask_app" / "static" / "geo" / "generated"
        )
        flask_geo_dir.mkdir(parents=True, exist_ok=True)
        target = flask_geo_dir / f"hotspots_{args.space}.geojson"
        shutil.copyfile(boundaries_path, target)
        print(f"Exported Flask static GeoJSON: {target}", flush=True)

    preds_df = pd.DataFrame(
        {
            "latitude": xy_test[:, 0],
            "longitude": xy_test[:, 1],
            "y_true": y_test,
            "y_pred": y_pred,
        }
    )
    preds_df.to_csv(out_dir / "binary_predictions.csv", index=False)

    metrics_payload = {
        "best_eps": result.best_eps,
        "best_min_samples": result.best_min_samples,
        "metrics": result.best_metrics,
        "grid_search": result.all_runs.to_dict(orient="records"),
        "train_years": "2015–2022",
        "test_label_years_for_crimes": "2023–2024",
        "run_config": {
            "columns_loaded": ["latitude", "longitude", "year", "date"],
            "space": args.space,
            "k_distance_plot": bool(args.k_distance_plot),
            "k_distance_k": int(args.k_distance_k),
            "negative_strategy": args.negative_strategy,
            "grid_lat_bins": args.grid_lat_bins,
            "grid_lon_bins": args.grid_lon_bins,
            "max_crimes_per_cell": args.max_crimes_per_cell,
            "min_sep_deg": args.min_sep_deg,
            "max_train_cap": args.max_train,
            "test_crimes_positive_cap": args.test_crimes_max,
            "n_negatives": args.n_neg,
            "random_seeds": {
                "train_subsample": 42,
                "test_positive_subsample": 43,
                "negatives": 44,
            },
        },
        "notes": (
            "y_true=1: subsampled 2023–2024 crime locations. y_true=0: default strategy "
            "sparse_grid draws points inside histogram cells with at most max_crimes_per_cell "
            "incidents (full test-period histogram). "
            "Use --negative-strategy distance for min-distance negatives."
        ),
    }
    save_metrics_json(metrics_payload, out_dir / "metrics.json")

    fig = plot_hotspots(
        train_latlon,
        result.train_labels,
        result.boundaries,
        out_path=out_dir / "hotspots_train.png",
    )
    plt.close(fig)

    print("Outputs written to:", out_dir.resolve())
    print(
        "Best eps=", result.best_eps,
        "min_samples=", result.best_min_samples,
        result.best_metrics,
    )


if __name__ == "__main__":
    main()
