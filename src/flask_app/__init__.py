"""Flask application for Chicago Crime Analysis."""

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template, request

from . import knn_paths
from .data import load_crime_data


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    geo_dir = Path(__file__).resolve().parent / "static" / "geo"
    generated_dir = geo_dir / "generated"
    hotspot_variants = {
        "meters": (generated_dir / "hotspots_meters.geojson", geo_dir / "hotspots_meters.geojson"),
        "degrees": (
            generated_dir / "hotspots_degrees.geojson",
            geo_dir / "hotspots_degrees.geojson",
        ),
    }

    # Load dataset once at startup (keeps routes simple).
    app.config["CRIME_DF"] = load_crime_data()

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/viz/placeholder")
    def viz_placeholder():
        return render_template("viz_placeholder.html")

    @app.route("/dashboards/time")
    def dashboard_time():
        return render_template("dashboards/time.html", rows=len(app.config["CRIME_DF"]))

    @app.route("/dashboards/space")
    def dashboard_space():
        return render_template("dashboards/space.html", rows=len(app.config["CRIME_DF"]))

    @app.route("/dashboards/types")
    def dashboard_types():
        return render_template("dashboards/types.html", rows=len(app.config["CRIME_DF"]))

    @app.route("/viz/hotspots")
    def viz_hotspots():
        return render_template("hotspots.html")

    @app.route("/api/hotspots")
    def api_hotspots():
        variant = request.args.get("variant", "meters").strip().lower()
        paths = hotspot_variants.get(variant)
        if paths is None:
            return jsonify(
                {
                    "error": "unknown_variant",
                    "message": f"Unknown variant: {variant}",
                    "available": sorted(hotspot_variants.keys()),
                }
            ), 400
        path = None
        for p in paths:
            if p.exists():
                path = p
                break
        if path is None:
            return jsonify(
                {
                    "error": "missing_geojson",
                    "message": f"GeoJSON not found for variant: {variant}",
                }
            ), 404
        with path.open(encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)

    @app.route("/viz/knn")
    def viz_knn():
        k = int(request.args.get("k", 20))
        month = (request.args.get("month") or "").strip()
        if k not in (5, 10, 20):
            k = 20

        available = knn_paths.knn_artifacts_present()
        metrics_blob = knn_paths.load_forecast_metrics() if available else {}
        metrics_by_k = metrics_blob.get("metrics_by_k") or {}
        k_metrics = metrics_by_k.get(str(k), {})

        months: list[str] = []
        rows: list[dict] = []
        map_points: list[dict] = []

        pred_path = knn_paths.predictions_csv_path(k)
        if available and pred_path.is_file():
            df = pd.read_csv(pred_path)
            df["month"] = pd.to_datetime(df["month"])
            months = sorted(df["month"].dt.strftime("%Y-%m-%d").unique().tolist())
            if not month and months:
                month = months[-1]
            show = df[df["month"] == pd.to_datetime(month)] if month else df.iloc[0:0]
            rows = show.sort_values("community_area").to_dict("records")
            for _, r in show.iterrows():
                map_points.append(
                    {
                        "lat": float(r["lat"]),
                        "lon": float(r["lon"]),
                        "pred": float(r["predicted_crime_count"]),
                        "actual": float(r["crime_count"]),
                        "ca": float(r["community_area"]),
                    }
                )

        return render_template(
            "knn.html",
            k=k,
            month=month,
            months=months,
            rows=rows,
            map_points=map_points,
            available=available,
            k_metrics=k_metrics,
        )

    @app.route("/api/knn/predict")
    def api_knn_predict():
        from src.algorithms.knn_scratch import SpatiotemporalKNN

        if not knn_paths.knn_artifacts_present():
            return jsonify(
                {
                    "error": "knn_outputs_missing",
                    "message": "Run scripts/run_knn_prep.py --write-split then scripts/run_knn_forecast.py",
                }
            ), 503
        try:
            k = int(request.args.get("k", 10))
            time_scale = float(request.args.get("time_scale", 0.0))
            ca = float(request.args["community_area"])
            month = request.args["month"]
        except KeyError, ValueError:
            return jsonify(
                {
                    "error": "bad_request",
                    "message": "Required: community_area, month (YYYY-MM-DD). Optional: k, time_scale",
                }
            ), 400

        knn_dir = knn_paths.knn_dir()
        train_path = knn_dir / "monthly_ca_train_2015_2022.parquet"
        test_path = knn_dir / "monthly_ca_test_2023_2024.parquet"
        train_df = pd.read_parquet(train_path)
        test_df = pd.read_parquet(test_path)
        month_ts = pd.to_datetime(month)
        match = test_df[
            (pd.to_numeric(test_df["community_area"], errors="coerce") == ca)
            & (pd.to_datetime(test_df["month"]) == month_ts)
        ]
        if match.empty:
            return jsonify(
                {"error": "not_found", "message": "No test row for that community_area and month"}
            ), 404

        r = match.iloc[0]
        model = SpatiotemporalKNN(k=k, space="meters", time_scale=time_scale)
        model.train(train_df)
        pred = model.predict((float(r["lat"]), float(r["lon"])), float(r["month_index"]))
        return jsonify(
            {
                "k": k,
                "time_scale": time_scale,
                "community_area": ca,
                "month": month,
                "actual_crime_count": float(r["crime_count"]),
                "predicted_crime_count": pred,
                "lat": float(r["lat"]),
                "lon": float(r["lon"]),
                "month_index": float(r["month_index"]),
            }
        )

    return app
