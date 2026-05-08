"""
Flask application for Chicago Crime Analysis.
"""
from flask import Flask, render_template
import datetime as _dt
import math as _math
import numpy as _np
from flask import Flask, render_template, jsonify, request

from .data import load_crime_data
from .load_crime_artifacts import load_knn_arrays
from .estimators import (
    build_crime_type_slug_map,
    estimate_arrest_probability_naive_community_area,
    load_community_area_polygons,
    precompute_naive_stats,
)
from algorithms.knn_lrr import predict_arrest_probability
from algorithms.knn_sklearn import predict_arrest_probability_sklearn


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates")

    crime_df = load_crime_data()

    # Debug output
    print(f"\n{'='*50}")
    print(f"DATA LOADED: {len(crime_df)} rows, {len(crime_df.columns)} columns")
    if len(crime_df) > 0:
        print(f"Columns: {crime_df.columns.tolist()}")
        print(f"Date range: {crime_df['date'].min()} to {crime_df['date'].max()}")
    else:
        print("WARNING: Empty DataFrame loaded!")
    print(f"{'='*50}\n")

    app.config["CRIME_DF"] = crime_df
    app.config["CRIME_TYPE_SLUG_MAP"] = build_crime_type_slug_map(crime_df)

    print("Precomputing naive community area stats...")
    app.config["NAIVE_STATS"] = precompute_naive_stats(crime_df)
    print("Naive stats ready.")

    try:
        community_area_polygons = load_community_area_polygons()
        print(f"COMMUNITY AREA POLYGONS LOADED: {len(community_area_polygons)} areas")
    except FileNotFoundError:
        community_area_polygons = []
        print("WARNING: Community area polygons not found.")
    app.config["COMMUNITY_AREA_POLYGONS"] = community_area_polygons

    # Load precomputed KNN artifacts
    try:
        knn_artifacts = load_knn_arrays()
        print(f"\n{'='*50}")
        print(f"KNN ARTIFACTS LOADED: {len(knn_artifacts)} crime types")
        for slug, artifact in sorted(knn_artifacts.items()):
            print(f"  {slug:<40} n={artifact['label'].shape[0]:>7}")
        print(f"{'='*50}\n")
    except FileNotFoundError:
        knn_artifacts = {}
        print(f"\n{'='*50}")
        print("WARNING: KNN artifacts not found; continuing without precomputed bundles.")
        print(f"{'='*50}\n")
    app.config["KNN_ARTIFACTS"] = knn_artifacts

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/method")
    def method():
        return render_template("method.html")


    @app.route("/about")
    def about():
        return render_template("about.html")
    
    @app.route("/data-exploration")
    def overview():
        return render_template("overview.html")


    @app.route("/dashboards/time")
    def dashboard_time():
        return render_template(
            "dashboards/time.html",
            rows=len(app.config["CRIME_DF"]),
        )

    @app.route("/dashboards/space")
    def dashboard_space():
        return render_template(
            "dashboards/space.html",
            rows=len(app.config["CRIME_DF"]),
        )

    @app.route("/dashboards/types")
    def dashboard_types():
        return render_template(
            "dashboards/types.html",
            rows=len(app.config["CRIME_DF"]),
        )

    @app.route("/algorithm")
    def dashboard_algorithm():
        crime_types = sorted(app.config["KNN_ARTIFACTS"].keys())
        return render_template(
            "dashboards/algorithm.html",
            crime_types=crime_types,
        )
    
    @app.route("/codebook")
    def codebook():
        return render_template(
            "codebook.html")

    @app.route("/api/predict", methods=["POST"])
    def api_predict():
        payload = request.get_json(silent=True) or {}

        try:
            algorithm = payload["algorithm"]
            crime_type = payload["crime_type"]
            lat = float(payload["lat"])
            lon = float(payload["lon"])
            date_str = payload["date"]
            hour = int(payload["hour"])
            k = int(payload["k"])
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({"error": f"Invalid input: {e}"}), 400

        if algorithm == "naive_community_area":
            return jsonify({"error": "Naive community area baseline not implemented yet."}), 501
        if algorithm not in {"knn", "knn_sklearn"}:
            return jsonify({"error": f"Unknown algorithm: {algorithm}"}), 400

        artifacts = app.config["KNN_ARTIFACTS"]
        if crime_type not in artifacts:
            return jsonify({
                "error": f"Unknown crime type: {crime_type}",
                "available": sorted(artifacts.keys()),
            }), 400
        if not (0 <= hour <= 23):
            return jsonify({"error": "hour must be 0..23"}), 400
        if not (1 <= k <= 100):
            return jsonify({"error": "k must be 1..100"}), 400

        try:
            date = _dt.date.fromisoformat(date_str)
        except ValueError:
            return jsonify({"error": "date must be YYYY-MM-DD"}), 400
        if date.year != 2026:
            return jsonify({"error": "date must be in 2026"}), 400

        day_of_week = date.weekday()
        month = date.month
        day_of_year = date.timetuple().tm_yday

        if algorithm == "naive_community_area":
            probability, derived = estimate_arrest_probability_naive_community_area(
                stats=app.config["NAIVE_STATS"],
                crime_type_slug=crime_type,
                lat=lat,
                lon=lon,
                hour=hour,
                day_of_week=day_of_week,
                month=month,
                polygons=app.config["COMMUNITY_AREA_POLYGONS"],
                slug_map=app.config["CRIME_TYPE_SLUG_MAP"],
            )
            return jsonify({
                "probability": probability,
                "algorithm": "naive_community_area",
                "k": None,
                "crime_type": crime_type,
                "n_total": len(app.config["CRIME_DF"]),
                "derived": derived,
            })

        def _sincos(value, period):
            angle = 2 * _math.pi * value / period
            return _math.sin(angle), _math.cos(angle)

        hour_sin, hour_cos = _sincos(hour, 24)
        dow_sin, dow_cos = _sincos(day_of_week, 7)
        month_sin, month_cos = _sincos(month, 12)
        doy_sin, doy_cos = _sincos(day_of_year, 365)

        query_raw = _np.array([
            lat, lon,
            hour_sin, hour_cos,
            dow_sin, dow_cos,
            month_sin, month_cos,
            doy_sin, doy_cos,
        ], dtype=float)

        predictor = (
            predict_arrest_probability_sklearn
            if algorithm == "knn_sklearn"
            else predict_arrest_probability
        )
        try:
            probability = predictor(
                artifact=artifacts[crime_type],
                query_raw=query_raw,
                k=k,
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400

        return jsonify({
            "probability": probability,
            "algorithm": algorithm,
            "k": k,
            "crime_type": crime_type,
            "n_total": int(artifacts[crime_type]["label"].shape[0]),
            "derived": {"day_of_week": day_of_week, "day_of_year": day_of_year},
        })
    
    @app.route("/dashboards")
    def dashboards():
        return render_template("dashboards.html")
    
    @app.route("/api/temporal")
    def api_temporal():
        from flask import jsonify, request
        df = app.config["CRIME_DF"]
        crime_type = request.args.get("type", "ALL")

        # crime types for dropdown (before filtering)
        types = sorted(df["primary_type"].dropna().unique().tolist()) if crime_type == "ALL" else None

        if crime_type != "ALL":
            df = df[df["primary_type"] == crime_type]

        # chart 1: total by year
        by_year = df.groupby("year").size().reset_index(name="count").sort_values("year")

        # chart 2: avg by month
        avg_month = (
            df.groupby(["year", "month"]).size().reset_index(name="count")
            .groupby("month")["count"].mean().round(0).reset_index()
            .sort_values("month")
        )

        # chart 3: hour x day_of_week heatmap
        day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        day_order = list(day_map.values())
        heatmap = (
            df.assign(day_name=df["day_of_week"].map(day_map))
            .groupby(["hour", "day_name"]).size().unstack(fill_value=0)
            .reindex(index=range(24), columns=day_order, fill_value=0)
        )
        matrix = heatmap.values.tolist()

        result = {
            "by_year": {"years": by_year["year"].tolist(), "counts": by_year["count"].tolist()},
            "by_month": {"months": avg_month["month"].tolist(), "counts": avg_month["count"].tolist()},
            "heatmap": {"matrix": matrix, "days": day_order, "hours": list(range(24))},
        }
        if types:
            result["crime_types"] = types

        return jsonify(result)

    return app
