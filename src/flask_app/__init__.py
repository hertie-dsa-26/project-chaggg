"""
Flask application for Chicago Crime Analysis.
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from .data import load_crime_data


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    # Load dataset once at startup (keeps routes simple).
    app.config["CRIME_DF"] = load_crime_data()

    geo_dir = Path(__file__).resolve().parent / "static" / "geo"
    generated_dir = geo_dir / "generated"
    hotspot_variants = {
        "meters": (generated_dir / "hotspots_meters.geojson", geo_dir / "hotspots_meters.geojson"),
        "degrees": (generated_dir / "hotspots_degrees.geojson", geo_dir / "hotspots_degrees.geojson"),
    }

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/viz/placeholder")
    def viz_placeholder():
        return render_template("viz_placeholder.html")

    @app.route("/viz/hotspots")
    def viz_hotspots():
        return render_template("hotspots.html")

    @app.route("/dashboards/time")
    def dashboard_time():
        return render_template("dashboards/time.html", rows=len(app.config["CRIME_DF"]))

    @app.route("/dashboards/space")
    def dashboard_space():
        return render_template("dashboards/space.html", rows=len(app.config["CRIME_DF"]))

    @app.route("/dashboards/types")
    def dashboard_types():
        return render_template("dashboards/types.html", rows=len(app.config["CRIME_DF"]))

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
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)

    return app
