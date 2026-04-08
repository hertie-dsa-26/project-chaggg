"""
Flask application for Chicago Crime Analysis.
"""
import json
from pathlib import Path

from flask import Flask, jsonify, render_template, request


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates", static_folder="static")

    geo_dir = Path(__file__).resolve().parent / "static" / "geo"
    hotspot_variants = {
        "meters": geo_dir / "hotspots_meters.geojson",
        "degrees": geo_dir / "hotspots_degrees.geojson",
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

    @app.route("/api/hotspots")
    def api_hotspots():
        variant = request.args.get("variant", "meters").strip().lower()
        path = hotspot_variants.get(variant)
        if path is None:
            return jsonify(
                {
                    "error": "unknown_variant",
                    "message": f"Unknown variant: {variant}",
                    "available": sorted(hotspot_variants.keys()),
                }
            ), 400
        if not path.exists():
            return jsonify(
                {
                    "error": "missing_geojson",
                    "message": f"GeoJSON not found: {path.name}",
                }
            ), 404
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return jsonify(data)

    return app
