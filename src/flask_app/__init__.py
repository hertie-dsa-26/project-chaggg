"""
Flask application for Chicago Crime Analysis.
"""
from flask import Flask, render_template


def create_app():
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="templates")

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/about")
    def about():
        return render_template("about.html")

    @app.route("/viz/placeholder")
    def viz_placeholder():
        return render_template("viz_placeholder.html")

    return app
