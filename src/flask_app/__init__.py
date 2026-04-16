"""
Flask application for Chicago Crime Analysis.
"""
from flask import Flask, render_template

from .data import load_crime_data


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
    
    @app.route("/api/temporal")
    def api_temporal():
        from flask import jsonify, request
        df = app.config["CRIME_DF"]
        crime_type = request.args.get("type", "ALL")

        if crime_type != "ALL":
            df = df[df["primary_type"] == crime_type]

        # chart 1: total by year
        by_year = df.groupby("year").size().reset_index(name="count")
        by_year = by_year.sort_values("year")

        # chart 2: avg by month
        monthly = df.groupby(["year", "month"]).size().reset_index(name="count")
        avg_month = monthly.groupby("month")["count"].mean().round(0).reset_index()
        avg_month = avg_month.sort_values("month")

        # chart 3: hour x day_of_week heatmap
        day_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday", 3: "Thursday", 4: "Friday", 5: "Saturday", 6: "Sunday"}
        df_copy = df.copy()
        df_copy["day_name"] = df_copy["day_of_week"].map(day_map)
        heatmap = df_copy.groupby(["hour", "day_name"]).size().reset_index(name="count")
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        matrix = []
        for h in range(24):
            row = []
            for d in day_order:
                val = heatmap[(heatmap["hour"] == h) & (heatmap["day_name"] == d)]
                row.append(int(val["count"].values[0]) if len(val) > 0 else 0)
            matrix.append(row)

        # crime types for dropdown
        types = sorted(df["primary_type"].dropna().unique().tolist()) if crime_type == "ALL" else None

        result = {
            "by_year": {"years": by_year["year"].tolist(), "counts": by_year["count"].tolist()},
            "by_month": {"months": avg_month["month"].tolist(), "counts": avg_month["count"].tolist()},
            "heatmap": {"matrix": matrix, "days": day_order, "hours": list(range(24))},
        }
        if types:
            result["crime_types"] = types

        return jsonify(result)

    return app
