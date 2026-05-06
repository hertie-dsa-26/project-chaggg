"""
Convert Chicago community areas geojson to a static SVG background.
Run once. Output goes to static/img/chicago-outline.svg.
"""
import json
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).resolve().parents[3]
GEOJSON_PATH = PROJECT_ROOT / "src/flask_app/static/data/space/chicago_community_areas.geojson"
OUTPUT_PATH = PROJECT_ROOT / "src/flask_app/static/img/chicago-outline.svg"

# SVG canvas size (logical units; SVG itself scales)
SVG_W, SVG_H = 1000, 1200


def project_polygon(coords, bounds, w, h):
    """Map lon/lat polygon ring to SVG x/y, flipping y-axis."""
    min_lon, min_lat, max_lon, max_lat = bounds
    lon_span = max_lon - min_lon
    lat_span = max_lat - min_lat
    # Preserve aspect ratio: scale by the smaller axis
    scale = min(w / lon_span, h / lat_span) * 0.95
    # Center offsets
    ox = (w - lon_span * scale) / 2
    oy = (h - lat_span * scale) / 2
    pts = []
    for lon, lat in coords:
        x = (lon - min_lon) * scale + ox
        y = h - ((lat - min_lat) * scale + oy)  # flip y
        pts.append(f"{x:.2f},{y:.2f}")
    return " ".join(pts)


def get_bounds(features):
    """Compute bounding box across all features."""
    min_lon, min_lat = float("inf"), float("inf")
    max_lon, max_lat = float("-inf"), float("-inf")
    for f in features:
        geom = f["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            for ring in poly:
                for lon, lat in ring:
                    min_lon = min(min_lon, lon)
                    max_lon = max(max_lon, lon)
                    min_lat = min(min_lat, lat)
                    max_lat = max(max_lat, lat)
    return (min_lon, min_lat, max_lon, max_lat)


def main():
    with open(GEOJSON_PATH) as f:
        gj = json.load(f)
    features = gj["features"]
    bounds = get_bounds(features)

    polygons = []
    for feat in features:
        geom = feat["geometry"]
        polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
        for poly in polys:
            outer = poly[0]  # outer ring only; ignore holes for background
            pts = project_polygon(outer, bounds, SVG_W, SVG_H)
            polygons.append(f'<polygon points="{pts}" />')

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SVG_W} {SVG_H}" preserveAspectRatio="xMidYMid meet">
  <g fill="none" stroke="#22c55e" stroke-width="1" stroke-linejoin="round">
    {chr(10).join("    " + p for p in polygons)}
  </g>
</svg>
'''
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(svg)
    print(f"Wrote {OUTPUT_PATH} ({len(polygons)} polygons)")


if __name__ == "__main__":
    main()