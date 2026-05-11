from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
from scripts.utils import normalise_crime_type

GEOJSON_PATH = Path(__file__).resolve().parent / "static" / "data" / "space" / "chicago_community_areas.geojson"

MIN_COUNT_TEMPORAL = 20
MIN_COUNT_CRIME_AREA = 20
MIN_COUNT_AREA_ONLY = 10


def _point_in_ring(point: tuple[float, float], ring: list[tuple[float, float]]) -> bool:
    """Ray-casting point-in-polygon test for a single ring."""
    x, y = point
    inside = False
    n = len(ring)
    for i in range(n):
        x_i, y_i = ring[i]
        x_j, y_j = ring[(i + 1) % n]
        intersects = ((y_i > y) != (y_j > y))
        if intersects:
            x_intersect = x_i + (y - y_i) * (x_j - x_i) / (y_j - y_i)
            if x < x_intersect:
                inside = not inside
    return inside


def _point_in_polygon(point: tuple[float, float], polygon: list[list[tuple[float, float]]]) -> bool:
    """Return True if point is inside polygon and outside any holes."""
    if not polygon or not _point_in_ring(point, polygon[0]):
        return False
    for hole in polygon[1:]:
        if _point_in_ring(point, hole):
            return False
    return True


def load_community_area_polygons(geojson_path: Path = GEOJSON_PATH) -> list[tuple[int, list[list[list[tuple[float, float]]]]]]:
    """Load community area polygons from the web app's geojson file."""
    if not geojson_path.exists():
        raise FileNotFoundError(f"Community area geojson not found at {geojson_path}")

    with geojson_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    polygons: list[tuple[int, list[list[list[tuple[float, float]]]]]] = []
    for feature in data.get("features", []):
        props = feature.get("properties", {})
        area_str = props.get("area_numbe") or props.get("area_num_1") or props.get("community")
        if area_str is None:
            continue
        try:
            area_id = int(area_str)
        except (TypeError, ValueError):
            continue

        geometry = feature.get("geometry", {})
        geom_type = geometry.get("type")
        coords = geometry.get("coordinates", [])

        if geom_type == "Polygon":
            polygon = [
                [tuple(coord) for coord in ring]
                for ring in coords
            ]
            polygons.append((area_id, [polygon]))
        elif geom_type == "MultiPolygon":
            multi_polygon = []
            for polygon in coords:
                multi_polygon.append(
                    [
                        [tuple(coord) for coord in ring]
                        for ring in polygon
                    ]
                )
            polygons.append((area_id, multi_polygon))

    return polygons


def lookup_community_area(lat: float, lon: float, polygons: list[tuple[int, Any]]) -> int | None:
    """Return the community area id for a lat/lon point, or None if unknown."""
    point = (lon, lat)
    for area_id, multi_polygon in polygons:
        for polygon in multi_polygon:
            if _point_in_polygon(point, polygon):
                return area_id
    return None


def build_crime_type_slug_map(df: pd.DataFrame) -> dict[str, str]:
    """Build a mapping from normalized crime-type slug to the raw primary_type string."""
    cols = {col.lower(): col for col in df.columns}
    primary_col = cols.get("primary_type") or cols.get("primary type")
    if primary_col is None:
        raise ValueError("Crime dataframe is missing a primary_type column")

    slug_map: dict[str, str] = {}
    for primary_type in df[primary_col].dropna().unique():
        slug = normalise_crime_type(str(primary_type))
        if slug not in slug_map:
            slug_map[slug] = str(primary_type)
    return slug_map


def _resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    cols = {col.lower(): col for col in df.columns}
    return {
        "primary_type": cols.get("primary_type") or cols.get("primary type"),
        "arrest": cols.get("arrest"),
        "community_area": cols.get("community_area") or cols.get("community area"),
        "hour": cols.get("hour"),
        "day_of_week": cols.get("day_of_week"),
        "month": cols.get("month"),
    }


def precompute_naive_stats(df: pd.DataFrame) -> dict[str, Any]:
    """Precompute grouped arrest-rate tables so each prediction is an O(1) lookup."""
    cols = _resolve_columns(df)
    if cols["primary_type"] is None or cols["arrest"] is None:
        raise ValueError("Crime dataframe is missing required columns")

    df2 = df.assign(_a=df[cols["arrest"]].astype(int))

    def _agg(group_cols: list[str]) -> dict:
        g = df2.groupby(group_cols)["_a"].agg(count="count", rate="mean")
        return {k: (int(v["count"]), float(v["rate"])) for k, v in g.iterrows()}

    stats: dict[str, Any] = {
        "global_rate": float(df2["_a"].mean()),
        "crime": _agg([cols["primary_type"]]),
    }
    if cols["community_area"]:
        stats["area"] = _agg([cols["community_area"]])
        stats["area_crime"] = _agg([cols["primary_type"], cols["community_area"]])
        if cols["hour"] and cols["day_of_week"] and cols["month"]:
            stats["temporal"] = _agg([
                cols["primary_type"], cols["community_area"],
                cols["hour"], cols["day_of_week"], cols["month"],
            ])
    return stats


def estimate_arrest_probability_naive_community_area(
    stats: dict[str, Any],
    crime_type_slug: str,
    lat: float,
    lon: float,
    hour: int,
    day_of_week: int,
    month: int,
    polygons: list[tuple[int, Any]],
    slug_map: dict[str, str],
) -> tuple[float, dict[str, Any]]:
    """Estimate P(arrest) using precomputed community-area arrest-rate tables."""
    crime_type = slug_map.get(crime_type_slug)
    if crime_type is None:
        raise ValueError(f"Unknown crime type slug: {crime_type_slug}")

    community_area = lookup_community_area(lat, lon, polygons)

    crime_count = stats["crime"].get(crime_type, (0, 0.0))[0]
    derived: dict[str, Any] = {
        "community_area": community_area,
        "crime_type": crime_type_slug,
        "fallback": None,
        "n_matches": 0,
        "total_count": crime_count,
    }

    if community_area is not None:
        temporal_key = (crime_type, community_area, hour, day_of_week, month)
        count, rate = stats.get("temporal", {}).get(temporal_key, (0, 0.0))
        if count >= MIN_COUNT_TEMPORAL:
            derived.update({"fallback": "community_area + crime_type + temporal", "n_matches": count})
            return rate, derived

        area_crime_key = (crime_type, community_area)
        count, rate = stats.get("area_crime", {}).get(area_crime_key, (0, 0.0))
        if count >= MIN_COUNT_CRIME_AREA:
            derived.update({"fallback": "community_area + crime_type", "n_matches": count})
            return rate, derived

        count, rate = stats.get("area", {}).get(community_area, (0, 0.0))
        if count >= MIN_COUNT_AREA_ONLY:
            derived.update({"fallback": "community_area only", "n_matches": count})
            return rate, derived

    count, rate = stats["crime"].get(crime_type, (0, 0.0))
    if count > 0:
        derived.update({"fallback": "crime_type only", "n_matches": count})
        return rate, derived

    derived.update({"fallback": "global arrest rate", "n_matches": -1})
    return stats["global_rate"], derived
