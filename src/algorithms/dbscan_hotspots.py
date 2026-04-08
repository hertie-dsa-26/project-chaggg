"""
DBSCAN hotspot detection on crime coordinates (lat/lon in degrees).

Uses sklearn.cluster.DBSCAN per project acceptance criteria. Training labels
clusters on historical incidents; hotspot = core/border cluster (label >= 0).
Noise points (label -1) are not hotspots.

Prediction: test point is "high crime" (class 1) if it lies inside any cluster
convex hull (optionally buffered for thin hulls).

Coordinate order: all internal XY arrays are shape (n, 2) with columns
(latitude, longitude) to match the crime dataset and sklearn's row layout.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree
from shapely.geometry import MultiPoint, Point, mapping
from shapely.ops import unary_union
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def extract_coordinates(
    df: pd.DataFrame,
    year_min: int | None = None,
    year_max: int | None = None,
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> np.ndarray:
    """Return (n, 2) float array [lat, lon] with optional year filter."""
    d = df
    if year_min is not None:
        d = d[d["year"] >= year_min]
    if year_max is not None:
        d = d[d["year"] <= year_max]
    if lat_col not in d.columns or lon_col not in d.columns:
        raise KeyError(f"Expected columns {lat_col!r}, {lon_col!r}")
    lat = d[lat_col].to_numpy(dtype=float)
    lon = d[lon_col].to_numpy(dtype=float)
    return np.column_stack([lat, lon])


def project_latlon_to_meters(
    latlon: np.ndarray,
    dst_crs: str = "EPSG:32616",
) -> np.ndarray:
    """
    Project WGS84 (lat, lon) into planar meters.

    Returns (n, 2) array (x_m, y_m).
    Default CRS is UTM Zone 16N, reasonable for Chicago.
    """
    if latlon.ndim != 2 or latlon.shape[1] != 2:
        raise ValueError("latlon must be (n, 2) with [lat, lon]")
    transformer = Transformer.from_crs("EPSG:4326", dst_crs, always_xy=True)
    lon = latlon[:, 1].astype(float, copy=False)
    lat = latlon[:, 0].astype(float, copy=False)
    x, y = transformer.transform(lon, lat)
    return np.column_stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)])


def k_distance_curve(
    xy: np.ndarray,
    k: int,
    max_points: int = 60_000,
    random_state: int = 42,
) -> np.ndarray:
    """
    Compute the k-distance curve for DBSCAN eps selection.

    Returns sorted distances to the k-th nearest neighbor (ascending).
    For large n, subsamples to max_points for speed.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (n, 2)")
    x = xy
    if len(x) > max_points:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(x), size=max_points, replace=False)
        x = x[idx]
    nn = NearestNeighbors(n_neighbors=k, algorithm="auto", n_jobs=-1)
    nn.fit(x)
    dists, _ = nn.kneighbors(x, return_distance=True)
    kth = dists[:, -1]
    return np.sort(kth)


def plot_k_distance_curve(
    distances_sorted: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    """Save a k-distance elbow plot to disk."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(distances_sorted, linewidth=1.2)
    ax.set_xlabel("points (sorted)")
    ax.set_ylabel("distance to k-th neighbor")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fit_dbscan(
    xy: np.ndarray,
    eps: float,
    min_samples: int,
    **kwargs: Any,
) -> tuple[DBSCAN, np.ndarray]:
    """
    Fit DBSCAN on lat/lon rows using Euclidean distance in degree space.

    Parameters
    ----------
    xy : (n, 2) array, columns [lat, lon]
    eps : neighborhood radius in degrees (per assignment grid)
    min_samples : DBSCAN min_samples
    """
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must be (n, 2) with [lat, lon]")
    kwargs = dict(kwargs)
    kwargs.setdefault("n_jobs", -1)
    model = DBSCAN(eps=eps, min_samples=min_samples, **kwargs)
    labels = model.fit_predict(xy)
    return model, labels


def _geometry_for_cluster_points(
    lats: np.ndarray,
    lons: np.ndarray,
    buffer_deg: float,
):
    """Convex hull for cluster points; buffer LineString/Point hulls slightly."""
    pts = np.column_stack([lons, lats])
    if len(pts) < 2:
        return None
    multip = MultiPoint([(float(p[0]), float(p[1])) for p in pts])
    hull = multip.convex_hull
    if hull.is_empty:
        return None
    if hull.geom_type in ("LineString", "Point"):
        hull = hull.buffer(buffer_deg)
    if hull.geom_type in ("Polygon", "MultiPolygon"):
        return hull
    return None


def cluster_boundaries_geodataframe(
    xy: np.ndarray,
    labels: np.ndarray,
    buffer_deg: float = 0.0008,
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Build one polygon per cluster id (>= 0). Noise (-1) omitted.

    buffer_deg expands degenerate hulls (~90 m at Chicago latitude if 0.0008).
    """
    rows = []
    for cid in sorted(set(labels.tolist())):
        if cid < 0:
            continue
        mask = labels == cid
        lats = xy[mask, 0]
        lons = xy[mask, 1]
        geom = _geometry_for_cluster_points(lats, lons, buffer_deg)
        if geom is not None:
            rows.append({"cluster_id": int(cid), "geometry": geom})
    if not rows:
        return gpd.GeoDataFrame({"cluster_id": [], "geometry": []}, crs=crs)
    return gpd.GeoDataFrame(rows, crs=crs)


def point_in_hotspot(lat: float, lon: float, boundaries: gpd.GeoDataFrame) -> bool:
    """True if (lat, lon) lies inside any hotspot polygon."""
    if boundaries is None or len(boundaries) == 0:
        return False
    union = unary_union(boundaries.geometry.values)
    if union.is_empty:
        return False
    pt = Point(float(lon), float(lat))
    return bool(union.covers(pt))


def _hotspot_union(boundaries: gpd.GeoDataFrame):
    return unary_union(boundaries.geometry.values)


def predict_hotspot_labels(
    xy: np.ndarray,
    boundaries: gpd.GeoDataFrame,
) -> np.ndarray:
    """Return int array 1 = inside hotspot, 0 = not (vectorized, fast)."""
    out = np.zeros(len(xy), dtype=np.int64)
    if boundaries is None or len(boundaries) == 0:
        return out
    union = _hotspot_union(boundaries)
    if union.is_empty:
        return out
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(xy[:, 1], xy[:, 0]),
        crs=boundaries.crs,
    )
    inside = pts.geometry.within(union) | pts.geometry.touches(union)
    return inside.fillna(False).to_numpy().astype(np.int64)


def sample_negative_points(
    crime_xy: np.ndarray,
    n_sample: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    min_sep_deg: float,
    random_state: int = 42,
    max_draws: int | None = None,
) -> np.ndarray:
    """
    Random [lat, lon] points far from given crime_xy (Euclidean deg space).

    Used to build a mixed-survey test set with label 0 (no nearby test crime).
    """
    if max_draws is None:
        max_draws = max(n_sample * 100, 10_000)
    rng = np.random.default_rng(random_state)
    if len(crime_xy) == 0:
        raise ValueError("crime_xy must be non-empty for KD-tree")
    tree = cKDTree(crime_xy)
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    out: list[list[float]] = []
    draws = 0
    while len(out) < n_sample and draws < max_draws:
        draws += 1
        lat = float(rng.uniform(lat_min, lat_max))
        lon = float(rng.uniform(lon_min, lon_max))
        dist, _ = tree.query([lat, lon], k=1)
        if dist > min_sep_deg:
            out.append([lat, lon])
    if len(out) < n_sample:
        raise RuntimeError(
            f"Only collected {len(out)} negatives (need {n_sample}). "
            "Try lowering min_sep_deg or increasing max_draws."
        )
    return np.asarray(out, dtype=float)


def sample_negative_points_sparse_grid(
    crime_xy_all_test_period: np.ndarray,
    n_sample: int,
    lat_range: tuple[float, float],
    lon_range: tuple[float, float],
    n_lat_bins: int = 30,
    n_lon_bins: int = 30,
    max_count_in_cell: int = 0,
    random_state: int = 42,
) -> np.ndarray:
    """
    Draw negative points from grid cells with at most ``max_count_in_cell``
    test-period crimes (default 0 → cells with no incident in the test window).

    Histogram uses **all** ``crime_xy_all_test_period`` points so the label 0
    matches “sparse / no reported crime in that cell,” not merely “far from a
    subsample” of incidents.
    """
    lat_min, lat_max = lat_range
    lon_min, lon_max = lon_range
    rng = np.random.default_rng(random_state)

    counts = np.zeros((n_lat_bins, n_lon_bins), dtype=np.int32)
    if len(crime_xy_all_test_period) > 0:
        ilat = np.floor(
            (crime_xy_all_test_period[:, 0] - lat_min)
            / (lat_max - lat_min + 1e-12)
            * n_lat_bins
        ).astype(np.int32)
        ilon = np.floor(
            (crime_xy_all_test_period[:, 1] - lon_min)
            / (lon_max - lon_min + 1e-12)
            * n_lon_bins
        ).astype(np.int32)
        ilat = np.clip(ilat, 0, n_lat_bins - 1)
        ilon = np.clip(ilon, 0, n_lon_bins - 1)
        np.add.at(counts, (ilat, ilon), 1)

    candidates = np.argwhere(counts <= max_count_in_cell)
    if len(candidates) == 0:
        raise RuntimeError(
            "No grid cells satisfy max_count_in_cell; "
            "use fewer bins, raise --max-crimes-per-cell, or widen the bbox."
        )

    lat_edges = np.linspace(lat_min, lat_max, n_lat_bins + 1)
    lon_edges = np.linspace(lon_min, lon_max, n_lon_bins + 1)

    out = np.empty((n_sample, 2), dtype=float)
    for k in range(n_sample):
        r = candidates[rng.integers(0, len(candidates))]
        i, j = int(r[0]), int(r[1])
        out[k, 0] = float(rng.uniform(lat_edges[i], lat_edges[i + 1]))
        out[k, 1] = float(rng.uniform(lon_edges[j], lon_edges[j + 1]))
    return out


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


@dataclass
class TuningResult:
    best_eps: float
    best_min_samples: int
    best_metrics: dict[str, float]
    all_runs: pd.DataFrame
    train_labels: np.ndarray
    boundaries: gpd.GeoDataFrame


def grid_search_dbscan(
    train_xy: np.ndarray,
    train_latlon_for_boundaries: np.ndarray | None,
    xy_test: np.ndarray,
    y_test: np.ndarray,
    eps_values: list[float] | None = None,
    min_samples_values: list[int] | None = None,
    hull_buffer_deg: float = 0.0008,
    verbose: bool = False,
) -> TuningResult:
    """
    Try (eps, min_samples) combinations; pick best F1 on provided test set.
    """
    if eps_values is None:
        eps_values = [0.01, 0.05, 0.1]
    if min_samples_values is None:
        min_samples_values = [10, 50, 100]

    rows = []
    best_f1 = -1.0
    best: tuple[float, int, dict[str, float], np.ndarray, gpd.GeoDataFrame] | None = None

    for eps in eps_values:
        for ms in min_samples_values:
            if verbose:
                print(f"  DBSCAN fit… eps={eps} min_samples={ms}", flush=True)
            _, labels = fit_dbscan(train_xy, eps=eps, min_samples=ms)
            boundary_points = train_latlon_for_boundaries
            if boundary_points is None:
                boundary_points = train_xy
            bounds = cluster_boundaries_geodataframe(
                boundary_points, labels, buffer_deg=hull_buffer_deg
            )
            if verbose:
                print(f"    predict {len(xy_test):,} test points…", flush=True)
            y_pred = predict_hotspot_labels(xy_test, bounds)
            m = classification_metrics(y_test, y_pred)
            if verbose:
                print(f"    f1={m['f1']:.4f} clusters={int(len(set(labels)) - (1 if -1 in set(labels) else 0))}", flush=True)
            rows.append(
                {
                    "eps": eps,
                    "min_samples": ms,
                    **m,
                    "n_clusters": int(len(set(labels.tolist())) - (1 if -1 in labels else 0)),
                    "noise_fraction_train": float(np.mean(labels == -1)),
                }
            )
            if m["f1"] > best_f1:
                best_f1 = m["f1"]
                best = (eps, ms, m, labels.copy(), bounds)

    if best is None:
        raise RuntimeError("Grid search failed — no parameter combo evaluated")

    all_runs = pd.DataFrame(rows)
    _, best_ms, best_m, best_labels, best_bounds = best
    best_eps = best[0]
    return TuningResult(
        best_eps=best_eps,
        best_min_samples=best_ms,
        best_metrics=best_m,
        all_runs=all_runs,
        train_labels=best_labels,
        boundaries=best_bounds,
    )


def plot_hotspots(
    xy: np.ndarray,
    labels: np.ndarray,
    boundaries: gpd.GeoDataFrame | None,
    out_path: Path | None = None,
    figsize: tuple[float, float] = (10, 10),
    point_size: float = 0.3,
    alpha: float = 0.35,
    max_plot_points: int = 55_000,
    plot_random_state: int = 42,
) -> plt.Figure:
    """Scatter incidents by cluster; draw boundary outlines."""
    if len(xy) > max_plot_points:
        rng = np.random.default_rng(plot_random_state)
        idx = rng.choice(len(xy), size=max_plot_points, replace=False)
        xy = xy[idx]
        labels = labels[idx]
    fig, ax = plt.subplots(figsize=figsize)
    uniq = sorted(set(labels.tolist()))
    cmap = plt.colormaps["tab20"]
    for j, cid in enumerate(uniq):
        mask = labels == cid
        color = "#555555" if cid == -1 else cmap((j % 20) / 20.0)
        label = "noise" if cid == -1 else f"cluster {cid}"
        ax.scatter(
            xy[mask, 1],
            xy[mask, 0],
            s=point_size,
            c=[color],
            alpha=alpha,
            label=label if cid == -1 or j < 12 else None,
        )
    if boundaries is not None and len(boundaries):
        boundaries.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("DBSCAN crime hotspots (train)")
    ax.set_aspect("equal", adjustable="box")
    plt.tight_layout()
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return fig


def save_cluster_boundaries(
    boundaries: gpd.GeoDataFrame,
    path: Path,
) -> None:
    """Write GeoJSON for downstream map / Flask use."""
    path.parent.mkdir(parents=True, exist_ok=True)
    boundaries.to_file(path, driver="GeoJSON")


def boundaries_to_json_serializable(boundaries: gpd.GeoDataFrame) -> list[dict]:
    """GeoJSON-like dict list (for lightweight API responses)."""
    out = []
    for _, row in boundaries.iterrows():
        out.append(
            {
                "cluster_id": int(row["cluster_id"]),
                "geometry": mapping(row.geometry),
            }
        )
    return out


def save_metrics_json(metrics: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
