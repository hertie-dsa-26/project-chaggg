# US-2.6 — DBSCAN Spatial Clustering (Hotspot Detection) Report

## Goal

Implement **DBSCAN hotspot detection** to identify spatial crime clusters using **latitude/longitude** and use the learned hotspots to produce **binary predictions** (“high crime” vs “not high crime”) for a held-out test set. Per acceptance criteria, **`sklearn.cluster.DBSCAN`** is allowed and used.

---

## What was implemented (code locations)

- **Core algorithm / utilities:** `src/algorithms/dbscan_hotspots.py`
  - `extract_coordinates(df, ...)`: pull `(lat, lon)` into an \(n \times 2\) array
  - `project_latlon_to_meters(latlon, dst_crs="EPSG:32616")`: project to planar meters for meaningful eps
  - `k_distance_curve(xy, k, ...)` + `plot_k_distance_curve(...)`: generate eps “elbow” evidence
  - `fit_dbscan(xy, eps, min_samples, ...)`: runs **sklearn DBSCAN** (defaults to `n_jobs=-1`)
  - `cluster_boundaries_geodataframe(xy, labels, ...)`: builds **cluster boundary polygons** (convex hull per cluster, buffered for degenerate cases)
  - `predict_hotspot_labels(xy_test, boundaries)`: predicts 1/0 using **point-in-(union of) cluster boundary** (vectorized)
  - `grid_search_dbscan(...)`: grid search over \(\epsilon\) and `min_samples`, selects best **F1**
  - Metrics: accuracy, precision, recall, F1
  - Negative sampling options:
    - `sample_negative_points_sparse_grid(...)` (**default**) — draws negatives from grid cells with ≤ `max_crimes_per_cell` crimes in test period (default 0)
    - `sample_negative_points(...)` — draws negatives far from test-period crimes by `min_sep_deg`

- **End-to-end runner:** `scripts/run_dbscan_hotspots.py`
  - Loads only required columns: `["latitude", "longitude", "year", "date"]`
  - Train window: **2015–2022**
  - Test positives: crimes from **2023–2024**
  - Test negatives: **default `sparse_grid`** (see above), configurable via CLI
  - Saves all artifacts under `outputs/dbscan/` (or `--output-dir`)

---

## How to run

From repo root (after cleaned data exists in `data/cleaned/`):

```bash
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplot uv run python scripts/run_dbscan_hotspots.py
```

Useful flags:

- **Speed / scale**
  - `--max-train 45000` (default) or increase for heavier run (e.g. `120000`)
  - `--test-crimes-max 4000` (default)
  - `--n-neg 4000` (default)
- **Clustering space**
  - `--space degrees` (default; eps in degrees)
  - `--space meters` (project to UTM16N; eps in meters)
- **Negative sampling strategy**
  - `--negative-strategy sparse_grid` (default)
  - `--negative-strategy distance`
  - `--grid-lat-bins 32 --grid-lon-bins 32`
  - `--max-crimes-per-cell 0`
  - `--min-sep-deg 0.015` (only for `distance`)
- **Verbosity**
  - `--quiet` (disables per-combo progress prints)
- **Eps justification helper**
  - `--k-distance-plot` (saves `k_distance_plot.png`)
  - `--k-distance-k 50` (k for curve; often equals `min_samples`)

---

## Parameter tuning (acceptance criteria)

The grid search tries (default, **degree space**):

- \(\epsilon \in \{0.01, 0.05, 0.1\}\) **degrees**
- `min_samples ∈ {10, 50, 100}`

Selection criterion: **best F1** on the binary test set.

### CV-quality improvement: meter-space DBSCAN (optional)

Because lat/lon degrees are not a constant physical distance, the runner supports
clustering in **projected meters** (Chicago: **UTM16N `EPSG:32616`**) and still
exports boundaries in WGS84 for web mapping.

- Enable via: `--space meters`
- Meter eps grid used in this mode: `eps ∈ {500, 1500, 3000}` (meters)

### CV-quality improvement: k-distance (elbow) plot for eps selection (optional)

To justify an eps choice more rigorously, the runner can generate a
**k-distance curve**: for each point, compute the distance to its **k-th nearest
neighbor**, then sort these distances. The “elbow” region is a common heuristic
for selecting DBSCAN’s `eps`.

- Enable via: `--k-distance-plot`
- Choose k via: `--k-distance-k` (often set equal to `min_samples`)
- Output: `k_distance_plot.png` saved next to other artifacts in the output dir

---

## Prediction logic (acceptance criteria)

1. Fit DBSCAN on training incidents → labels: **-1 = noise**, **0+ = cluster IDs**
2. For each cluster, construct a boundary polygon (convex hull; buffered if needed)
3. For a test point:
   - If the point lies **inside/touches** the union of cluster polygons → predict **1 (high crime)**
   - Else → predict **0**

---

## Outputs

Default output folder: `outputs/dbscan/`

- `cluster_boundaries.geojson` — cluster polygons for mapping / Flask use
- `hotspots_train.png` — visualization of clustered points + boundary outlines
- `binary_predictions.csv` — per test point: lat, lon, `y_true`, `y_pred`
- `metrics.json` — best params, metrics, full grid results, and `run_config` (for reproducibility)

## Flask integration (interactive visualization)

To support the project requirement for **interactive visualizations**, hotspot
boundaries can be explored in the Flask app:

- **Page:** `/viz/hotspots`
  - Leaflet map + GeoJSON overlay
  - Dropdown allows switching between **meters** vs **degrees** variants
- **API:** `/api/hotspots?variant=meters|degrees`
  - Serves the corresponding GeoJSON

Static GeoJSON variants committed for the demo:

- `src/flask_app/static/geo/hotspots_meters.geojson`
- `src/flask_app/static/geo/hotspots_degrees.geojson`

Run the app from repo root:

```bash
MPLCONFIGDIR=/tmp/matplot uv run python run_app.py
```

### Keeping the Flask map in sync (export helper)

To update what the Flask map displays after re-running DBSCAN, you can export the
latest boundary GeoJSON into a generated static location:

```bash
PYTHONUNBUFFERED=1 MPLCONFIGDIR=/tmp/matplot uv run python scripts/run_dbscan_hotspots.py \
  --space meters \
  --export-flask-static
```

This writes to:

- `src/flask_app/static/geo/generated/hotspots_meters.geojson`

The Flask API `/api/hotspots` prefers the **generated** files when present, and
falls back to the committed demo GeoJSON otherwise. The generated folder is
ignored by git.

---

## Notes / interpretation

Binary metrics depend heavily on how **negative points** are defined.

- **Default (`sparse_grid`)** negatives come from grid cells with **0** test-period crimes (configurable). This is usually more interpretable than sampling “just far from a subsample.”
- If you want the older behavior, run with `--negative-strategy distance`.

### Comparison snapshot (degrees vs meters)

Same evaluation setup (train cap 20,000; 1,500 positives + 1,500 negatives; `sparse_grid`; k-distance plot enabled):

| Space | Best eps | Best min_samples | Accuracy | Precision | Recall | F1 |
|------:|---------:|-----------------:|---------:|----------:|-------:|---:|
| degrees | 0.01 | 50 | 0.9760 | 0.9824 | 0.9693 | 0.9758 |
| meters (UTM16N) | 500 | 10 | 0.9787 | 0.9742 | 0.9833 | 0.9788 |

Artifacts were written to:
- `outputs/dbscan_compare_deg/` (includes `k_distance_plot.png`)
- `outputs/dbscan_compare_meters/` (includes `k_distance_plot.png`)

### Robustness check: harder negatives (sparse_grid)

We also varied `max_crimes_per_cell` to draw negatives from grid cells that are
**not strictly empty** in the test period. This stress-tests whether hotspot
polygons stay meaningful when negatives come from “almost empty” areas.

Same setup as above (meters; train cap 20,000; 1,500 positives + 1,500 negatives; k-distance plot enabled):

| max_crimes_per_cell | Best eps (m) | Best min_samples | Accuracy | Precision | Recall | F1 | Artifacts |
|--------------------:|-------------:|-----------------:|---------:|----------:|-------:|---:|----------|
| 1 | 500 | 10 | 0.9780 | 0.9730 | 0.9833 | 0.9781 | `outputs/dbscan_hardneg_cell1/` |
| 2 | 500 | 10 | 0.9783 | 0.9736 | 0.9833 | 0.9784 | `outputs/dbscan_hardneg_cell2/` |

---

## Acceptance criteria checklist

- **DBSCAN applied to lat/lon**: yes (train coordinates from cleaned data)
- **Parameters tuned**: yes (grid search over eps/min_samples)
- **Clusters labeled (-1 noise, 0+ cluster)**: yes (sklearn semantics)
- **Prediction logic**: yes (point-in-hotspot polygon)
- **Visualization with boundaries**: yes (`hotspots_train.png`)
- **Binary metrics**: yes (accuracy, precision, recall, F1)
- **sklearn allowed/used**: yes (`sklearn.cluster.DBSCAN`)

