from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .kdtree import KDNode, build_kdtree, knn_query


@dataclass
class SpatiotemporalKNN:
    """
    From-scratch Spatiotemporal KNN forecaster.

    - train(data): builds feature matrix + KD-tree
    - predict(location, time): returns inverse-distance weighted average of neighbor counts
    """

    k: int = 10
    time_scale: float = 0.0  # placeholder (meters-per-month or degrees-per-month)
    eps: float = 1e-9

    _root: KDNode | None = None
    _X: np.ndarray | None = None
    _y: np.ndarray | None = None

    def train(self, data: pd.DataFrame) -> None:
        """
        Expect columns: lat, lon, month_index, crime_count.
        """
        required = ["lat", "lon", "month_index", "crime_count"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            raise KeyError(f"Missing required columns: {missing}")

        lat = data["lat"].to_numpy(dtype=float)
        lon = data["lon"].to_numpy(dtype=float)
        t = data["month_index"].to_numpy(dtype=float)
        y = data["crime_count"].to_numpy(dtype=float)

        X = np.column_stack([lat, lon, self.time_scale * t])
        self._X = X
        self._y = y
        self._root = build_kdtree(X)

    def predict(self, location: tuple[float, float], time: float) -> float:
        """
        location: (lat, lon)
        time: month_index (float or int)
        """
        if self._root is None or self._X is None or self._y is None:
            raise RuntimeError("Model not trained. Call train(data) first.")

        lat, lon = float(location[0]), float(location[1])
        q = np.array([lat, lon, self.time_scale * float(time)], dtype=float)
        neighbors = knn_query(self._root, q, k=self.k)
        if not neighbors:
            return float("nan")

        d2 = np.array([dist2 for dist2, _ in neighbors], dtype=float)
        idx = np.array([i for _, i in neighbors], dtype=int)
        yv = self._y[idx]

        # exact match shortcut
        if np.any(d2 == 0.0):
            return float(yv[np.argmin(d2)])

        d = np.sqrt(d2)
        w = 1.0 / (d + self.eps)
        return float(np.sum(w * yv) / np.sum(w))

