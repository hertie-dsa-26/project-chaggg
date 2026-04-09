from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .distance import SpaceTimeScaler
from .kdtree import KDNode, build_kdtree, knn_bruteforce, knn_query


@dataclass
class SpatiotemporalKNN:
    """
    From-scratch Spatiotemporal KNN forecaster.

    - train(data): builds feature matrix + KD-tree
    - predict(location, time): returns inverse-distance weighted average of neighbor counts
    """

    k: int = 10
    space: str = "degrees"  # "degrees" | "meters"
    time_scale: float = 0.0  # meters-per-month (if meters) or degrees-per-month (if degrees)
    eps: float = 1e-9
    # Debug / validation: O(n) neighbor search (same weighting as KD path)
    use_bruteforce: bool = False

    _root: KDNode | None = None
    _X: np.ndarray | None = None
    _y: np.ndarray | None = None
    _scaler: SpaceTimeScaler | None = None

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

        self._scaler = SpaceTimeScaler(space=self.space, time_scale=self.time_scale)
        X = self._scaler.transform(lat=lat, lon=lon, month_index=t)
        self._X = X
        self._y = y
        self._root = None if self.use_bruteforce else build_kdtree(X)

    def predict(self, location: tuple[float, float], time: float) -> float:
        """
        location: (lat, lon)
        time: month_index (float or int)
        """
        if self._X is None or self._y is None:
            raise RuntimeError("Model not trained. Call train(data) first.")
        if not self.use_bruteforce and self._root is None:
            raise RuntimeError("Model not trained. Call train(data) first.")
        if self._scaler is None:
            raise RuntimeError("Internal error: scaler missing. Call train(data) first.")

        lat, lon = float(location[0]), float(location[1])
        q = self._scaler.transform_one(lat=lat, lon=lon, month_index=float(time))
        if self.use_bruteforce:
            neighbors = knn_bruteforce(self._X, q, k=self.k)
        else:
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

