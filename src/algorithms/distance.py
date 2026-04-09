from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from pyproj import Transformer


@dataclass(frozen=True)
class SpaceTimeScaler:
    """
    Convert (lat, lon, month_index) into a numeric feature vector for KNN.

    If space="degrees": uses (lat, lon) directly.
    If space="meters": projects to UTM16N (x_m, y_m) for meaningful eps/distances.

    time_scale multiplies month_index so time participates in Euclidean distance.
    """

    space: str = "degrees"  # "degrees" | "meters"
    time_scale: float = 0.0
    dst_crs: str = "EPSG:32616"  # UTM16N (Chicago)

    def _project_latlon(self, lat: np.ndarray, lon: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        transformer = Transformer.from_crs("EPSG:4326", self.dst_crs, always_xy=True)
        x, y = transformer.transform(lon.astype(float, copy=False), lat.astype(float, copy=False))
        return np.asarray(x, dtype=float), np.asarray(y, dtype=float)

    def transform(self, lat: np.ndarray, lon: np.ndarray, month_index: np.ndarray) -> np.ndarray:
        lat = np.asarray(lat, dtype=float)
        lon = np.asarray(lon, dtype=float)
        t = np.asarray(month_index, dtype=float)
        if lat.shape != lon.shape or lat.shape != t.shape:
            raise ValueError("lat, lon, month_index must have same shape")

        if self.space == "meters":
            x, y = self._project_latlon(lat, lon)
            return np.column_stack([x, y, self.time_scale * t])
        if self.space == "degrees":
            return np.column_stack([lat, lon, self.time_scale * t])
        raise ValueError("space must be 'degrees' or 'meters'")

    def transform_one(self, lat: float, lon: float, month_index: float) -> np.ndarray:
        return self.transform(
            np.array([lat], dtype=float),
            np.array([lon], dtype=float),
            np.array([month_index], dtype=float),
        )[0]
