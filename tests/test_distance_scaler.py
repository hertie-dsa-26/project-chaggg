import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.algorithms.distance import SpaceTimeScaler


def _euclid(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


class TestSpaceTimeScalerDistances(unittest.TestCase):
    def test_zero_distance_identical_point_degrees(self):
        s = SpaceTimeScaler(space="degrees", time_scale=0.1)
        x = s.transform_one(41.88, -87.63, 12.0)
        assert_allclose(_euclid(x, x), 0.0, atol=1e-12)

    def test_symmetry_two_points_meters(self):
        s = SpaceTimeScaler(space="meters", time_scale=1000.0)
        x1 = s.transform_one(41.88, -87.63, 0.0)
        x2 = s.transform_one(41.90, -87.60, 3.0)
        d12 = _euclid(x1, x2)
        d21 = _euclid(x2, x1)
        self.assertAlmostEqual(d12, d21, places=9)

    def test_same_spatial_different_time_nonzero_when_scaled(self):
        s = SpaceTimeScaler(space="meters", time_scale=500.0)
        x1 = s.transform_one(41.88, -87.63, 0.0)
        x2 = s.transform_one(41.88, -87.63, 2.0)
        self.assertGreater(_euclid(x1, x2), 1e-6)


if __name__ == "__main__":
    unittest.main()
