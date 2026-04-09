import unittest

import numpy as np
import pandas as pd

from src.algorithms.kdtree import build_kdtree, knn_bruteforce, knn_query
from src.algorithms.knn_scratch import SpatiotemporalKNN


class TestKDTree(unittest.TestCase):
    def test_knn_query_matches_bruteforce_indices(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        q = rng.normal(size=(3,))
        k = 7

        root = build_kdtree(X)
        kd = knn_query(root, q, k=k)
        kd_idx = [i for _, i in kd]
        bf_idx = [i for _, i in knn_bruteforce(X, q, k=k)]

        self.assertEqual(set(kd_idx), set(bf_idx))

    def test_knn_query_when_k_exceeds_point_count(self):
        X = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]], dtype=float)
        q = np.array([0.5, 0.5])
        root = build_kdtree(X)
        kd = knn_query(root, q, k=100)
        bf = knn_bruteforce(X, q, k=100)
        self.assertEqual(len(kd), 3)
        self.assertEqual({i for _, i in kd}, {i for _, i in bf})


class TestSpatiotemporalKNN(unittest.TestCase):
    def test_predict_exact_match_returns_exact_value(self):
        df = pd.DataFrame(
            {
                "lat": [41.0, 41.1, 41.2],
                "lon": [-87.0, -87.1, -87.2],
                "month_index": [0, 1, 2],
                "crime_count": [10, 12, 14],
            }
        )
        model = SpatiotemporalKNN(k=3, space="degrees", time_scale=0.1)
        model.train(df)
        pred = model.predict((41.1, -87.1), 1)
        self.assertAlmostEqual(pred, 12.0, places=8)

    def test_bruteforce_predict_matches_kdtree_path(self):
        rng = np.random.default_rng(2)
        n = 40
        df = pd.DataFrame(
            {
                "lat": rng.uniform(41.6, 42.0, size=n),
                "lon": rng.uniform(-87.8, -87.5, size=n),
                "month_index": rng.integers(0, 24, size=n, endpoint=False).astype(float),
                "crime_count": rng.uniform(5, 80, size=n),
            }
        )
        k = 5
        pts = list(zip(df["lat"], df["lon"], df["month_index"]))
        m_kd = SpatiotemporalKNN(k=k, space="degrees", time_scale=0.05, use_bruteforce=False)
        m_kd.train(df)
        m_bf = SpatiotemporalKNN(k=k, space="degrees", time_scale=0.05, use_bruteforce=True)
        m_bf.train(df)
        for lat, lon, t in pts[:15]:
            a = m_kd.predict((lat, lon), t)
            b = m_bf.predict((lat, lon), t)
            self.assertAlmostEqual(a, b, places=5, msg=f"lat={lat} lon={lon} t={t}")


if __name__ == "__main__":
    unittest.main()

