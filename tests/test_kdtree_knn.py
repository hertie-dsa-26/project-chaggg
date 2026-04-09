import unittest

import numpy as np
import pandas as pd

from src.algorithms.kdtree import build_kdtree, knn_query
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

        d2 = np.sum((X - q) ** 2, axis=1)
        bf_idx = np.argsort(d2)[:k].tolist()

        self.assertEqual(set(kd_idx), set(bf_idx))


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


if __name__ == "__main__":
    unittest.main()

