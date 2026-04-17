import unittest

import numpy as np
from numpy.testing import assert_allclose

from src.algorithms.metrics import mae, rmse


class TestMetrics(unittest.TestCase):
    def test_mae_rmse_basic(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.5, 2.0, 2.5])
        assert_allclose(mae(y_true, y_pred), 1.0 / 3.0)
        assert_allclose(rmse(y_true, y_pred), np.sqrt((0.25 + 0 + 0.25) / 3.0))

    def test_shape_mismatch_raises(self):
        with self.assertRaises(ValueError):
            mae(np.array([1.0]), np.array([1.0, 2.0]))
        with self.assertRaises(ValueError):
            rmse(np.array([1.0]), np.array([1.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
