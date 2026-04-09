import unittest

import pandas as pd

from src.algorithms.data_prep import MonthlyCommunityAreaTable, split_train_test_by_year


class TestTrainTestSplit(unittest.TestCase):
    def test_no_year_overlap(self):
        df = pd.DataFrame(
            {
                "community_area": [1.0, 1.0, 1.0, 1.0],
                "month": pd.to_datetime(["2015-01-01", "2022-12-01", "2023-01-01", "2024-12-01"]),
                "crime_count": [10, 20, 30, 40],
                "lat": [41.88, 41.88, 41.88, 41.88],
                "lon": [-87.63, -87.63, -87.63, -87.63],
                "month_index": [0.0, 95.0, 96.0, 107.0],
            }
        )
        train, test = split_train_test_by_year(monthly=MonthlyCommunityAreaTable(df=df))
        self.assertEqual(len(train), 2)
        self.assertEqual(len(test), 2)
        train_years = pd.to_datetime(train["month"]).dt.year
        test_years = pd.to_datetime(test["month"]).dt.year
        self.assertTrue(train_years.max() <= 2022)
        self.assertTrue(test_years.min() >= 2023)
        self.assertLess(train["month"].max(), test["month"].min())


if __name__ == "__main__":
    unittest.main()
