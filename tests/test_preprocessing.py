import unittest

import pandas as pd

from data.clean import preprocess


class TestPreprocess(unittest.TestCase):
    def test_drops_missing_lat_lon(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01T12:00:00.000", "2025-01-02T12:00:00.000"],
                "latitude": [41.9, None],
                "longitude": [-87.7, -87.7],
                "primary_type": ["THEFT", "THEFT"],
            }
        )
        out = preprocess(df)
        self.assertEqual(len(out), 1)

    def test_adds_temporal_features(self):
        df = pd.DataFrame(
            {
                "date": ["2025-01-01T12:34:56.000"],
                "latitude": [41.9],
                "longitude": [-87.7],
            }
        )
        out = preprocess(df)
        self.assertIn("Year", out.columns)
        self.assertIn("Month", out.columns)
        self.assertIn("Day", out.columns)
        self.assertIn("Hour", out.columns)
        self.assertIn("Day of Week", out.columns)
        self.assertEqual(int(out.iloc[0]["Year"]), 2025)
        self.assertEqual(int(out.iloc[0]["Month"]), 1)
        self.assertEqual(int(out.iloc[0]["Day"]), 1)
        self.assertEqual(int(out.iloc[0]["Hour"]), 12)

    def test_filters_out_2026_plus(self):
        df = pd.DataFrame(
            {
                "date": ["2026-01-01T00:00:00.000"],
                "latitude": [41.9],
                "longitude": [-87.7],
            }
        )
        out = preprocess(df)
        self.assertEqual(len(out), 0)


if __name__ == "__main__":
    unittest.main()

