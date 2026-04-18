"""Unit tests for scripts.estimators.compute_arrest_probability."""

import math
import unittest

import pandas as pd

from scripts.estimators import compute_arrest_probability


def _make_df():
    """Small synthetic DataFrame with known arrest probabilities."""
    return pd.DataFrame(
        {
            "primary_type": [
                "THEFT", "THEFT", "THEFT", "THEFT",
                "ASSAULT", "ASSAULT",
                "ROBBERY", "ROBBERY",
                "NARCOTICS",
            ],
            "block": [
                "100 W MAIN ST", "100 W MAIN ST", "200 E OAK AVE", "200 E OAK AVE",
                "100 W MAIN ST", "200 E OAK AVE",
                "100 W MAIN ST", "100 W MAIN ST",
                "300 N ELM ST",
            ],
            "district": [1, 1, 2, 2, 1, 2, 1, 1, 3],
            "arrest": [True, False, True, True, False, False, True, True, False],
        }
    )


class TestCorrectProbability(unittest.TestCase):
    def test_district_theft(self):
        df = _make_df()
        result = compute_arrest_probability(df, "THEFT", "district")
        # district 1: 1 arrest / 2 total = 0.5
        self.assertAlmostEqual(result[1], 0.5)
        # district 2: 2 arrests / 2 total = 1.0
        self.assertAlmostEqual(result[2], 1.0)

    def test_block_theft(self):
        df = _make_df()
        result = compute_arrest_probability(df, "THEFT", "block")
        self.assertAlmostEqual(result["100 W MAIN ST"], 0.5)
        self.assertAlmostEqual(result["200 E OAK AVE"], 1.0)


class TestNaNForEmptyGroups(unittest.TestCase):
    def test_unknown_crime_type_returns_empty_series(self):
        df = _make_df()
        result = compute_arrest_probability(df, "ARSON", "district")
        self.assertTrue(result.empty)

    def test_area_absent_for_crime_type_is_not_in_result(self):
        # NARCOTICS only appears in district 3; districts 1 and 2 should be absent
        df = _make_df()
        result = compute_arrest_probability(df, "NARCOTICS", "district")
        self.assertNotIn(1, result.index)
        self.assertNotIn(2, result.index)
        self.assertIn(3, result.index)


class TestSupportedAreaCols(unittest.TestCase):
    def test_block_col_works(self):
        df = _make_df()
        result = compute_arrest_probability(df, "THEFT", "block")
        self.assertIn("100 W MAIN ST", result.index)

    def test_district_col_works(self):
        df = _make_df()
        result = compute_arrest_probability(df, "THEFT", "district")
        self.assertIn(1, result.index)

    def test_invalid_area_col_raises(self):
        df = _make_df()
        with self.assertRaises(ValueError):
            compute_arrest_probability(df, "THEFT", "ward")


class TestHundredPercentArrestRate(unittest.TestCase):
    def test_robbery_district_1_is_100_percent(self):
        # ROBBERY in district 1: 2 arrests / 2 total = 1.0
        df = _make_df()
        result = compute_arrest_probability(df, "ROBBERY", "district")
        self.assertAlmostEqual(result[1], 1.0)


class TestZeroPercentArrestRate(unittest.TestCase):
    def test_assault_both_districts_zero(self):
        # ASSAULT: 0 arrests in district 1 and 0 in district 2
        df = _make_df()
        result = compute_arrest_probability(df, "ASSAULT", "district")
        self.assertAlmostEqual(result[1], 0.0)
        self.assertAlmostEqual(result[2], 0.0)

    def test_narcotics_district_3_zero(self):
        df = _make_df()
        result = compute_arrest_probability(df, "NARCOTICS", "district")
        self.assertAlmostEqual(result[3], 0.0)


if __name__ == "__main__":
    unittest.main()
