"""Unit tests for scripts.knn_location_only."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.knn_location_only import (
    _parse_int_csv,
    _valid_k_candidates,
    evaluate_all_crime_types,
    evaluate_crime_type,
    main,
    slugify,
    standardize_locations,
)
from scripts.utils import temporal_split


def _make_two_cluster_df(
    n_per_cluster: int = 200, seed: int = 0
) -> pd.DataFrame:
    """Two well-separated lat/lon clusters with distinct arrest rates.

    Cluster 1 (around 41.85, -87.65) has high arrest probability (0.9).
    Cluster 2 (around 41.95, -87.75) has low  arrest probability (0.1).
    A KNN classifier on location alone should easily separate them.
    """
    rng = np.random.default_rng(seed)
    lat1 = 41.85 + rng.normal(0, 0.005, n_per_cluster)
    lon1 = -87.65 + rng.normal(0, 0.005, n_per_cluster)
    arr1 = (rng.random(n_per_cluster) < 0.9).astype(int)
    lat2 = 41.95 + rng.normal(0, 0.005, n_per_cluster)
    lon2 = -87.75 + rng.normal(0, 0.005, n_per_cluster)
    arr2 = (rng.random(n_per_cluster) < 0.1).astype(int)
    n = 2 * n_per_cluster
    years = rng.choice([2015, 2018, 2021, 2023, 2024], size=n)
    return pd.DataFrame(
        {
            "primary_type": ["THEFT"] * n,
            "latitude": np.concatenate([lat1, lat2]),
            "longitude": np.concatenate([lon1, lon2]),
            "arrest": np.concatenate([arr1, arr2]),
            "date": pd.to_datetime(
                [f"{int(y)}-06-15" for y in years], errors="coerce"
            ),
        }
    )


class TestSlugify(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(slugify("THEFT"), "theft")

    def test_punctuation_collapsed(self):
        self.assertEqual(
            slugify("CRIMINAL DAMAGE / VANDALISM"), "criminal_damage_vandalism"
        )

    def test_empty_falls_back(self):
        self.assertEqual(slugify("---"), "unnamed")
        self.assertEqual(slugify(""), "unnamed")


class TestParseHelpers(unittest.TestCase):
    def test_parse_int_csv(self):
        self.assertEqual(_parse_int_csv("10, 25 ,50,100, 250"), [10, 25, 50, 100, 250])
        self.assertEqual(_parse_int_csv(""), [])


class TestSharedTemporalSplit(unittest.TestCase):
    """Sanity tests proving we use @ght-1's temporal_split unchanged."""

    def test_temporal_split_uses_train_end_inclusive(self):
        df = pd.DataFrame(
            {
                "date": pd.to_datetime(
                    [
                        "2022-12-31",
                        "2023-01-01",
                        "2024-06-15",
                        "2015-06-15",
                    ]
                ),
                "value": [1, 2, 3, 4],
            }
        )
        train, test = temporal_split(df, date_col="date", train_end="2022-12-31")
        self.assertEqual(sorted(train["value"].tolist()), [1, 4])
        self.assertEqual(sorted(test["value"].tolist()), [2, 3])


class TestStandardize(unittest.TestCase):
    def test_train_only_fit_matches_ght_helpers(self):
        train = pd.DataFrame({"latitude": [0.0, 2.0, 4.0], "longitude": [10.0, 12.0, 14.0]})
        test = pd.DataFrame({"latitude": [2.0], "longitude": [12.0]})
        X_train, X_test, (mean, std) = standardize_locations(train, test)
        np.testing.assert_allclose(mean, [2.0, 12.0], atol=1e-9)
        # @ght-1's fit_scaler adds 1e-8 to std (vs StandardScaler) for stability
        self.assertAlmostEqual(float(std[0]), train["latitude"].std(ddof=0) + 1e-8, places=9)
        # Test point at training mean must map to ~0 after transform
        self.assertAlmostEqual(float(X_test[0, 0]), 0.0, places=6)
        self.assertAlmostEqual(float(X_test[0, 1]), 0.0, places=6)
        # Train should be (approximately) zero-mean after transform
        self.assertAlmostEqual(float(np.mean(X_train[:, 0])), 0.0, places=9)


class TestValidKCandidates(unittest.TestCase):
    def test_caps_to_fold_training_size(self):
        # 5-fold on n=100 -> training fold size 80
        valid = _valid_k_candidates([10, 25, 50, 100, 250], n_train=100, cv_folds=5)
        self.assertEqual(valid, [10, 25, 50])

    def test_falls_back_when_all_too_large(self):
        valid = _valid_k_candidates([100, 250], n_train=10, cv_folds=5)
        self.assertEqual(valid, [8])  # min(100, fold_size=8)


class TestEvaluateCrimeType(unittest.TestCase):
    def test_well_separated_clusters_yield_low_log_loss(self):
        df = _make_two_cluster_df(n_per_cluster=300, seed=0)
        train, test = temporal_split(df, date_col="date", train_end="2022-12-31")
        result = evaluate_crime_type(
            crime_type="THEFT",
            train=train,
            test=test,
            k_candidates=[5, 10, 25],
            cv_folds=3,
            random_state=0,
        )
        self.assertIsNone(result.skipped)
        self.assertIn(result.best_k, [5, 10, 25])
        self.assertIsNotNone(result.test_log_loss)
        self.assertIsNotNone(result.test_accuracy)
        self.assertIsNotNone(result.test_roc_auc)
        # With well-separated, calibrated clusters, KNN must do better than
        # the pure positive-rate baseline of ~0.5 -> log(2) ≈ 0.693
        self.assertLess(result.test_log_loss, 0.55)
        self.assertGreater(result.test_accuracy, 0.7)
        self.assertGreater(result.test_roc_auc, 0.85)
        self.assertEqual(len(result.cv_neg_log_loss), 3)
        self.assertIsNotNone(result.predictions)
        self.assertEqual(
            sorted(result.predictions.columns.tolist()),
            sorted(["latitude", "longitude", "y_true", "y_pred", "p_arrest"]),
        )

    def test_skipped_on_single_class_train(self):
        df = pd.DataFrame(
            {
                "primary_type": ["THEFT"] * 6,
                "latitude": [41.0, 41.1, 41.2, 41.3, 41.4, 41.5],
                "longitude": [-87.0, -87.1, -87.2, -87.3, -87.4, -87.5],
                "arrest": [0, 0, 0, 0, 1, 1],
                "date": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2016-01-01",
                        "2017-01-01",
                        "2018-01-01",
                        "2023-01-01",
                        "2024-01-01",
                    ]
                ),
            }
        )
        train, test = temporal_split(df, date_col="date", train_end="2022-12-31")
        result = evaluate_crime_type("THEFT", train=train, test=test, cv_folds=2)
        self.assertEqual(result.skipped, "single-class training set")
        self.assertIsNone(result.test_log_loss)

    def test_skipped_on_empty_test(self):
        df = pd.DataFrame(
            {
                "primary_type": ["THEFT"] * 4,
                "latitude": [41.0, 41.1, 41.2, 41.3],
                "longitude": [-87.0, -87.1, -87.2, -87.3],
                "arrest": [0, 1, 0, 1],
                "date": pd.to_datetime(
                    [
                        "2015-01-01",
                        "2016-01-01",
                        "2017-01-01",
                        "2018-01-01",
                    ]
                ),
            }
        )
        train, test = temporal_split(df, date_col="date", train_end="2022-12-31")
        result = evaluate_crime_type("THEFT", train=train, test=test, cv_folds=2)
        self.assertEqual(result.skipped, "empty train or test split")


class TestEvaluateAllCrimeTypes(unittest.TestCase):
    def test_runs_for_multiple_types(self):
        a = _make_two_cluster_df(n_per_cluster=120, seed=1)
        a["primary_type"] = "THEFT"
        b = _make_two_cluster_df(n_per_cluster=120, seed=2)
        b["primary_type"] = "ASSAULT"
        df = pd.concat([a, b], ignore_index=True)

        results = evaluate_all_crime_types(
            df,
            k_candidates=[5, 10],
            cv_folds=3,
            random_state=0,
            progress=False,
        )
        self.assertEqual(sorted(r.crime_type for r in results), ["ASSAULT", "THEFT"])
        for r in results:
            self.assertIsNone(r.skipped)
            self.assertIsNotNone(r.test_log_loss)

    def test_missing_columns_raises(self):
        df = pd.DataFrame({"latitude": [1.0], "longitude": [2.0]})
        with self.assertRaises(KeyError):
            evaluate_all_crime_types(df, progress=False)


class TestMainIntegration(unittest.TestCase):
    def test_main_writes_outputs(self):
        import tempfile
        from unittest import mock

        df = _make_two_cluster_df(n_per_cluster=150, seed=3)
        df["primary_type"] = "THEFT"

        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "out"
            with mock.patch(
                "src.preprocess_data.preprocess_data", return_value=df, create=True
            ):
                exit_code = main(
                    [
                        "--k-candidates",
                        "5,10",
                        "--cv-folds",
                        "3",
                        "--output-dir",
                        str(out_dir),
                        "--random-state",
                        "0",
                    ]
                )

            self.assertEqual(exit_code, 0)
            self.assertTrue((out_dir / "metrics.json").exists())
            self.assertTrue((out_dir / "summary.csv").exists())
            self.assertTrue((out_dir / "predictions_theft.csv").exists())

            payload = json.loads((out_dir / "metrics.json").read_text())
            self.assertIn("run_config", payload)
            self.assertEqual(payload["run_config"]["k_candidates"], [5, 10])
            self.assertEqual(payload["run_config"]["train_end"], "2022-12-31")
            self.assertEqual(
                payload["run_config"]["shared_data_prep"]["split"],
                "scripts.utils.temporal_split",
            )
            self.assertEqual(len(payload["per_crime_type"]), 1)
            self.assertEqual(payload["per_crime_type"][0]["crime_type"], "THEFT")
            self.assertIsNone(payload["per_crime_type"][0].get("skipped"))


if __name__ == "__main__":
    unittest.main()
