import numpy as np
import pandas as pd
import pytest

from scripts.precompute_knn_arrays import precompute_one, FEATURE_COLS, LABEL_COL


def make_fake_df(n_per_type: int = 100, crime_types=("THEFT", "BATTERY")):
    """Build a minimal DataFrame with the columns precompute_one needs."""
    rng = np.random.default_rng(42)
    rows = []
    for ct in crime_types:
        for _ in range(n_per_type):
            rows.append({
                "primary_type": ct,
                "latitude": rng.uniform(41.64, 42.05),
                "longitude": rng.uniform(-87.95, -87.52),
                "hour_sin": rng.uniform(-1, 1),
                "hour_cos": rng.uniform(-1, 1),
                "day_of_week_sin": rng.uniform(-1, 1),
                "day_of_week_cos": rng.uniform(-1, 1),
                "month_sin": rng.uniform(-1, 1),
                "month_cos": rng.uniform(-1, 1),
                "day_of_year_sin": rng.uniform(-1, 1),
                "day_of_year_cos": rng.uniform(-1, 1),
                "arrest": int(rng.integers(0, 2)),
            })
    return pd.DataFrame(rows)


class TestPrecomputeOne:
    def test_returns_expected_keys(self):
        df = make_fake_df()
        artifact = precompute_one(df, "THEFT")
        assert set(artifact.keys()) == {
            "features_aug", "label", "features_mean", "features_std"
        }

    def test_shapes_are_correct(self):
        df = make_fake_df(n_per_type=100)
        artifact = precompute_one(df, "THEFT")
        assert artifact["features_aug"].shape == (100, 11)  # 10 features + intercept
        assert artifact["label"].shape == (100,)
        assert artifact["features_mean"].shape == (10,)
        assert artifact["features_std"].shape == (10,)

    def test_dtypes_are_correct(self):
        df = make_fake_df()
        artifact = precompute_one(df, "THEFT")
        assert artifact["features_aug"].dtype == np.float32
        assert artifact["label"].dtype == np.int8
        assert artifact["features_mean"].dtype == np.float32
        assert artifact["features_std"].dtype == np.float32

    def test_intercept_column_is_all_ones(self):
        df = make_fake_df()
        artifact = precompute_one(df, "THEFT")
        np.testing.assert_array_equal(
            artifact["features_aug"][:, 0],
            np.ones(artifact["features_aug"].shape[0], dtype=np.float32),
        )

    def test_standardization_produces_zero_mean_unit_std(self):
        df = make_fake_df(n_per_type=500)
        artifact = precompute_one(df, "THEFT")
        # Skip intercept column at index 0; check the 10 standardized features
        scaled = artifact["features_aug"][:, 1:]
        # float32 + epsilon means we won't hit exact 0/1, but should be very close
        assert scaled.mean(axis=0) == pytest.approx(np.zeros(10), abs=1e-5)
        assert scaled.std(axis=0) == pytest.approx(np.ones(10), abs=1e-3)

    def test_only_selected_crime_type_included(self):
        df = make_fake_df(n_per_type=100, crime_types=("THEFT", "BATTERY"))
        artifact = precompute_one(df, "THEFT")
        # 100 THEFT rows, BATTERY rows excluded
        assert artifact["label"].shape[0] == 100

    def test_label_values_are_zero_or_one(self):
        df = make_fake_df()
        artifact = precompute_one(df, "THEFT")
        assert set(np.unique(artifact["label"])).issubset({0, 1})

    def test_std_includes_epsilon(self):
        # Build a df where one feature has zero variance
        df = make_fake_df(n_per_type=50)
        df.loc[df["primary_type"] == "THEFT", "latitude"] = 41.8
        artifact = precompute_one(df, "THEFT")
        # std for latitude should be ~1e-8, not 0 (otherwise division would blow up)
        assert artifact["features_std"][0] > 0

    def test_query_scaling_reproducible_from_artifact(self):
        """The saved mean/std should reproduce the standardization exactly."""
        df = make_fake_df(n_per_type=200)
        artifact = precompute_one(df, "THEFT")

        # Take a raw row from the original df and scale it manually
        subset = df[df["primary_type"] == "THEFT"].reset_index(drop=True)
        raw_row = subset.iloc[0][FEATURE_COLS].to_numpy().astype(np.float32)

        mean = artifact["features_mean"].astype(np.float64)
        std = artifact["features_std"].astype(np.float64)
        scaled_manually = (raw_row - mean) / std

        # That should match the corresponding row in features_aug (skip intercept)
        scaled_in_artifact = artifact["features_aug"][0, 1:].astype(np.float64)
        np.testing.assert_allclose(scaled_manually, scaled_in_artifact, atol=1e-5)


class TestPrecomputeOneEdgeCases:
    def test_missing_crime_type_raises(self):
        df = make_fake_df()
        with pytest.raises(ValueError, match="No rows"):
            precompute_one(df, "NONEXISTENT")

    def test_single_row_subset(self):
        # One row means std=0 before epsilon; epsilon prevents division blow-up
        df = make_fake_df(n_per_type=1, crime_types=("THEFT",))
        artifact = precompute_one(df, "THEFT")
        assert artifact["features_aug"].shape == (1, 11)
        assert np.all(np.isfinite(artifact["features_aug"]))