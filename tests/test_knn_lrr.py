"""Tests for algorithms/knn_lrr.py — the from-scratch KNN-LRR predictor."""
import numpy as np
import pytest

from algorithms.knn_lrr import (
    FEATURE_COLUMNS,
    N_FEATURES,
    standardize_and_augment_query,
    fit_logistic_ridge,
    predict_arrest_probability,
)


@pytest.fixture
def tiny_artifact():
    """Synthetic artifact with 50 rows, mimicking the .npz structure.

    Built so that points near (lat=41.88, lon=-87.63) at hour=14 tend to
    have arrest=1, and points far away tend to have arrest=0. Lets us
    test directional sanity ('closer query → higher probability') without
    pinning to a specific value.
    """
    rng = np.random.default_rng(seed=42)
    n = 50

    lat = rng.uniform(41.7, 42.0, n)
    lon = rng.uniform(-87.9, -87.5, n)
    hour = rng.integers(0, 24, n)
    dow = rng.integers(0, 7, n)
    month = rng.integers(1, 13, n)
    doy = rng.integers(1, 366, n)

    raw = np.column_stack([
        lat, lon,
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
        np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12),
        np.sin(2 * np.pi * doy / 365), np.cos(2 * np.pi * doy / 365),
    ]).astype(np.float32)

    # Label: closer to (41.88, -87.63) → more likely arrest=1
    dist = np.sqrt((lat - 41.88) ** 2 + (lon + 87.63) ** 2)
    label = (dist < np.median(dist)).astype(np.int8)

    features_mean = raw.mean(axis=0)
    features_std = raw.std(axis=0) + 1e-8
    scaled = (raw - features_mean) / features_std
    features_aug = np.hstack([np.ones((n, 1), dtype=np.float32), scaled])

    return {
        "features_aug": features_aug,
        "label": label,
        "features_mean": features_mean,
        "features_std": features_std,
    }


@pytest.fixture
def sample_query():
    """Raw 10-d query vector at downtown Chicago, 2pm."""
    return np.array([
        41.8781, -87.6298,
        np.sin(2 * np.pi * 14 / 24), np.cos(2 * np.pi * 14 / 24),
        np.sin(2 * np.pi * 2 / 7), np.cos(2 * np.pi * 2 / 7),
        np.sin(2 * np.pi * 6 / 12), np.cos(2 * np.pi * 6 / 12),
        np.sin(2 * np.pi * 166 / 365), np.cos(2 * np.pi * 166 / 365),
    ], dtype=float)


class TestFeatureColumns:
    def test_n_features_is_ten(self):
        assert N_FEATURES == 10

    def test_feature_columns_length_matches(self):
        assert len(FEATURE_COLUMNS) == N_FEATURES

    def test_feature_order_starts_with_lat_lon(self):
        # Critical: clean.py and the Flask route both rely on this ordering.
        assert FEATURE_COLUMNS[0] == "latitude"
        assert FEATURE_COLUMNS[1] == "longitude"


class TestStandardizeAndAugmentQuery:
    def test_output_shape_includes_intercept(self, tiny_artifact, sample_query):
        result = standardize_and_augment_query(
            sample_query,
            tiny_artifact["features_mean"],
            tiny_artifact["features_std"],
        )
        assert result.shape == (N_FEATURES + 1,)

    def test_intercept_is_one(self, tiny_artifact, sample_query):
        result = standardize_and_augment_query(
            sample_query,
            tiny_artifact["features_mean"],
            tiny_artifact["features_std"],
        )
        assert result[0] == 1.0

    def test_scaling_matches_manual_computation(self, tiny_artifact, sample_query):
        expected = (sample_query - tiny_artifact["features_mean"]) / tiny_artifact["features_std"]
        result = standardize_and_augment_query(
            sample_query,
            tiny_artifact["features_mean"],
            tiny_artifact["features_std"],
        )
        assert result[1:] == pytest.approx(expected)

    def test_wrong_shape_raises(self, tiny_artifact):
        bad_query = np.array([1.0, 2.0, 3.0])  # only 3 features
        with pytest.raises(ValueError, match="shape"):
            standardize_and_augment_query(
                bad_query,
                tiny_artifact["features_mean"],
                tiny_artifact["features_std"],
            )


class TestFitLogisticRidge:
    def test_output_shape_matches_features(self):
        X = np.random.default_rng(0).normal(size=(20, 5))
        X[:, 0] = 1.0  # intercept column
        y = np.random.default_rng(1).integers(0, 2, 20).astype(float)
        beta = fit_logistic_ridge(X, y)
        assert beta.shape == (5,)

    def test_separable_data_learns_correct_sign(self):
        # Construct linearly separable data: positive feature → label 1
        rng = np.random.default_rng(42)
        n = 100
        feature = rng.normal(size=n)
        y = (feature > 0).astype(float)
        X = np.column_stack([np.ones(n), feature])
        beta = fit_logistic_ridge(X, y, n_iter=2000)
        # Coefficient on the predictive feature should be positive
        assert beta[1] > 0


class TestPredictArrestProbability:
    @pytest.mark.parametrize("k", [1, 5, 25, 50])
    def test_returns_probability_in_unit_interval(self, tiny_artifact, sample_query, k):
        p = predict_arrest_probability(tiny_artifact, sample_query, k)
        assert 0.0 <= p <= 1.0

    def test_returns_float(self, tiny_artifact, sample_query):
        p = predict_arrest_probability(tiny_artifact, sample_query, k=10)
        assert isinstance(p, float)

    def test_k_too_large_raises(self, tiny_artifact, sample_query):
        n = tiny_artifact["features_aug"].shape[0]
        with pytest.raises(ValueError, match="exceeds dataset size"):
            predict_arrest_probability(tiny_artifact, sample_query, k=n + 1)

    def test_k_zero_raises(self, tiny_artifact, sample_query):
        with pytest.raises(ValueError, match="k must be >= 1"):
            predict_arrest_probability(tiny_artifact, sample_query, k=0)

    def test_negative_k_raises(self, tiny_artifact, sample_query):
        with pytest.raises(ValueError, match="k must be >= 1"):
            predict_arrest_probability(tiny_artifact, sample_query, k=-5)

    def test_return_details_returns_dict(self, tiny_artifact, sample_query):
        result = predict_arrest_probability(
            tiny_artifact, sample_query, k=10, return_details=True
        )
        assert set(result.keys()) == {
            "probability", "beta", "neighbor_indices", "k", "n_total"
        }
        assert result["k"] == 10
        assert len(result["neighbor_indices"]) == 10
        assert result["beta"].shape == (N_FEATURES + 1,)

    def test_neighbor_indices_are_within_bounds(self, tiny_artifact, sample_query):
        n = tiny_artifact["features_aug"].shape[0]
        result = predict_arrest_probability(
            tiny_artifact, sample_query, k=10, return_details=True
        )
        for idx in result["neighbor_indices"]:
            assert 0 <= idx < n

    def test_deterministic(self, tiny_artifact, sample_query):
        # Same inputs → same output (no randomness in the algorithm).
        p1 = predict_arrest_probability(tiny_artifact, sample_query, k=10)
        p2 = predict_arrest_probability(tiny_artifact, sample_query, k=10)
        assert p1 == p2