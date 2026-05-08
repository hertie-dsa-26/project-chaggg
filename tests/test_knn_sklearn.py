"""Tests for the sklearn KNN-LRR validation mirror.

The sklearn implementation must agree with the from-scratch implementation
to within |Δp| < 0.05 on identical (artifact, query_raw, k) inputs. These
tests run against real precomputed artifacts; if the artifacts directory is
missing, the whole module is skipped.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.flask_app.load_crime_artifacts import load_knn_arrays
from algorithms.knn_lrr import predict_arrest_probability, N_FEATURES
from algorithms.knn_sklearn import predict_arrest_probability_sklearn


# ── Module-level fixture: load artifacts once, skip if unavailable ───────────

@pytest.fixture(scope="module")
def artifacts():
    try:
        return load_knn_arrays()
    except FileNotFoundError as e:
        pytest.skip(f"KNN artifacts not available: {e}")


CRIME_TYPES = ["theft", "battery", "criminal_damage"]
K_VALUES = [10, 25, 100]
AGREEMENT_TOL = 0.05  # |Δp| target from the sklearn module docstring


# A handful of plausible Chicago query points (lat, lon, hour, day_of_week,
# month, day_of_year). Cyclic encodings are derived in the helper below.
QUERY_POINTS = [
    # downtown loop, weekday afternoon, mid-summer
    (41.8781, -87.6298, 14, 2, 7, 196),
    # north side, friday night
    (41.9484, -87.6553, 23, 4, 10, 280),
    # south side, sunday morning
    (41.7510, -87.6051, 9, 6, 2, 45),
]


def _build_query_raw(lat, lon, hour, dow, month, doy):
    """Match the cyclic encoding used in the Flask route and clean.py."""
    def sincos(value, period):
        angle = 2.0 * np.pi * value / period
        return np.sin(angle), np.cos(angle)

    h_s, h_c = sincos(hour, 24)
    d_s, d_c = sincos(dow, 7)
    m_s, m_c = sincos(month, 12)
    y_s, y_c = sincos(doy, 365)
    return np.array([lat, lon, h_s, h_c, d_s, d_c, m_s, m_c, y_s, y_c], dtype=float)


# ─────────────────────────────────────────────────────────────────────────────
# Agreement: sklearn vs from-scratch
# ─────────────────────────────────────────────────────────────────────────────

class TestAgreement:
    """Both implementations should produce predictions within AGREEMENT_TOL."""

    @pytest.mark.parametrize("crime_type", CRIME_TYPES)
    @pytest.mark.parametrize("query", QUERY_POINTS)
    @pytest.mark.parametrize("k", K_VALUES)
    def test_predictions_agree(self, artifacts, crime_type, query, k):
        if crime_type not in artifacts:
            pytest.skip(f"Artifact {crime_type!r} not in precomputed set")

        artifact = artifacts[crime_type]
        query_raw = _build_query_raw(*query)

        p_scratch = predict_arrest_probability(artifact, query_raw, k)
        p_sklearn = predict_arrest_probability_sklearn(artifact, query_raw, k)

        # Both must be valid probabilities first.
        assert 0.0 <= p_scratch <= 1.0
        assert 0.0 <= p_sklearn <= 1.0

        delta = abs(p_scratch - p_sklearn)
        assert delta < AGREEMENT_TOL, (
            f"Disagreement for crime={crime_type}, k={k}, query={query}: "
            f"scratch={p_scratch:.4f}, sklearn={p_sklearn:.4f}, |Δ|={delta:.4f}"
        )

    def test_neighbor_sets_match(self, artifacts):
        """Neighbour indices should be identical (modulo tie ordering).

        The from-scratch search uses the augmented 11-d space; sklearn searches
        the un-augmented 10-d space. Because the bias column is constant, the
        induced neighbourhoods must be the same set.
        """
        if "theft" not in artifacts:
            pytest.skip("theft artifact not available")

        artifact = artifacts["theft"]
        query_raw = _build_query_raw(*QUERY_POINTS[0])
        k = 25

        scratch = predict_arrest_probability(
            artifact, query_raw, k, return_details=True
        )
        sklearn = predict_arrest_probability_sklearn(
            artifact, query_raw, k, return_details=True
        )

        assert set(scratch["neighbor_indices"]) == set(sklearn["neighbor_indices"])


# ─────────────────────────────────────────────────────────────────────────────
# Contract: input validation matches the from-scratch contract
# ─────────────────────────────────────────────────────────────────────────────

class TestInputValidation:
    """sklearn mirror must raise the same ValueErrors as the from-scratch fn."""

    def test_k_too_large(self, artifacts):
        if "theft" not in artifacts:
            pytest.skip("theft artifact not available")
        artifact = artifacts["theft"]
        n_total = artifact["label"].shape[0]
        query_raw = _build_query_raw(*QUERY_POINTS[0])
        with pytest.raises(ValueError, match="exceeds dataset size"):
            predict_arrest_probability_sklearn(artifact, query_raw, k=n_total + 1)

    def test_k_zero(self, artifacts):
        if "theft" not in artifacts:
            pytest.skip("theft artifact not available")
        artifact = artifacts["theft"]
        query_raw = _build_query_raw(*QUERY_POINTS[0])
        with pytest.raises(ValueError, match="k must be >= 1"):
            predict_arrest_probability_sklearn(artifact, query_raw, k=0)

    def test_query_wrong_shape(self, artifacts):
        if "theft" not in artifacts:
            pytest.skip("theft artifact not available")
        artifact = artifacts["theft"]
        bad_query = np.zeros(N_FEATURES - 1)
        with pytest.raises(ValueError, match="query_raw must have shape"):
            predict_arrest_probability_sklearn(artifact, bad_query, k=10)


# ─────────────────────────────────────────────────────────────────────────────
# Behaviour: idempotence and return-shape
# ─────────────────────────────────────────────────────────────────────────────

class TestBehaviour:
    def test_idempotent(self, artifacts):
        """Same inputs → same output (sklearn LR with lbfgs is deterministic)."""
        if "theft" not in artifacts:
            pytest.skip("theft artifact not available")
        artifact = artifacts["theft"]
        query_raw = _build_query_raw(*QUERY_POINTS[0])

        p1 = predict_arrest_probability_sklearn(artifact, query_raw, k=50)
        p2 = predict_arrest_probability_sklearn(artifact, query_raw, k=50)
        assert p1 == pytest.approx(p2)

    def test_return_details_structure(self, artifacts):
        if "theft" not in artifacts:
            pytest.skip("theft artifact not available")
        artifact = artifacts["theft"]
        query_raw = _build_query_raw(*QUERY_POINTS[0])

        details = predict_arrest_probability_sklearn(
            artifact, query_raw, k=50, return_details=True
        )

        assert set(details.keys()) >= {
            "probability", "coef", "intercept",
            "neighbor_indices", "k", "n_total", "degenerate",
        }
        assert 0.0 <= details["probability"] <= 1.0
        assert details["coef"].shape == (N_FEATURES,)
        assert details["k"] == 50
        assert len(details["neighbor_indices"]) == 50

    def test_probability_in_unit_interval(self, artifacts):
        """Sweep k values; sklearn must always return a valid probability."""
        if "battery" not in artifacts:
            pytest.skip("battery artifact not available")
        artifact = artifacts["battery"]
        query_raw = _build_query_raw(*QUERY_POINTS[1])
        for k in [1, 5, 10, 50, 100]:
            p = predict_arrest_probability_sklearn(artifact, query_raw, k=k)
            assert 0.0 <= p <= 1.0, f"out-of-range probability {p} at k={k}"