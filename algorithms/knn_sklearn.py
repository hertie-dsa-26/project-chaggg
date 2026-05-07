"""
algorithms/knn_sklearn.py
─────────────────────────────────────────────────────────────────────────────
sklearn-based mirror of the from-scratch knn_lrr algorithm.

Purpose
-------
This module is a **trusted benchmark / validation mirror**, not the academic
deliverable. Given the same (query, crime_type, k) inputs, it should produce
arrest-probability predictions that agree with the from-scratch implementation
to within |Δp| < 0.05.

The from-scratch version remains the primary deliverable; this implementation
exists so the team can detect regressions and implementation drift.

Regularisation note (C ↔ alpha mapping)
----------------------------------------
The from-scratch implementation uses Ridge penalty  α = 1.0:

    loss = NLL + α * ||w||²

sklearn's LogisticRegression uses the *inverse* regularisation strength C:

    loss = NLL + (1/C) * ||w||²

So  α = 1.0  ↔  C = 1/α = 1.0  (they happen to be equal here).
If the from-scratch α changes, update C = 1 / α accordingly.

Intercept handling
------------------
The from-scratch version manually prepends a bias column of 1s.
sklearn handles the intercept internally when fit_intercept=True (default).
Do NOT add a bias column manually here — it would double-count the intercept.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# ── Re-use the shared helper from the from-scratch module ─────────────────────
# Import whichever function your team settled on; adjust the import path if
# select_crime lives in a different module (e.g. algorithms.knn or algorithms.utils).
#from src.crime_knn import select_crime, commented out just in case
from src.preprocess_data import preprocess_data

# ── Feature columns (must match the from-scratch implementation exactly) ──────
SPATIAL_FEATURES = ["latitude", "longitude"]
CYCLIC_FEATURES = [
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
]
FEATURE_COLS = SPATIAL_FEATURES + CYCLIC_FEATURES
LABEL_COL = "arrest"


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def knn_sklearn(
    query: dict,
    crime_type: str | None = None,
    k: int | None = None,
    *,
    df: pd.DataFrame | None = None,
    alpha: float = 1.0,
    verbose: bool = True,
    random_state: int = 42,
) -> float:
    """
    sklearn-based KNN + locally-fitted Logistic Ridge Regression.

    This is a **validation mirror** of the from-scratch knn_lrr function.
    Both implementations must be called with identical inputs; predictions
    should agree within |Δp| < 0.05.

    Parameters
    ----------
    query : dict
        Feature dictionary for the single query point. Must contain at minimum
        the keys in FEATURE_COLS (spatial + 8 cyclic time features).
        Example::

            {
                "latitude": 41.85,
                "longitude": -87.65,
                "hour_sin": 0.0,
                "hour_cos": 1.0,
                "day_of_week_sin": 0.866,
                "day_of_week_cos": 0.5,
                "month_sin": 0.5,
                "month_cos": 0.866,
                "day_of_year_sin": 0.3,
                "day_of_year_cos": 0.95,
            }

    crime_type : str or None
        Primary crime type string (e.g. "THEFT", "BATTERY").  If None the
        full dataset is used (not recommended — produces a degenerate model).
    k : int or None
        Number of nearest neighbours.  Defaults to 100 if None.
    df : pd.DataFrame or None
        Pre-loaded dataset. If None the function loads the cleaned parquet
        from the default path (``data/cleaned/chicago_crimes_cleaned.parquet``).
    alpha : float
        Ridge penalty strength used in the from-scratch implementation.
        C is set to 1/alpha so that both implementations use the same penalty.
        Default: 1.0  →  C = 1.0.
    verbose : bool
        If True, print diagnostic output mirroring the from-scratch function
        (intercept, coefficients, probability, scenario message).
    random_state : int
        Passed to LogisticRegression for reproducibility.

    Returns
    -------
    float
        Predicted arrest probability for the query point (in [0, 1]).
    """
    # ── Defaults ──────────────────────────────────────────────────────────────
    if k is None:
        k = 100

    # ── Load data ─────────────────────────────────────────────────────────────
    if df is None:
        df = _load_default_data()


    # ── Filter by crime type ───────────────────────────────────────────────────
    if crime_type is not None:
        df_filtered = df[df["primary_type"] == crime_type].reset_index(drop=True)
    else:
        df_filtered = df.copy()

    if df_filtered.empty:
        raise ValueError(
            f"No records found for crime_type={crime_type!r}. "
            "Check the spelling against df['primary_type'].unique()."
        )

    # Remove this comment once the pipeline is wired up.
    _check_features_present(df_filtered)

    # ── Build feature matrix ───────────────────────────────────────────────────
    X = df_filtered[FEATURE_COLS].values.astype(float)
    y = df_filtered[LABEL_COL].astype(int).values

    # ── Build query vector ─────────────────────────────────────────────────────
    q = np.array([[query[col] for col in FEATURE_COLS]], dtype=float)

    # ── Standardise (fit on filtered dataset, apply to query) ─────────────────
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    q_scaled = scaler.transform(q)

    # ── Find k nearest neighbours ──────────────────────────────────────────────
    nn = NearestNeighbors(algorithm="kd_tree", metric="euclidean", n_neighbors=k)
    nn.fit(X_scaled)
    distances, indices = nn.kneighbors(q_scaled)
    neighbor_indices = indices[0]  # shape (k,)

    X_neighbors = X_scaled[neighbor_indices]
    y_neighbors = y[neighbor_indices]

    # Guard: LogisticRegression requires at least one positive and one negative
    # label in the training set.
    if len(np.unique(y_neighbors)) < 2:
        # All neighbours have the same label — return the majority probability.
        prob = float(y_neighbors.mean())
        if verbose:
            _print_diagnostics(
                intercept=np.nan,
                coefs=np.full(len(FEATURE_COLS), np.nan),
                prob=prob,
                crime_type=crime_type,
                k=k,
                degenerate=True,
            )
        return prob

    # ── Fit local Logistic Regression ─────────────────────────────────────────
    # C = 1/alpha  (see module docstring for derivation)
    C = 1.0 / alpha
    clf = LogisticRegression(
        penalty="l2",
        C=C,
        solver="lbfgs",
        fit_intercept=True,   # sklearn handles the bias — do NOT add a 1-column
        max_iter=1000,
        random_state=random_state,
    )
    clf.fit(X_neighbors, y_neighbors)

    # ── Predict for the query ─────────────────────────────────────────────────
    # predict_proba returns [[P(arrest=0), P(arrest=1)]]
    prob = float(clf.predict_proba(q_scaled)[0, 1])

    # ── Diagnostics ───────────────────────────────────────────────────────────
    if verbose:
        _print_diagnostics(
            intercept=float(clf.intercept_[0]),
            coefs=clf.coef_[0],
            prob=prob,
            crime_type=crime_type,
            k=k,
            degenerate=False,
        )

    return prob


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_default_data() -> pd.DataFrame:
    return preprocess_data()

def _check_features_present(df: pd.DataFrame) -> None:
    """Raise a clear error if any expected feature column is missing."""
    missing = [c for c in FEATURE_COLS + [LABEL_COL] if c not in df.columns]
    if missing:
        raise KeyError(
            f"The following columns are missing from the dataset: {missing}. "
            "Ensure preprocess_data() has been run and the parquet contains "
            "the cyclic time features."
        )


def _print_diagnostics(
    *,
    intercept: float,
    coefs: np.ndarray,
    prob: float,
    crime_type: str | None,
    k: int,
    degenerate: bool,
) -> None:
    """Print side-by-side-comparable diagnostics (mirrors from-scratch output)."""
    print("=" * 60)
    print("[sklearn mirror] knn_sklearn diagnostic output")
    print(f"  crime_type : {crime_type!r}")
    print(f"  k          : {k}")
    if degenerate:
        print("  NOTE: All neighbours share the same label — returning mean.")
    else:
        print(f"  intercept  : {intercept:.6f}")
        # Top-5 coefficients by absolute magnitude
        order = np.argsort(np.abs(coefs))[::-1]
        print("  top-5 coefficients (by |magnitude|):")
        for rank, idx in enumerate(order[:5], 1):
            print(f"    {rank}. {FEATURE_COLS[idx]:<25s}  {coefs[idx]:+.6f}")
    print(f"  P(arrest)  : {prob:.4f}  ({prob * 100:.1f}%)")
    _print_scenario_message(prob)
    print("=" * 60)


def _print_scenario_message(prob: float) -> None:
    """Qualitative interpretation of the probability (mirrors from-scratch)."""
    if prob >= 0.75:
        msg = "HIGH likelihood of arrest."
    elif prob >= 0.50:
        msg = "MODERATE–HIGH likelihood of arrest."
    elif prob >= 0.25:
        msg = "MODERATE–LOW likelihood of arrest."
    else:
        msg = "LOW likelihood of arrest."
    print(f"  scenario   : {msg}")