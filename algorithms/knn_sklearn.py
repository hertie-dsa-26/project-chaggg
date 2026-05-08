"""KNN Logistic Ridge Regression — sklearn validation mirror.

Mirrors algorithms/knn_lrr.py:predict_arrest_probability using sklearn
primitives. Consumes the same precomputed artifact (features_aug, label,
features_mean, features_std) so both implementations operate on identical
scaled neighborhoods.

Agreement target: |Δp| < 0.05 versus the from-scratch implementation for
identical (artifact, query_raw, k) inputs.

Regularisation mapping
----------------------
From-scratch loss:  NLL + alpha * ||w||²        (alpha = 1.0 default)
sklearn loss:       NLL + (1/C) * ||w||²
So C = 1 / alpha.

Intercept handling
------------------
The artifact's features_aug has a leading bias column of 1s (column 0).
sklearn's LogisticRegression adds its own intercept when fit_intercept=True,
so we slice the bias column off before passing to sklearn. Distances in the
augmented space are identical to the un-augmented scaled space because the
bias column is constant, so neighbor sets agree.
"""
from __future__ import annotations

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression

from algorithms.knn_lrr import N_FEATURES


def predict_arrest_probability_sklearn(
    artifact,
    query_raw,
    k,
    *,
    alpha=1.0,
    return_details=False,
):
    """Predict P(arrest) for a query using sklearn KNN + local logistic ridge.

    Args:
        artifact: dict with 'features_aug', 'label', 'features_mean', 'features_std'.
        query_raw: shape (10,) — raw feature vector in FEATURE_COLUMNS order.
        k: number of nearest neighbors (1..n_total).
        alpha: ridge penalty strength (matches from-scratch). C = 1/alpha.
        return_details: if True, return dict with probability, coef, intercept,
            neighbor indices.

    Returns:
        float in [0, 1], or dict if return_details=True.
    """
    features_aug = artifact["features_aug"]
    label = artifact["label"]
    features_mean = artifact["features_mean"]
    features_std = artifact["features_std"]

    n_total = features_aug.shape[0]
    if k > n_total:
        raise ValueError(f"k={k} exceeds dataset size n={n_total}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if query_raw.shape != (N_FEATURES,):
        raise ValueError(
            f"query_raw must have shape ({N_FEATURES},), got {query_raw.shape}"
        )

    # Drop the bias column — sklearn handles the intercept internally.
    X_scaled = features_aug[:, 1:]  # shape (n, 10)

    # Standardise the query using the artifact's saved scaler — same transform
    # used to build features_aug, so query and neighbours live in the same space.
    query_scaled = ((query_raw - features_mean) / features_std).reshape(1, -1)

    # k nearest neighbours on the scaled 10-d space.
    nn = NearestNeighbors(algorithm="auto", metric="euclidean", n_neighbors=k)
    nn.fit(X_scaled)
    _, indices = nn.kneighbors(query_scaled)
    neighbor_indices = indices[0]

    X_local = X_scaled[neighbor_indices]
    y_local = label[neighbor_indices].astype(int)

    # Degenerate case: all neighbours share a label. sklearn's LR raises here,
    # so fall back to the majority probability (matches the natural limit of
    # the from-scratch optimiser when all y_local are identical).
    if len(np.unique(y_local)) < 2:
        probability = float(y_local.mean())
        if return_details:
            return {
                "probability": probability,
                "coef": np.full(N_FEATURES, np.nan),
                "intercept": np.nan,
                "neighbor_indices": neighbor_indices.tolist(),
                "k": k,
                "n_total": n_total,
                "degenerate": True,
            }
        return probability

    clf = LogisticRegression(
        penalty="l2",
        C=1.0 / alpha,
        solver="lbfgs",
        fit_intercept=True,
        max_iter=1000,
    )
    clf.fit(X_local, y_local)

    probability = float(clf.predict_proba(query_scaled)[0, 1])

    if return_details:
        return {
            "probability": probability,
            "coef": clf.coef_[0],
            "intercept": float(clf.intercept_[0]),
            "neighbor_indices": neighbor_indices.tolist(),
            "k": k,
            "n_total": n_total,
            "degenerate": False,
        }
    return probability