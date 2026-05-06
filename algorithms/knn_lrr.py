"""KNN Logistic Ridge Regression — from-scratch arrest probability prediction.

Single source of truth for the algorithm. Imported by:
  - src/flask_app/__init__.py  (the /api/predict route)
  - src/crime_knn.py           (the CLI demo)
"""
from __future__ import annotations

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from max_heap import MaxHeap, euclidean_distance


# Feature column order — must match precompute_knn_arrays.py and clean.py
FEATURE_COLUMNS = [
    "latitude", "longitude",
    "hour_sin", "hour_cos",
    "day_of_week_sin", "day_of_week_cos",
    "month_sin", "month_cos",
    "day_of_year_sin", "day_of_year_cos",
]
N_FEATURES = len(FEATURE_COLUMNS)  # 10


def standardize_and_augment_query(query_raw, features_mean, features_std):
    """Apply the artifact's saved scaler to a raw query and prepend the intercept."""
    if query_raw.shape != (N_FEATURES,):
        raise ValueError(
            f"query_raw must have shape ({N_FEATURES},), got {query_raw.shape}"
        )
    query_scaled = (query_raw - features_mean) / features_std
    return np.concatenate([[1.0], query_scaled])


def find_k_nearest(features_aug, query_aug, k):
    """Find indices of the k nearest neighbors using the from-scratch MaxHeap."""
    heap = MaxHeap(capacity=k)
    query_list = query_aug.tolist()
    for idx in range(features_aug.shape[0]):
        dist = euclidean_distance(features_aug[idx].tolist(), query_list)
        heap.add(dist, float(idx))
    return [int(target) for _, target in heap.get_all()]


def fit_logistic_ridge(X_local, y_local, alpha=1.0, lr=0.1, n_iter=1000):
    """Fit logistic ridge regression by gradient descent (intercept not penalized)."""
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    beta = np.zeros(X_local.shape[1])
    k = len(y_local)
    for _ in range(n_iter):
        p_hat = sigmoid(X_local @ beta)
        grad = (X_local.T @ (p_hat - y_local)) / k
        grad[1:] += alpha * beta[1:]   # L2 penalty, skip intercept
        beta -= lr * grad
    return beta


def predict_arrest_probability(artifact, query_raw, k, *, return_details=False):
    """Predict P(arrest) for a query using KNN logistic ridge regression.

    Args:
        artifact: dict with 'features_aug', 'label', 'features_mean', 'features_std'.
        query_raw: shape (10,) — raw feature vector in FEATURE_COLUMNS order.
        k: number of nearest neighbors (1..n_total).
        return_details: if True, return dict with probability, beta, neighbor indices.

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

    query_aug = standardize_and_augment_query(query_raw, features_mean, features_std)
    neighbor_indices = find_k_nearest(features_aug, query_aug, k)

    X_local = features_aug[neighbor_indices]
    y_local = label[neighbor_indices].astype(float)
    beta = fit_logistic_ridge(X_local, y_local)

    z = float(query_aug @ beta)
    probability = float(1.0 / (1.0 + np.exp(-np.clip(z, -500, 500))))

    if return_details:
        return {
            "probability": probability,
            "beta": beta,
            "neighbor_indices": neighbor_indices,
            "k": k,
            "n_total": n_total,
        }
    return probability