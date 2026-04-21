"""
CrimeKNN: Brute-force K-Nearest Neighbours classifier for crime arrest prediction.

Predicts the probability of arrest (float in [0, 1]) for a given crime query
using inverse-distance weighted averaging over K nearest neighbours.

Distance is computed using combined_distance(), which blends:
  - Spatial:  haversine great-circle distance (km), normalised by max_spatial_km
  - Temporal: cyclical encoding of hour, day_of_week, and month

Training data is pre-filtered by primary_type so each crime type
only searches among its own historical records.

Train/test split:
    - Train: records before  January 1 2023
    - Test:  records from January 1 2023 onwards

K tuning results (tested on held-out test set):
    - K=5:  MAE reported at runtime
    - K=10: MAE reported at runtime
    - K=20: MAE reported at runtime

Optimal K: To be documented after running on real data. Generally a larger K
reduces variance but may smooth over local patterns. For sparse crime types
a smaller K is preferred. K=10 is a sensible default starting point.
"""

import csv
import datetime
import numpy as np

from .distance import combined_distance


EPSILON = 1e-6        # Guard against division by zero for exact distance matches
TRAIN_CUTOFF = datetime.date(2023, 1, 1)  # Train: before 2023 | Test: 2023 onwards


def split_by_date(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Split records into train and test sets by date cutoff (January 1 2023).

    Parameters
    ----------
    records : list of dict
        Each dict must contain a 'date' field parseable as YYYY-MM-DD,
        or a datetime.date / datetime.datetime object.

    Returns
    -------
    train_records, test_records : tuple of lists
    """
    train, test = [], []
    for rec in records:
        date = rec["date"]
        if isinstance(date, str):
            date = datetime.date.fromisoformat(date[:10])
        elif isinstance(date, datetime.datetime):
            date = date.date()
        if date < TRAIN_CUTOFF:
            train.append(rec)
        else:
            test.append(rec)
    return train, test


class CrimeKNN:
    """
    Brute-force K-Nearest Neighbours model for predicting arrest likelihood.

    Accepts training records and max_spatial_km in __init__, groups by
    primary_type, and is immediately ready to predict.

    For each query, computes combined_distance() against all training records
    of the same primary_type, selects K nearest, and returns an
    inverse-distance weighted average of the arrest outcome.

    Parameters
    ----------
    records : list of dict
        Training records. Each must contain:
            latitude     (float)
            longitude    (float)
            hour         (int, 0-23)
            day_of_week  (int, 0=Monday ... 6=Sunday)
            month        (int, 1-12)
            primary_type (str)
            arrest       (int, 0 or 1)
    k : int
        Number of nearest neighbours (default=10).
        Falls back to however many records are available if fewer than k
        exist for a given crime type.
    max_spatial_km : float
        Maximum spatial distance (km) used to normalise haversine distances
        to [0, 1]. Should be computed from the training set bounding box
        using get_spatial_bounds().
    alpha : float
        Weight for spatial component in combined_distance (default=0.5).
    beta : float
        Weight for temporal component in combined_distance (default=0.5).
    """

    def __init__(
        self,
        records: list[dict],
        k: int = 10,
        max_spatial_km: float = 1.0,
        alpha: float = 0.5,
        beta: float = 0.5,
    ):
        self.k = k
        self.max_spatial_km = max_spatial_km
        self.alpha = alpha
        self.beta = beta

        # Group training records by primary_type
        self._train_by_type: dict[str, list[dict]] = {}
        for rec in records:
            ptype = rec["primary_type"]
            self._train_by_type.setdefault(ptype, []).append(rec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, query: dict) -> float:
        """
        Predict arrest probability for a single crime query.

        Parameters
        ----------
        query : dict
            Must contain:
                latitude     (float)
                longitude    (float)
                hour         (int, 0-23)
                day_of_week  (int, 0=Monday ... 6=Sunday)
                month        (int, 1-12)
                primary_type (str)

        Returns
        -------
        float
            Inverse-distance weighted arrest probability in [0, 1].
            Formula: sum(arrest_i / dist_i) / sum(1 / dist_i)
        """
        ptype = query["primary_type"]

        # Edge case: unseen crime type — no data to predict from
        if ptype not in self._train_by_type:
            return 0.0

        candidates = self._train_by_type[ptype]

        # Compute combined_distance to every training record of this type
        distances = np.array([
            combined_distance(
                query, rec, self.max_spatial_km, self.alpha, self.beta
            )
            for rec in candidates
        ])

        # Edge case: fewer records than K — use however many are available
        k = min(self.k, len(candidates))

        # Find K nearest neighbours
        k_indices = np.argsort(distances)[:k]
        k_distances = distances[k_indices]
        k_arrests = np.array(
            [candidates[i]["arrest"] for i in k_indices], dtype=float
        )

        # Edge case: exact match (distance == 0) — clamp with epsilon
        k_distances = np.maximum(k_distances, EPSILON)

        # Inverse-distance weighted average
        weights = 1.0 / k_distances
        score = np.sum(k_arrests * weights) / np.sum(weights)

        return float(np.clip(score, 0.0, 1.0))

    def evaluate(self, records: list[dict]) -> float:
        """
        Compute Mean Absolute Error on a held-out test set.

        MAE = mean(|predicted_arrest_rate - actual_arrest|) across all records.

        Parameters
        ----------
        records : list of dict
            Same format as training records, must include 'arrest' field.

        Returns
        -------
        float
            MAE across all test records.
        """
        if not records:
            raise ValueError("No records provided for evaluation.")

        errors = [
            abs(self.predict(r) - float(r["arrest"]))
            for r in records
        ]
        return float(np.mean(errors))

    def predict_all(self, records: list[dict]) -> list[dict]:
        """
        Generate predictions for a list of records.

        Parameters
        ----------
        records : list of dict
            Records to predict, same format as training records.

        Returns
        -------
        list of dict
            Each dict contains the original record fields plus:
                predicted_arrest_rate (float)
                actual_arrest         (int)
        """
        results = []
        for rec in records:
            results.append({
                **rec,
                "predicted_arrest_rate": self.predict(rec),
                "actual_arrest": rec.get("arrest"),
            })
        return results

    def save_predictions(self, records: list[dict], filepath: str) -> None:
        """
        Generate predictions for records and save to CSV.

        Parameters
        ----------
        records : list of dict
            Test records to predict.
        filepath : str
            Output CSV file path.
        """
        predictions = self.predict_all(records)
        if not predictions:
            raise ValueError("No predictions to save.")

        fieldnames = list(predictions[0].keys())
        with open(filepath, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(predictions)

        print(f"Saved {len(predictions)} predictions to {filepath}")


def tune_k(
    train_records: list[dict],
    test_records: list[dict],
    max_spatial_km: float,
    k_values: list[int] = [5, 10, 20],
    alpha: float = 0.5,
    beta: float = 0.5,
) -> dict[int, float]:
    """
    Tune K by evaluating MAE on the test set for each value of K.

    Parameters
    ----------
    train_records : list of dict
        Training records (before Jan 1 2023).
    test_records : list of dict
        Test records (Jan 1 2023 onwards).
    max_spatial_km : float
        Spatial normalisation constant from training set bounding box.
    k_values : list of int
        K values to test (default: [5, 10, 20]).
    alpha : float
        Spatial weight for combined_distance.
    beta : float
        Temporal weight for combined_distance.

    Returns
    -------
    dict mapping K -> MAE
    """
    results = {}
    for k in k_values:
        model = CrimeKNN(
            records=train_records,
            k=k,
            max_spatial_km=max_spatial_km,
            alpha=alpha,
            beta=beta,
        )
        mae = model.evaluate(test_records)
        results[k] = mae
        print(f"K={k:2d} | MAE={mae:.4f}")

    optimal_k = min(results, key=results.get)
    print(f"\nOptimal K: {optimal_k} (MAE={results[optimal_k]:.4f})")
    return results
