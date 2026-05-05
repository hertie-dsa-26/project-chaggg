"""
Location-only KNN baseline (sklearn) for arrest probability prediction.

This is the ablation baseline against the from-scratch KNN + local logistic
regression pipeline in ``src/crime_knn.py``. For every crime ``primary_type``
we fit ``sklearn.neighbors.KNeighborsClassifier`` on standardised
``(latitude, longitude)`` features only, tune ``k`` via stratified CV on the
training set (candidates: 10, 25, 50, 100, 250) and report test-set
log-loss and accuracy.

The from-scratch requirement does not apply here — sklearn is allowed for
the comparison baseline.

Dependency note (issue: location-only KNN comparison model)
-----------------------------------------------------------
This script is meant to share data preparation with the from-scratch model.
Until @ght-1's "Train/test split + standardization" PR lands on ``main``,
the helpers below (``split_train_test_by_year`` and the inline
``StandardScaler`` step) replicate the expected behaviour. When that PR
merges, swap those helpers for the shared utilities so both models consume
identical training and test sets.

Run from repo root (after cleaned data exists):

.. code:: shell

    PYTHONUNBUFFERED=1 uv run python scripts/knn_location_only.py
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_K_CANDIDATES: tuple[int, ...] = (10, 25, 50, 100, 250)
DEFAULT_TRAIN_YEARS: tuple[int, int] = (2015, 2022)
DEFAULT_TEST_YEARS: tuple[int, int] = (2023, 2024)


# ---------------------------------------------------------------------------
# Data preparation helpers
# ---------------------------------------------------------------------------
def split_train_test_by_year(
    df: pd.DataFrame,
    train_years: tuple[int, int] = DEFAULT_TRAIN_YEARS,
    test_years: tuple[int, int] = DEFAULT_TEST_YEARS,
    date_col: str = "date",
    year_col: str = "year",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Year-based train/test split.

    Mirrors the convention used elsewhere in the project (DBSCAN hotspots,
    KNN forecast). Replace this with @ght-1's shared splitter once that PR
    is merged so the location-only baseline and the main model share an
    identical split.
    """
    if year_col in df.columns:
        year = pd.to_numeric(df[year_col], errors="coerce")
    else:
        year = pd.to_datetime(df[date_col], errors="coerce").dt.year

    train_mask = year.between(train_years[0], train_years[1])
    test_mask = year.between(test_years[0], test_years[1])
    train = df.loc[train_mask].reset_index(drop=True)
    test = df.loc[test_mask].reset_index(drop=True)
    return train, test


def standardize_locations(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: tuple[str, str] = ("latitude", "longitude"),
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit ``StandardScaler`` on train only, apply to both splits.

    Replace with the shared standardisation helper once @ght-1's PR is
    merged so the from-scratch model and this baseline share the exact
    same scaler state.
    """
    scaler = StandardScaler()
    X_train = scaler.fit_transform(train[list(feature_cols)].to_numpy(dtype=float))
    X_test = scaler.transform(test[list(feature_cols)].to_numpy(dtype=float))
    return X_train, X_test, scaler


def slugify(name: str) -> str:
    """Turn a crime ``primary_type`` into a safe filename component."""
    s = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_").lower()
    return s or "unnamed"


# ---------------------------------------------------------------------------
# Per-crime-type training + evaluation
# ---------------------------------------------------------------------------
@dataclass
class CrimeTypeResult:
    """Container for per-crime-type CV+test outcomes."""

    crime_type: str
    n_train: int
    n_test: int
    positive_rate_train: float
    positive_rate_test: float
    skipped: str | None = None
    k_candidates: list[int] = field(default_factory=list)
    cv_neg_log_loss: list[dict[str, float]] = field(default_factory=list)
    best_k: int | None = None
    test_log_loss: float | None = None
    test_accuracy: float | None = None
    fit_seconds: float | None = None
    predictions: pd.DataFrame | None = field(default=None, repr=False)

    def metrics_dict(self) -> dict[str, object]:
        """Serialisable metrics (no DataFrame)."""
        out: dict[str, object] = {
            "crime_type": self.crime_type,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "positive_rate_train": self.positive_rate_train,
            "positive_rate_test": self.positive_rate_test,
        }
        if self.skipped is not None:
            out["skipped"] = self.skipped
            return out
        out.update(
            {
                "k_candidates": self.k_candidates,
                "best_k": self.best_k,
                "cv_neg_log_loss": self.cv_neg_log_loss,
                "test_log_loss": self.test_log_loss,
                "test_accuracy": self.test_accuracy,
                "fit_seconds": self.fit_seconds,
            }
        )
        return out


def _valid_k_candidates(
    candidates: Iterable[int], n_train: int, cv_folds: int
) -> list[int]:
    """Trim ``k`` candidates to those that fit inside a CV training fold."""
    fold_size = (n_train * (cv_folds - 1)) // cv_folds
    valid = sorted({int(k) for k in candidates if 1 <= int(k) <= max(fold_size, 1)})
    if not valid:
        # Fall back to the smallest candidate clipped to fold size.
        smallest = max(min(int(k) for k in candidates), 1)
        valid = [min(smallest, max(fold_size, 1))]
    return valid


def evaluate_crime_type(
    crime_type: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    k_candidates: Iterable[int] = DEFAULT_K_CANDIDATES,
    cv_folds: int = 5,
    random_state: int = 42,
    feature_cols: tuple[str, str] = ("latitude", "longitude"),
    target_col: str = "arrest",
) -> CrimeTypeResult:
    """Tune ``k`` on ``train`` via stratified CV, evaluate on ``test``.

    Returns a :class:`CrimeTypeResult` with metrics and predictions. If the
    crime type cannot be evaluated (e.g. single-class train/test, empty
    splits) the ``skipped`` field explains why.
    """
    n_train = len(train)
    n_test = len(test)

    base = CrimeTypeResult(
        crime_type=crime_type,
        n_train=n_train,
        n_test=n_test,
        positive_rate_train=(
            float(train[target_col].mean()) if n_train and target_col in train.columns else 0.0
        ),
        positive_rate_test=(
            float(test[target_col].mean()) if n_test and target_col in test.columns else 0.0
        ),
    )

    if n_train == 0 or n_test == 0:
        base.skipped = "empty train or test split"
        return base

    y_train = train[target_col].to_numpy(dtype=int)
    y_test = test[target_col].to_numpy(dtype=int)
    if len(np.unique(y_train)) < 2:
        base.skipped = "single-class training set"
        return base
    if len(np.unique(y_test)) < 2:
        base.skipped = "single-class test set"
        return base

    X_train, X_test, _ = standardize_locations(train, test, feature_cols)

    valid_k = _valid_k_candidates(k_candidates, n_train, cv_folds)
    base.k_candidates = valid_k

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    grid = GridSearchCV(
        KNeighborsClassifier(weights="uniform"),
        param_grid={"n_neighbors": valid_k},
        scoring="neg_log_loss",
        cv=cv,
        n_jobs=-1,
        refit=True,
        error_score="raise",
    )

    t0 = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UndefinedMetricWarning)
        grid.fit(X_train, y_train)
    fit_seconds = time.time() - t0

    base.best_k = int(grid.best_params_["n_neighbors"])
    base.fit_seconds = float(fit_seconds)
    base.cv_neg_log_loss = [
        {
            "k": int(k_val),
            "mean_neg_log_loss": float(mean),
            "std_neg_log_loss": float(std),
        }
        for k_val, mean, std in zip(
            grid.cv_results_["param_n_neighbors"],
            grid.cv_results_["mean_test_score"],
            grid.cv_results_["std_test_score"],
            strict=True,
        )
    ]

    classes = list(grid.classes_)
    proba = grid.predict_proba(X_test)
    base.test_log_loss = float(log_loss(y_test, proba, labels=classes))
    base.test_accuracy = float(accuracy_score(y_test, grid.predict(X_test)))

    p_arrest_idx = classes.index(1) if 1 in classes else None
    p_arrest = (
        proba[:, p_arrest_idx]
        if p_arrest_idx is not None
        else np.zeros(len(y_test), dtype=float)
    )

    base.predictions = pd.DataFrame(
        {
            "latitude": test[feature_cols[0]].to_numpy(dtype=float),
            "longitude": test[feature_cols[1]].to_numpy(dtype=float),
            "y_true": y_test,
            "y_pred": grid.predict(X_test),
            "p_arrest": p_arrest,
        }
    )
    return base


def evaluate_all_crime_types(
    df: pd.DataFrame,
    *,
    train_years: tuple[int, int] = DEFAULT_TRAIN_YEARS,
    test_years: tuple[int, int] = DEFAULT_TEST_YEARS,
    k_candidates: Iterable[int] = DEFAULT_K_CANDIDATES,
    cv_folds: int = 5,
    random_state: int = 42,
    crime_types: Iterable[str] | None = None,
    max_rows_per_type: int = 0,
    progress: bool = True,
) -> list[CrimeTypeResult]:
    """Run the per-crime-type pipeline across the full dataset."""
    required = {"primary_type", "latitude", "longitude", "arrest"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"Input data missing required columns: {sorted(missing)}")

    train_full, test_full = split_train_test_by_year(
        df, train_years=train_years, test_years=test_years
    )

    if crime_types is None:
        types = sorted(df["primary_type"].dropna().unique().tolist())
    else:
        types = list(crime_types)

    results: list[CrimeTypeResult] = []
    for i, ct in enumerate(types, start=1):
        ct_train = train_full[train_full["primary_type"] == ct]
        ct_test = test_full[test_full["primary_type"] == ct]

        if max_rows_per_type and len(ct_train) > max_rows_per_type:
            ct_train = ct_train.sample(
                n=max_rows_per_type, random_state=random_state
            ).reset_index(drop=True)

        if progress:
            print(
                f"[{i}/{len(types)}] {ct}: train={len(ct_train):,} test={len(ct_test):,}",
                flush=True,
            )

        result = evaluate_crime_type(
            crime_type=ct,
            train=ct_train,
            test=ct_test,
            k_candidates=k_candidates,
            cv_folds=cv_folds,
            random_state=random_state,
        )
        if progress:
            if result.skipped:
                print(f"    skipped: {result.skipped}", flush=True)
            else:
                print(
                    f"    best k={result.best_k} "
                    f"log_loss={result.test_log_loss:.4f} "
                    f"acc={result.test_accuracy:.4f} "
                    f"({result.fit_seconds:.1f}s)",
                    flush=True,
                )
        results.append(result)
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_int_csv(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def _parse_year_pair(text: str) -> tuple[int, int]:
    parts = [int(x.strip()) for x in text.split(",") if x.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("expected '<start>,<end>' (e.g. 2015,2022)")
    return parts[0], parts[1]


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Location-only KNN baseline (sklearn) for arrest probability"
    )
    parser.add_argument(
        "--k-candidates",
        type=_parse_int_csv,
        default=list(DEFAULT_K_CANDIDATES),
        help="Comma-separated candidate K values (default: 10,25,50,100,250)",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of stratified CV folds for tuning K (default: 5)",
    )
    parser.add_argument(
        "--train-years",
        type=_parse_year_pair,
        default=DEFAULT_TRAIN_YEARS,
        help="Train year range as 'start,end' (default: 2015,2022)",
    )
    parser.add_argument(
        "--test-years",
        type=_parse_year_pair,
        default=DEFAULT_TEST_YEARS,
        help="Test year range as 'start,end' (default: 2023,2024)",
    )
    parser.add_argument(
        "--crime-types",
        type=str,
        default="",
        help="Comma-separated subset of primary_type to evaluate (default: all)",
    )
    parser.add_argument(
        "--max-rows-per-type",
        type=int,
        default=0,
        help="Optional cap on training rows per crime type (0 = no cap)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed used for CV and any subsampling (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "knn_location_only",
        help="Output directory (default: outputs/knn_location_only)",
    )
    parser.add_argument(
        "--no-write-predictions",
        action="store_true",
        help="Skip writing per-crime-type prediction CSVs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)

    from src.preprocess_data import preprocess_data

    print("Loading + preprocessing data via src.preprocess_data.preprocess_data()…")
    df = preprocess_data()
    print(f"  rows after preprocessing: {len(df):,}")

    crime_types = (
        [s.strip() for s in args.crime_types.split(",") if s.strip()]
        if args.crime_types
        else None
    )

    results = evaluate_all_crime_types(
        df,
        train_years=tuple(args.train_years),
        test_years=tuple(args.test_years),
        k_candidates=args.k_candidates,
        cv_folds=args.cv_folds,
        random_state=args.random_state,
        crime_types=crime_types,
        max_rows_per_type=args.max_rows_per_type,
        progress=True,
    )

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = [r.metrics_dict() for r in results]
    metrics_payload = {
        "run_config": {
            "train_years": list(args.train_years),
            "test_years": list(args.test_years),
            "k_candidates": list(args.k_candidates),
            "cv_folds": int(args.cv_folds),
            "random_state": int(args.random_state),
            "max_rows_per_type": int(args.max_rows_per_type),
            "feature_cols": ["latitude", "longitude"],
            "target_col": "arrest",
            "model": "sklearn.neighbors.KNeighborsClassifier(weights='uniform')",
        },
        "per_crime_type": summary_rows,
    }
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )

    summary_df = pd.DataFrame(
        [
            {
                "crime_type": r.crime_type,
                "n_train": r.n_train,
                "n_test": r.n_test,
                "positive_rate_train": r.positive_rate_train,
                "positive_rate_test": r.positive_rate_test,
                "best_k": r.best_k,
                "test_log_loss": r.test_log_loss,
                "test_accuracy": r.test_accuracy,
                "skipped": r.skipped,
            }
            for r in results
        ]
    )
    summary_df.to_csv(out_dir / "summary.csv", index=False)

    if not args.no_write_predictions:
        for r in results:
            if r.predictions is None:
                continue
            r.predictions.to_csv(
                out_dir / f"predictions_{slugify(r.crime_type)}.csv", index=False
            )

    print(f"\nSaved metrics: {out_dir / 'metrics.json'}")
    print(f"Saved summary: {out_dir / 'summary.csv'}")
    print("\nPer-crime-type test results:")
    eval_df = summary_df[summary_df["skipped"].isna()].copy()
    if len(eval_df):
        eval_df_sorted = eval_df.sort_values("test_log_loss")
        print(
            eval_df_sorted[
                ["crime_type", "best_k", "test_log_loss", "test_accuracy"]
            ].to_string(index=False)
        )
    skipped_df = summary_df[~summary_df["skipped"].isna()]
    if len(skipped_df):
        print("\nSkipped crime types:")
        print(skipped_df[["crime_type", "skipped"]].to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
