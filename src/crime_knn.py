"""CLI demo for KNN Logistic Ridge Regression arrest prediction.

Thin wrapper around algorithms.knn_lrr.predict_arrest_probability.
Loads precomputed artifacts and prompts the user for inputs.

Run with:
    uv run python -m src.crime_knn
"""
import sys, os
import numpy as np
import pyinputplus as pyip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from algorithms.knn_lrr import predict_arrest_probability
from src.flask_app.load_crime_artifacts import load_knn_arrays


def main():
    print("=" * 60)
    print(" K-nearest-neighbours Logistic Ridge Regression")
    print("=" * 60)

    artifacts = load_knn_arrays()
    slug = pyip.inputMenu(sorted(artifacts.keys()), lettered=False, numbered=True)
    artifact = artifacts[slug]
    print(f"\nFilter applied: {slug}")
    print(f"{artifact['label'].shape[0]} observations available.")

    k = pyip.inputInt("Enter number of neighbours: ", min=1, max=100)
    print(f"Finding {k} nearest neighbours...")
    print("Running Logistic Ridge Regression.")

    # Demo query — a real Chicago lat/lon, midnight on a Sunday in late March
    example = np.array([
        41.840450785499996, -87.664206366,
        -0.2588190451025215, 0.2588190451025203,
        0.0, -0.2225209339563143,
        -2.4492935982947064e-16, -1.8369701987210294e-16,
        -0.1372787721132651, -0.0473213883224323,
    ])

    result = predict_arrest_probability(artifact, example, k, return_details=True)
    p = result["probability"]
    beta = result["beta"]

    print(f"\nIntercept   : {beta[0]:.6f}")
    print(f"Coefficients: {beta[1:]}")
    print(f"\nPredicted probability of arrest for query: {p * 100:.2f}%")

    if p <= 0.49:
        print("Most likely scenario: will not get arrested.")
    elif p < 0.51:
        print("Most likely scenario: toss a (fair) coin.")
    else:
        print("Most likely scenario: will get arrested.")


if __name__ == "__main__":
    main()