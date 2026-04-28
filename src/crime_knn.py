import pandas as pd
import numpy as np
import sys, os
import pyinputplus as pyip

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))) # so max_heap.py is importable
from max_heap import MaxHeap, euclidean_distance

from src.preprocess_data import preprocess_data

# --- Primary Type filter:
 
def select_crime(crime: str = None):

    df = preprocess_data()
    valid_types = df['primary_type'].unique()

    if crime is None:
        choices = valid_types.tolist()
        response = pyip.inputMenu(choices)
        # pyip.inputMenu already guarantees a valid choice, no check needed
    else:
        if crime not in valid_types:
            raise ValueError(f"Primary type '{crime}' not included in catalogue.")
        response = crime

    df = df[df['primary_type'] == response]
    df = df.reset_index(drop=True)

    print('Filter applied.')
    print(f'{df.shape[0]} observations found for selected primary crime type.')

    return df

# --- KNN Logistic Ridge Regression:

def knn_lrr(query: list, crime_type: str = None) -> list:

    crime = select_crime(crime = crime_type)

    features = crime['latitude', 'longitude', 'hour_sin', 'hour_cos',
                     'day_of_week_sin', 'day_of_week_cos', 'month_sin',
                     'month_cos', 'day_of_year_sin', 'day_of_year_cos']
                      
    label = crime['arrest']

    # Standardize data before knn

    features = features.to_numpy().astype(float)
    label = label.to_numpy().astype(float)

    # 1. Standardize features (fit on X_local, apply same transform to query)
    features_mean = X.mean(axis=0)
    features_std  = X.std(axis=0) + 1e-8
    features_scaled = (X - features_mean) / features_std

    # 2. Augment with intercept column
    n, p = features_scaled.shape
    features_aug = np.hstack([np.ones((n, 1)), features_scaled]) 

    # 3. Scale and augment the query point
    query_arr    = np.array(query, dtype=float)
    query_scaled = (query_arr - features_mean) / features_std
    query_aug    = np.concatenate([[1.0], query_scaled])

    # Find k nearest neighbours

    k = pyip.inputInt('Enter number of neighbours: ', min = 10, max = 100)

    heap = MaxHeap(capacity= k)

    for idx, row in features_aug.iterrows():
        dist = euclidean_distance(row.tolist(), query_aug.tolist())
        heap.add(dist, float(idx))
        neighbour_indices = [int(target) for _, target in heap.get_all()]

    print(f'{k} nearest neighbours found.')
    print('Running Logistic Ridge Regression.')

    X_local = features_aug.iloc[neighbour_indices,:]
    y_local = label.loc[neighbour_indices]

    # --- Logistic Ridge Regression ---
    alpha  = 1.0    # L2 regularization strength
    lr     = 0.1    # learning rate
    n_iter = 1000   # gradient descent iterations

    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    beta = np.zeros(X_local.shape[1])

    for _ in range(n_iter):
        p_hat = sigmoid(X_local @ beta)                    # predicted probabilities
        grad  = (X_local.T @ (p_hat - y_local)) / n             # cross-entropy gradient
        grad[1:] += alpha * beta[1:]                     # L2 penalty (skip intercept)
        beta -= lr * grad

    # 4. Predict on query
    y_hat_logit = float(sigmoid(query_aug @ beta))

    print(f"Intercept   : {beta[0]:.6f}")
    print(f"Coefficients: {beta[1:]}")
    print(f"\nPredicted probability of arrest for query: {y_hat_logit*100:.2f}%")

    if y_hat_logit <= 0.49:
        print('Most likely scenario: will not get arrested.')
    elif y_hat_logit == 0.5:
        print('Most likely scenario: toss a (fair) coin.')
    else:
        print('Most likely scenario: will get arrested.')

    return y_hat_logit

def main():
    print("="*60)
    print(' K-nearest-neigbours Logistic Ridge Regression')
    print("="*60)

    example = [np.float64(41.840450785499996), np.float64(-87.664206366),
               np.float64(-0.2588190451025215), np.float64(0.2588190451025203), 
               np.float64(0.0), np.float64(-0.2225209339563143), np.float64(-2.4492935982947064e-16), 
               np.float64(-1.8369701987210294e-16), np.float64(-0.1372787721132651), np.float64(-0.0473213883224323)]

    p_arrested = knn_lrr(query = example)

if __name__ == "__main__":
    main()