import src.preprocess_data
import sys, os

sys.path.insert(0, os.path.abspath('..'))   # so max_heap.py is importable
from max_heap import MaxHeap, euclidean_distance


def select_crime(crime_type:str):

    incident_df = preprocess_data()

    if crime_type not in incident_df['primary_type'].unique():
        ValueError('Primary type not included in catalogue.')

    crime_df = incident_df[incident_df['primary_type'] == f'{crime_type}']
    crime_df = crime_df.reset_index(drop=True)

    return crime_df

def find_knn(k:int, crime_type: str, query:list) -> list:

    crime = select_crime(crime_type)

    features = crime[['latitude', 'longitude', 'day_of_week', 'day_of_year', \
                      'hour_sin', 'hour_cos', 'day_of_week_sin', 'day_of_week_cos',\
                      'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos']]
    label = crime['arrest']

    heap = MaxHeap(capacity= k)

    for idx, row in features.iterrows():
        dist = euclidean_distance(row.tolist(), query)
        heap.add(dist, float(idx))
        neighbour_indices = [int(target) for _, target in heap.get_all()]

    return neighbour_indices

