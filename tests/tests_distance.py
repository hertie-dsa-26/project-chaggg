import sys, os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'algorithms'))

from distance import haversine, cyclical_distance, temporal_distance, combined_distance

#distance between same points in Chicago is 0km
def test_haversine_identical_points():
    assert haversine(41.8781, -87.6298, 41.8781, -87.6298) == 0

#distance between Berlin and Chicago is ~ 7083.44km
def test_haversine_far_points():
    assert haversine(52.5200, 13.4050, 41.8781, -87.6298) > 7000

#cyclical distance between 23 and 0 hours should be small
def test_cyclical_distance_close_values_hour():
    assert cyclical_distance(23, 0, 24) < 0.15

#cyclical distance between 0 and 12 hours should be 1, furthest apart in 24 cycle
def test_cyclical_distance_far_values_hour():
    assert cyclical_distance(0, 12, 24) == 1

#cyclical distance between December and January should be small
def test_cyclical_distance_close_values_month():
    assert cyclical_distance(1, 12, 12) < 0.26

#cyclical distance between January and June should be large
def test_cyclical_distance_far_values_month():
    assert cyclical_distance(1, 6, 12) > 0.95

#cyclical distance between Sunday and Monday should be less than 0.5
def test_cyclical_distance_close_values_dow():
    assert cyclical_distance(0, 6, 7) < 0.5

#cyclical distance between Thursday and Monday should be more than 0.95
def test_cyclical_distance_far_values_dow():
    assert cyclical_distance(4, 1, 7) > 0.95

#temporal distance between identical time components should be 0
def test_temporal_distance_equal():
    assert temporal_distance(12, 23, 6, 12, 23, 6) == 0

#temporal distance between very different time components should be close to 1
def test_temporal_distance_far():
    assert temporal_distance(1, 0, 0, 6, 12, 4) > 0.75

#combined distance between identical crimes should be 0
def test_combined_distance_identical():
    crime1 = {'latitude': 41.8781, 'longitude': -87.6298, 'month': 12, 'hour': 23, 'day_of_week': 6}
    crime2 = {'latitude': 41.8781, 'longitude': -87.6298, 'month': 12, 'hour': 23, 'day_of_week': 6}
    max_distance = 54  # rounded max distance in dataset for normalisation
    assert combined_distance(crime1, crime2, max_distance, alpha=0.5, beta=0.5) == 0

#combined distance with alpha 1 and beta 0 should equal distance alone
def test_combined_distance_spatial():
    crime1 = {'latitude': 52.5200, 'longitude': 13.4050, 'month': 12, 'hour': 23, 'day_of_week': 6}
    crime2 = {'latitude': 41.8781, 'longitude': -87.6298, 'month': 12, 'hour': 23, 'day_of_week': 6}
    max_distance = 54  # rounded max distance in dataset for normalisation
    assert combined_distance(crime1, crime2, max_distance, alpha=1, beta=0) == pytest.approx(haversine(52.5200, 13.4050, 41.8781, -87.6298)/54)