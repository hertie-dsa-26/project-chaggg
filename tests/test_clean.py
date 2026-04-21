import sys, os
import pytest
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from clean import add_cyclical_time_features

def test_cyclical_encoding_hour_zero():
    df = pd.DataFrame({'hour': [0], 'day_of_week': [0], 'month': [1], 'day_of_year': [1]})
    df = add_cyclical_time_features(df)
    assert df['hour_sin'].iloc[0] == pytest.approx(0.0, abs=1e-9)
    assert df['hour_cos'].iloc[0] == pytest.approx(1.0, abs=1e-9)