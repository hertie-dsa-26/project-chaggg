import pandas as pd
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('..')) 
from scripts.utils import load_data

# --- Preprocess data

def preprocess_data():

    df = load_data()
    # -- Before collapsing: count people involved per incident
    people_per_incident = df.groupby('case_number').size().rename('people_involved')

    # -- Collapse: one row per incident
    # For most columns, take the first value (they should be the same across rows of the same incident)
    # For 'arrest', take max — True if ANY person was arrested (your binary variable)
    df = df.groupby('case_number').agg(
    id=('id', 'first'),
    date=('date', 'first'),
    block=('block', 'first'),
    iucr=('iucr', 'first'),
    primary_type=('primary_type', 'first'),
    description=('description', 'first'),
    location_description=('location_description', 'first'),
    domestic=('domestic', 'first'),
    beat=('beat', 'first'),
    district=('district', 'first'),
    ward=('ward', 'first'),
    community_area=('community_area', 'first'),
    fbi_code = ('fbi_code', 'first'),
    year=('year', 'first'),
    updated_on=('updated_on', 'first'),
    x_coordinate=('x_coordinate', 'first'),
    y_coordinate=('y_coordinate', 'first'),
    latitude=('latitude', 'first'),
    longitude=('longitude', 'first'),
    time=('time', 'first'),
    month=('month', 'first'),
    day=('day', 'first'),
    day_of_week=('day_of_week', 'first'),
    day_of_year=('day_of_year', 'first'),
    hour=('hour', 'first'),
    minute=('minute', 'first'),
    hour_sin=('hour_sin', 'first'),
    hour_cos=('hour_cos', 'first'),
    day_of_week_sin=('day_of_week_sin', 'first'),
    day_of_week_cos=('day_of_week_cos', 'first'),
    month_sin=('month_sin', 'first'),
    month_cos=('month_cos', 'first'),
    day_of_year_sin=('day_of_year_sin', 'first'),
    day_of_year_cos=('day_of_year_cos', 'first'),
    at_least_one_arrested=('arrest', 'max')   # True if any row had arrest=True
    ).reset_index()

    # -- Attach people_involved

    df = df.merge(people_per_incident, on='case_number')

    # -- Final cleaning decisions

    df = df[df['people_involved'] == 1]  # Drop any incident with more than one person involved
    df['arrest'] = df['at_least_one_arrested'].astype(int) # Encode target!
    df = df.groupby('primary_type').filter(lambda x: len(x) >= 5000) # Cut off every 'primary_type' with < 5k rows

    return df