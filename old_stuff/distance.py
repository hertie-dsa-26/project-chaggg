import math

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on Earth.
    
    Parameters:
        lat1, lon1: Latitude and longitude of point 1 (in decimal degrees)
        lat2, lon2: Latitude and longitude of point 2 (in decimal degrees)
    
    Returns:
        Distance in kilometers
    """
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

    # Differences
    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    R = 6371  # Earth's radius in kilometers
    return R * c


def get_spatial_bounds(df):
    """
    Calculate spatial bounds and maximum distance for normalization.
    
    Parameters:
        df: DataFrame with 'latitude' and 'longitude' columns
    
    Returns:
        dict with 'max_distance' (km) and 'bounds' (lat/lon ranges)
    """
    max_lat, min_lat = df['latitude'].max(), df['latitude'].min()
    max_lon, min_lon = df['longitude'].max(), df['longitude'].min()
    max_distance = haversine(min_lat, min_lon, max_lat, max_lon)
    
    return {
        'max_distance': max_distance,
        'bounds': {
            'lat': (min_lat, max_lat),
            'lon': (min_lon, max_lon)
        }
    }

def cyclical_distance(value1, value2, max_value):
    """
    Calculate distance between two cyclical values using sine-cosine encoding.
    
    Parameters:
        value1, value2: The cyclical values (e.g., hour=23, hour=1)
        max_value: Maximum value in the cycle (e.g., 24 for hours, 7 for day_of_week)
    
    Returns:
        Distance between 0 and 1
    """
    # Convert to angles on unit circle
    angle1 = 2 * math.pi * value1 / max_value
    angle2 = 2 * math.pi * value2 / max_value
    
    # Compute sin/cos coordinates
    sin1, cos1 = math.sin(angle1), math.cos(angle1)
    sin2, cos2 = math.sin(angle2), math.cos(angle2)
    
    # Euclidean distance on unit circle
    distance = math.sqrt((sin1 - sin2)**2 + (cos1 - cos2)**2)
    
    # Normalize to [0, 1] (max distance on unit circle is 2)
    return distance / 2.0

def temporal_distance(month1, hour1, dow1, month2, hour2, dow2):
    """
    Calculate normalized temporal distance between two crimes.
    Uses cyclical encoding for month, hour, and day_of_week.
    
    Parameters:
        month1, hour1, dow1: Time components of crime 1
        month2, hour2, dow2: Time components of crime 2
        (month: 1-12, dow = day_of_week: 0=Monday, 6=Sunday)
    
    Returns:
        Temporal distance score between 0 and 1
    """
    # Cyclical features
    month_dist = cyclical_distance(month1, month2, 12)     # months range 1-12
    hour_dist = cyclical_distance(hour1, hour2, 24)        # hours range 0-23
    dow_dist = cyclical_distance(dow1, dow2, 7)            # day_of_week range 0-6

    # Average the four normalized differences
    return (month_dist + hour_dist + dow_dist) / 3

def combined_distance(crime1, crime2, max_distance, alpha=0.5, beta=0.5):
    """
    Combined spatial and temporal distance between two crimes.
    
    Parameters:
        crime1, crime2: dicts or rows with keys 'latitude', 'longitude', 
                        'month', 'hour', 'day_of_week'
        alpha: weight for spatial distance
        beta: weight for temporal distance
    
    Returns:
        Combined distance score
    """
    spatial = haversine(crime1['latitude'], crime1['longitude'], 
                        crime2['latitude'], crime2['longitude'])
    temporal = temporal_distance(crime1['month'], crime1['hour'], crime1['day_of_week'],
                                 crime2['month'], crime2['hour'], crime2['day_of_week'])
    
    # Normalise spatial to [0, 1] scale using max_distance in dataset
    spatial_normalized = spatial / max_distance 
    
    return alpha * spatial_normalized + beta * temporal