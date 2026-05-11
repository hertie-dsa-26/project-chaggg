# Chicago Crime Data Cleaning Documentation

## Overview
This document records all data cleaning decisions made in the `clean.py` pipeline for the Chicago crime dataset (2001-2025).

## Data Cleaning Steps

### 1. Year Filtering
**Decision:** Filter to years 2001-2025 (defined in `YEAR_RANGE`)  
**Rationale:** Focus on modern crime patterns; older data may have different reporting standards  
**Impact:** Removes records outside this range

### 2. Date/Time Parsing
**Decision:** Split original `date` column into separate `date` and `time` columns  
**Rationale:** Enables separate temporal analysis; reduces redundancy  
**Implementation:**
- `date`: Converted to `datetime64` (YYYY-MM-DD format)
- `time`: Converted to Python `time` objects (HH:MM:SS format)

### 3. Temporal Feature Extraction
**Decision:** Extract year, month, day, hour, and day_of_week from date/time  
**Rationale:** Enable time-based analysis (seasonal patterns, hourly trends, weekday vs weekend)  
**Features created:**
- `year`: Integer (e.g., 2023)
- `month`: Integer 1-12
- `day`: Integer 1-31
- `hour`: Integer 0-23
- `day_of_week`: Integer 0-6 (0=Monday, 6=Sunday)

**Decision:** Drop original `date` and `time` columns after extraction  
**Rationale:** Extracted features provide all needed temporal information; reduces memory usage

### 4. Coordinate Cleaning
**Decision:** Remove rows where longitude OR latitude is NaN  
**Rationale:** Location is critical for spatial analysis; rows without coordinates cannot be mapped  
**Impact:** Geographic analysis requires complete coordinate pairs  
**Validation:** Checked that remaining coordinates fall within valid Chicago bounds:
- Latitude: 41.6-42.1°N
- Longitude: -87.9 to -87.5°W

### 5. Type Conversions
**Decision:** Convert district, ward, community_area to nullable Int64  
**Rationale:** These are administrative IDs; Int64 allows NaN values (some crimes occur outside defined areas)

**Decision:** Convert primary_type, description, location_description to categorical  
**Rationale:** Repeated string values; categorical encoding reduces memory by ~50-70%

### 6. Column Removal
**Decision:** Drop `location` column  
**Rationale:** Duplicates information already in x/y coordinate columns; redundant

## Data Quality Checks
- Missing value summary printed for all columns
- Coordinate validation against Chicago geographic bounds
- Row counts reported at each filtering step
- Memory usage tracked

## Not Cleaned (Intentional)
- **Out-of-range coordinates:** Flagged but not removed; may represent data entry errors or crimes near Chicago boundaries requiring domain expertise
- **Missing district/ward/community_area:** Kept as NaN; legitimate for crimes in parks, highways, or boundary areas
- **Case numbers, IUCR codes:** Kept as-is; assumed valid identifiers

## Output
**File:** `data/cleaned/chicago_crimes_cleaned.csv`  
**Format:** CSV with headers  
**Encoding:** UTF-8  
**Index:** Not saved (index=False)

---
**Last Updated:** 2026-04-08  
**Script Version:** clean.py v1.0