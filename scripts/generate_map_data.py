import json
import random
from src.flask_app.data import load_crime_data

df = load_crime_data()
df = df[df["year"].between(2002, 2025)]

# --- sampled points (20k per year) ---
valid = df[df["latitude"].notna() & df["longitude"].notna()].copy()
sampled = {}
for year, group in valid.groupby("year"):
    pts = group[["latitude", "longitude", "primary_type"]].values.tolist()
    sampled[str(int(year))] = random.sample(pts, min(20000, len(pts)))
    print(f"{int(year)}: {len(pts)} points -> sampled {len(sampled[str(int(year))])}")

# --- community stats per year ---
community_stats = {}
for year, yg in df.groupby("year"):
    year_key = str(int(year))
    community_stats[year_key] = {}
    for area, ag in yg.groupby("community_area"):
        area_key = str(int(area))
        type_counts = ag["primary_type"].value_counts()
        top3 = [[t, int(c)] for t, c in type_counts.head(3).items()]
        community_stats[year_key][area_key] = {"total": int(len(ag)), "top3": top3}

# --- save ---
with open("./src/flask_app/static/data/space/sampled_points_by_year.json", "w") as f:
    json.dump(sampled, f)

with open("./src/flask_app/static/data/space/community_stats_by_year.json", "w") as f:
    json.dump(community_stats, f)

print(f"\nYears: {len(sampled)}")
print("Files saved to ./src/flask_app/static/data/space/")