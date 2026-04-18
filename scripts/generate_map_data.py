import csv
import json
import random

points_by_year = {}
stats_by_year = {}

with open("./data/cleaned/chicago_crimes_cleaned.csv", "r") as f:
    reader = csv.DictReader(f)
    for row in reader:
        year = row['year']

        if year < '2002' or year > '2025':
            continue

        lat = row['latitude']
        lon = row['longitude']
        crime_type = row['primary_type']
        area = row['community_area']

        if year not in points_by_year:
            points_by_year[year] = []
        if year not in stats_by_year:
            stats_by_year[year] = {}

        if lat and lon:
            points_by_year[year].append([float(lat), float(lon), crime_type])

        if area:
            if area not in stats_by_year[year]:
                stats_by_year[year][area] = {'total': 0, 'types': {}}
            stats_by_year[year][area]['total'] += 1
            stats_by_year[year][area]['types'][crime_type] = stats_by_year[year][area]['types'].get(crime_type, 0) + 1

sampled = {}
community_stats = {}

for year in sorted(points_by_year.keys()):
    pts = points_by_year[year]
    sampled[year] = random.sample(pts, min(20000, len(pts)))
    print(f"{year}: {len(pts)} points -> sampled {len(sampled[year])}")

    community_stats[year] = {}
    if year in stats_by_year:
        for area, s in stats_by_year[year].items():
            top3 = sorted(s['types'].items(), key=lambda x: x[1], reverse=True)[:3]
            community_stats[year][area] = {
                'total': s['total'],
                'top3': [[t, c] for t, c in top3]
            }

with open("./src/flask_app/static/data/space/sampled_points_by_year.json", "w") as f:
    json.dump(sampled, f)

with open("./src/flask_app/static/data/space/community_stats_by_year.json", "w") as f:
    json.dump(community_stats, f)

print(f"\nYears: {len(sampled)}")
print("Files saved to ./src/flask_app/static/data/space/")