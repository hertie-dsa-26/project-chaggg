## Load packages and cleaned data
import pandas as pd
import matplotlib.pyplot as plt

import os
os.chdir('/Users/gracewang/Documents/Hertie/Data Structures & Algorithms/project-chaggg')

# ── Load cleaned data ─────────────────────────────────────────────────────────
df = pd.read_csv('data/cleaned/chicago_crimes_cleaned.csv', low_memory=False)

# district was saved as Int64 — restore it
df["district"] = pd.to_numeric(df["district"], errors="coerce").astype("Int64")

# Drop rows missing district or arrest
df = df.dropna(subset=["district", "arrest"])


# 1) Plot percentage of each crime type labeled domestic violence

# ── Standardize inconsistent primary_type labels ──────────────────────────────
type_remap = {
    "CRIM SEXUAL ASSAULT": "CRIMINAL SEXUAL ASSAULT",
}

df["primary_type"] = df["primary_type"].replace(type_remap)

# ── Calculate domestic rate by crime type ────────────────────────────────────
domestic_summary = (
    df.groupby("primary_type")
    .agg(
        total_count=("domestic", "count"),
        domestic_count=("domestic", "sum")
    )
    .assign(domestic_rate=lambda x: (x["domestic_count"] / x["total_count"] * 100).round(2))
    .sort_values("domestic_rate", ascending=False)
    .reset_index()
)

domestic_summary = domestic_summary[domestic_summary["total_count"] >= 1000]
print(domestic_summary)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.barh(
    domestic_summary["primary_type"],
    domestic_summary["domestic_rate"],
    color="steelblue",
    edgecolor="white",
    linewidth=0.5
)

# Annotate each bar with its percentage
for bar, val in zip(bars, domestic_summary["domestic_rate"]):
    ax.text(
        bar.get_width() + 0.3,
        bar.get_y() + bar.get_height() / 2,
        f'{val:.1f}%',
        ha="left", va="center", fontsize=8
    )

ax.set_xlabel("% Marked as Domestic", fontsize=12)
ax.set_ylabel("Crime Type", fontsize=12)
ax.set_title("% of Crimes Marked as Domestic Violence by Crime Type (2001–2025)", fontsize=14, fontweight="bold")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
ax.invert_yaxis()  # highest % at the top
ax.grid(axis="x", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# 2) Table summarizing arrest rate by crime type

# ── Calculate arrest rate by crime type ──────────────────────────────────────
arrest_summary = (
    df.groupby("primary_type")
    .agg(
        total_count=("arrest", "count"),
        arrest_count=("arrest", "sum")
    )
    .assign(arrest_rate=lambda x: (x["arrest_count"] / x["total_count"] * 100).round(2))
    .sort_values("total_count", ascending=False)
    .reset_index()
)

# ── Rename columns for display ────────────────────────────────────────────────
arrest_summary.columns = ["Crime Type", "Total Count", "Arrest Count", "Arrest Rate %"]

# ── Format columns ────────────────────────────────────────────────────────────
arrest_summary["Total Count"]   = arrest_summary["Total Count"].apply(lambda x: f'{x:,}')
arrest_summary["Arrest Count"]  = arrest_summary["Arrest Count"].apply(lambda x: f'{x:,}')
arrest_summary["Arrest Rate %"] = arrest_summary["Arrest Rate %"].apply(lambda x: f'{x:.2f}%')

print(arrest_summary.to_string(index=False))


# 3) Top 10 crime types by frequency

# ── Count crimes by primary type, sorted descending ──────────────────────────
type_counts = (
    df.groupby("primary_type")
    .size()
    .reset_index(name="incident_count")
    .sort_values("incident_count", ascending=False)
    .head(10)
)

print(type_counts)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 7))

bars = ax.bar(
    type_counts["primary_type"],
    type_counts["incident_count"],
    color="steelblue",
    edgecolor="white",
    linewidth=0.5
)

# Annotate each bar with its count
for bar, val in zip(bars, type_counts["incident_count"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f'{int(val):,}',
        ha="center", va="bottom", fontsize=6
    )

ax.set_xlabel("Crime Type", fontsize=12)
ax.set_ylabel("Number of Incidents", fontsize=12)
ax.set_title("Top 10 Crime Types by Frequency in Chicago (2001–2025)", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()


# 4) Plot avg crimes per month

#  ── Extract month from date ───────────────────────────────────────────────────
df["date"] = pd.to_datetime(df["date"])
df["month"] = df["date"].dt.month

# ── Count crimes per year-month, then average across years ───────────────────
monthly_avg = (
    df.groupby(["year", "month"])
    .size()
    .reset_index(name="incident_count")
    .groupby("month")["incident_count"]
    .mean()
    .reset_index(name="avg_incidents")
)

# ── Replace month numbers with names ─────────────────────────────────────────
month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
monthly_avg["month_name"] = monthly_avg["month"].apply(lambda x: month_names[x - 1])

print(monthly_avg)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

bars = ax.bar(
    monthly_avg["month_name"],
    monthly_avg["avg_incidents"],
    color="steelblue",
    edgecolor="white",
    linewidth=0.5
)

# Annotate each bar with its value
for bar, val in zip(bars, monthly_avg["avg_incidents"]):
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 100,
        f'{int(val):,}',
        ha="center", va="bottom", fontsize=8
    )

ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Average Number of Incidents", fontsize=12)
ax.set_title("Average Crime Count by Month in Chicago (2001–2025)", fontsize=14, fontweight="bold")
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()


# 5) Plot crime count by year

# ── Group by year and count incidents ────────────────────────────────────────
yearly_counts = (
    df.groupby("year")
    .size()
    .reset_index(name="incident_count")
    .sort_values("year")
)

print(yearly_counts)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    yearly_counts["year"],
    yearly_counts["incident_count"],
    color="steelblue",
    linewidth=2,
    marker="o",
    markersize=5
)

# Annotate each point with its count
for _, row in yearly_counts.iterrows():
    ax.text(
        row["year"],
        row["incident_count"] + 1000,
        f'{int(row["incident_count"]):,}',
        ha="center", va="bottom", fontsize=7
    )

ax.set_xlabel("Year", fontsize=12)
ax.set_ylabel("Number of Incidents", fontsize=12)
ax.set_title("Total Crime Incidents in Chicago by Year (2001–2025)", fontsize=14, fontweight="bold")
ax.set_xticks(yearly_counts["year"])
plt.xticks(rotation=45)
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{int(x):,}'))
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()