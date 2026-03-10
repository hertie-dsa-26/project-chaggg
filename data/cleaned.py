  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e63836fc-ad9a-448b-b007-c2eedf38a18a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load the data\n",
    "#don't do it this way\n",
    "#df = pd.read_csv(\"data/chicago_crime_data/chicago_crimes_2001_2025.csv\", low_memory = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "0b82a78a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Basic info ────────────────────────────────────────────────────────────────\n",
    "print(df.describe())\n",
    "print(df.shape)\n",
    "print(df.dtypes)\n",
    "print(df.columns.tolist())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "43977c2a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Filter to 2001-2025 ───────────────────────────────────────────────────────\n",
    "df = df[df['year'].between(2001, 2025)]\n",
    "print(df['year'].value_counts().sort_index())\n",
    "print(\"\\nTotal records (2001-2025):\", len(df))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9cb9ed85",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Missing value summary ─────────────────────────────────────────────────────\n",
    "missing = df.isnull().sum()\n",
    "missing_pct = (missing / len(df) * 100).round(2)\n",
    "\n",
    "missing_summary = pd.DataFrame({\n",
    "    \"missing_count\": missing,\n",
    "    \"missing_pct\": missing_pct\n",
    "}).sort_values(\"missing_pct\", ascending=False)\n",
    "\n",
    "missing_summary"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e6264df1",
   "metadata": {},
   "source": [
    "## Column Alignment Check\n",
    "\n",
    "All 22 columns are present and correctly ordered — no misalignment detected.\n",
    "\n",
    "Missing values are due to **genuine data gaps**, not structural issues:\n",
    "\n",
    "| Column | Missing | Reason |\n",
    "|--------|---------|--------|\n",
    "| `ward` / `community_area` | ~614k | Early records do not include these fields |\n",
    "| `x/y_coordinate` / `latitude` / `longitude` / `location` | ~94k | Addresses that could not be geocoded |\n",
    "| `location_description` | ~15k | Small number of records with no venue description |\n",
    "| `district` | 47 | Negligible |"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "62fc29e0",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Data type checks ──────────────────────────────────────────────────────────\n",
    "# fbi_code and iucr are strings, which is correct - they are category codes\n",
    "print(df.dtypes)\n",
    "df.info()\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "cc9685ea",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Verify arrest and domestic are boolean\n",
    "print(\"arrest unique values:\", df[\"arrest\"].unique())\n",
    "print(\"domestic unique values:\", df[\"domestic\"].unique())\n",
    "#ok good"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6d136958",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Type conversions ──────────────────────────────────────────────────────────\n",
    "\n",
    "# date and updated_on should be datetime\n",
    "# errors='coerce' turns unparseable values into NaT rather than crashing\n",
    "df[\"date\"]       = pd.to_datetime(df[\"date\"], errors=\"coerce\")\n",
    "df[\"updated_on\"] = pd.to_datetime(df[\"updated_on\"], errors=\"coerce\")\n",
    "\n",
    "# district, ward, community_area should be integer\n",
    "# Int64 (capital I) is used because these columns have missing values\n",
    "# regular int64 cannot store NaN, Int64 can\n",
    "df[\"district\"]       = pd.to_numeric(df[\"district\"], errors=\"coerce\").astype(\"Int64\")\n",
    "df[\"ward\"]           = pd.to_numeric(df[\"ward\"], errors=\"coerce\").astype(\"Int64\")\n",
    "df[\"community_area\"] = pd.to_numeric(df[\"community_area\"], errors=\"coerce\").astype(\"Int64\")\n",
    "\n",
    "# Recode categorical text columns to save memory\n",
    "df[\"primary_type\"]        = df[\"primary_type\"].astype(\"category\")\n",
    "df[\"description\"]         = df[\"description\"].astype(\"category\")\n",
    "df[\"location_description\"] = df[\"location_description\"].astype(\"category\")\n",
    "\n",
    "# Verify missing values did not increase after type conversions\n",
    "missing = df.isnull().sum()\n",
    "missing_pct = (missing / len(df) * 100).round(2)\n",
    "missing_summary = pd.DataFrame({\n",
    "    \"missing_count\": missing,\n",
    "    \"missing_pct\": missing_pct\n",
    "}).sort_values(\"missing_pct\", ascending=False)\n",
    "missing_summary"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3ff33d07",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Remove coordinates outside Chicago ───────────────────────────────────────\n",
    "# Valid Chicago latitude range: 41.6 - 42.1\n",
    "# Valid Chicago longitude range: -87.9 - -87.5\n",
    "#df.loc[~df[\"latitude\"].between(41.6, 42.1), \"latitude\"] = None\n",
    "#df.loc[~df[\"longitude\"].between(-87.9, -87.5), \"longitude\"] = None\n",
    "\n",
    "#print(\"Latitude range:\", df[\"latitude\"].min(), \"-\", df[\"latitude\"].max())\n",
    "#print(\"Longitude range:\", df[\"longitude\"].min(), \"-\", df[\"longitude\"].max())\n",
    "#commented out incase we don't want ot do it"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e13960ef",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Missing data analysis ─────────────────────────────────────────────────────\n",
    "\n",
    "# Total rows with at least one missing value\n",
    "print(\"Rows with at least one missing value:\", df.isnull().any(axis=1).sum())\n",
    "print(\"That is\", round(df.isnull().any(axis=1).sum() / len(df) * 100, 2), \"% of the data\")\n",
    "\n",
    "# Which columns are driving the missing values\n",
    "print(\"\\nMissing values per column for incomplete rows:\")\n",
    "print(df[df.isnull().any(axis=1)][df.columns].isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c77dc428",
   "metadata": {},
   "outputs": [],
   "source": [
    "# ── Is missing data concentrated in certain years? ────────────────────────────\n",
    "df[\"has_missing\"] = df.isnull().any(axis=1)\n",
    "\n",
    "missing_by_year = df.groupby(\"year\")[\"has_missing\"].agg([\"sum\", \"count\"])\n",
    "missing_by_year[\"pct\"] = (missing_by_year[\"sum\"] / missing_by_year[\"count\"] * 100).round(2)\n",
    "missing_by_year.columns = [\"missing_count\", \"total_rows\", \"missing_pct\"]\n",
    "print(missing_by_year)\n",
    "# Note: 2001 has high missing data - worth considering in analysis\n",
    "\n",
    "# Which specific columns are missing by year\n",
    "df.groupby(\"year\")[[\"latitude\", \"longitude\", \"ward\", \"community_area\"]].apply(lambda x: x.isnull().sum())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "36b0d0f4",
   "metadata": {},
   "outputs": [],
   "source": [
    "#double checking for things that may be considered full but just empty text, seems fine\n",
    "# ── Check for empty strings masquerading as non-null ─────────────────────────\n",
    "for col in df.select_dtypes(include=\"object\").columns:\n",
    "    empty_count = (df[col].astype(str).str.strip() == \"\").sum()\n",
    "    nan_string  = (df[col].astype(str) == \"nan\").sum()\n",
    "    if empty_count > 0 or nan_string > 0:\n",
    "        print(col, \"- empty strings:\", empty_count, \"| 'nan' strings:\", nan_string)\n",
    "        "
   ]
  },
  {
   "cell_type": "markdown",
   "id": "500cd953",
   "metadata": {},
   "source": [
    "## Suggestions for Further Cleaning\n",
    "\n",
    "The following are suggestions worth discussing with the team before implementing:\n",
    "\n",
    "- **`location` column** — contains JSON-like strings, redundant with `latitude`/`longitude`, worth removing\n",
    "- **`updated_on`** — could be recoded as a binary `was_updated` True/False variable since many values are missing\n",
    "- **`date`** — worth splitting into separate `date` and `time` columns for easier feature engineering\n",
    "- **Missing data** — ~7% of rows have at least one missing value, concentrated in 2001. Worth discussing whether to drop these rows depending on which features the ML algorithm uses\n",
    "- **Early years (2001)** — high proportion of missing location data, likely due to geocoding not being available at the time"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": ".venv",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.14.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
