# Chicago Crime Analysis Project

This repository is for a group project for the Data Structures and Algorithms Course taught at the Hertie School in Berlin. 

## Project Overview

Working in a team of 7 students, we are designing and developing a comprehensive **web application using Flask** that performs end-to-end analysis of a dataset.

## Dataset

We are using the City of Chicago's crime dataset, which provides comprehensive information on reported incidents from 2001 to present. We are using all data up to 31 December 2025.

[Crimes - 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

## Dataset Setup

The datasets are not stored in this repository. To download them, follow these steps in your terminal:
1. git clone <repo>
2. uv sync
3. python download_data.py

### Parquet distribution (recommended)

To ensure everyone uses the **same snapshot** (up to 31 Dec 2025) and to speed up loading, we distribute a cleaned **Parquet** file via the shared Google Drive.

- **Download raw CSV snapshot**

```bash
python download_data.py
```

- **Download cleaned Parquet snapshot** (once the Drive file id is available)

```bash
export CHAGGG_CLEANED_PARQUET_FILE_ID="<drive-file-id>"
python download_data.py --skip-raw
```

### Creating the Parquet file (for maintainers)

Once `data/cleaned/chicago_crimes_cleaned.csv` exists locally:

```bash
python scripts/convert_cleaned_csv_to_parquet.py
```

Then upload `data/cleaned/chicago_crimes_cleaned.parquet` to the shared Drive and share the **file id** (to be used as `CHAGGG_CLEANED_PARQUET_FILE_ID`).

## Development Workflow

To maintain code quality and ensure collaboration, please follow this workflow:

1. **Create a branch** – Create a new branch either:
   - Locally in your IDE, or
   - On GitHub (then switch to it in your IDE)
2. **Develop locally** – Write and test code in your personal notebooks until you're confident it works
3. **Integrate code** – Copy your working code into the relevant script
4. **Commit changes** – Commit your changes to this branch with a descriptive commit message that everyone understand
6. **Open a pull request** – Create a pull request with a brief description of your changes
7. **Request review** – Tag a team member for code review
8. **Merge** – After approval, the pull request can be merged into `main`

### Important Guidelines

⚠️ **DO NOT commit directly to `main`**

✅ **ALL CODE must be reviewed by a second person before merging**


[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D69TCBIW)
