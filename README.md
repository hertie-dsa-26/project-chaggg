# Chicago Crime Analysis Project

This repository is for a group project for the Data Structures and Algorithms Course taught at the Hertie School in Berlin. 

## Project Overview

Working in a team of 7 students, we are designing and developing a comprehensive **web application using Flask** that performs end-to-end analysis of a dataset.

## Dataset

We are using the City of Chicago's crime dataset, which provides comprehensive information on reported incidents from 2001 to present. We are using all data from 01 January 2002 up to 31 December 2025.

[Crimes - 2001 to Present](https://data.cityofchicago.org/Public-Safety/Crimes-2001-to-Present/ijzp-q8t2/about_data)

## Getting Started

### Prerequisites
- Python 3.11+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

### 1. Clone the repository

```
git clone https://github.com/hertie-dsa-26/project-chaggg.git
cd project-chaggg
````

### 2. Install dependencies

```
uv sync
```

### 3. Set up the data

Choose one of the two options below.

---

#### Option A — Download the data folder (recommended)

Download the `data/` folder from [Google Drive](<https://drive.google.com/drive/folders/1hVUJNurriQMhopf3Iv9zbpvWWFJcHixw?usp=sharing>) and place it at the
project root so the structure looks like:

```
project-chaggg/
└── data/
    ├── raw/
    ├── cleaned/
    └── precomputed/
```

The data folder only contains the cleaned parquet file and the Primary Crime Type Data Bundles. 
After successful download skip straight to **Step 4**.

---

#### Option B — Run the full data pipeline

This will download the raw data from the Chicago Data Portal, clean it, and
generate all precomputed artifacts. It may take several minutes.

```
uv run python -m scripts.main
```

Follow the prompts in the terminal.

---

### 4. Run the app

```
uv run run_app.py 
```

Then open http://127.0.0.1:5000 in your browser.

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-22041afd0340ce965d47ae6ef1cefeee28c7c493a6346c4f15d667ab976d596c.svg)](https://classroom.github.com/a/D69TCBIW)
