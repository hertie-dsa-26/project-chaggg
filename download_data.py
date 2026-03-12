#this script automates the data download from google drive and creates a local data folder with the raw and cleaned data

import os
import gdown

os.makedirs("data/raw", exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# Raw data
RAW_FILE_ID = "1HGIQVus5LLVN8ONsKPhN6pX1ViHQdrdJ"
RAW_OUTPUT = "data/raw/chicago_crimes_2001_2025_raw.csv"

if not os.path.exists(RAW_OUTPUT):
    print("Downloading raw dataset...")
    gdown.download(f"https://drive.google.com/uc?id={RAW_FILE_ID}", RAW_OUTPUT, quiet=False)
    print("Done.")
else:
    print("Raw dataset already exists, skipping.")

# Processed data
#PROCESSED_FILE_ID = "your_processed_file_id_here"
#PROCESSED_OUTPUT = "data/chicago_crimes_2001_2025_cleaned.csv"

#if not os.path.exists(PROCESSED_OUTPUT):
    #print("Downloading processed dataset...")
    #gdown.download(f"https://drive.google.com/uc?id={PROCESSED_FILE_ID}", PROCESSED_OUTPUT, quiet=False)
    #print("Done.")
#else:
    #print("Processed dataset already exists, skipping.")
