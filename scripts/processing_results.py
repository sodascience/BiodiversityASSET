# scripts/processing_results.py

import pandas as pd
from pathlib import Path
import re

"""
This script goes via all the CSV files in the assetization_features_scoring directory,
extracts the investor name from the first column, and saves a combined CSV with
an additional 'investor_name' column.
It also adds a 'batch_id' and 'source_csv_file' for tracking.
"""

#BASE_DIR = Path(r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\data\processed\assetization_features_scoring")
#OUTPUT_CSV = BASE_DIR / "combined_assetization_features_scoring.csv"


BASE_DIR = Path(r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\data\processed\investment_activity_classification")
OUTPUT_CSV = BASE_DIR / "combined_investment_activity_classification.csv"
# scripts/collect_assetization_csvs.py


def extract_investor(text):
    if pd.isna(text):
        return None
    # Match FfBF_*, Nature100_*, or UNPRI_* and grab what's after until the next "_"
    m = re.match(r"^(FfBF_|Nature100_|UNPRI_)([^_]+)_", str(text))
    return m.group(2) if m else None

def main():
    dfs = []
    total_files = 0

    for batch_dir in sorted([d for d in BASE_DIR.iterdir() if d.is_dir()]):
        batch_id = batch_dir.name
        csv_files = sorted(batch_dir.glob("*.csv"))
        for csv_path in csv_files:
            try:
                df = pd.read_csv(csv_path)
                if df.shape[1] < 2:
                    print(f"⚠️ Skipping {csv_path} — less than 2 columns.")
                    continue
                investor_names = df.iloc[:, 0].apply(extract_investor)
                df.insert(1, "investor_name", investor_names)
                df["batch_id"] = batch_id
                df["source_csv_file"] = csv_path.name
                dfs.append(df)
                total_files += 1
            except Exception as e:
                print(f"⚠️ Skipping {csv_path} ({e})")

    if dfs:
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
        print(f"✅ Combined {total_files} CSVs into {OUTPUT_CSV}")
        print(f"📊 Total rows in combined file: {len(combined)}")
    else:
        print("No CSV files found.")

if __name__ == "__main__":
    main()
