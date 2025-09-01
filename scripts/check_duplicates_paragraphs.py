# scripts/check_paragraph_duplicates.py
import argparse
import re
from pathlib import Path
import pandas as pd

"""
Check and remove duplicate paragraphs in a CSV based on normalized text.
This script reads a CSV file, normalizes the 'paragraph_text' column,
and identifies duplicates based on that normalized text.
It can keep the first, last, or drop all duplicates based on user choice.
"""
DEFAULT_COMBINED = Path(r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\data\processed\investment_activity_classification\combined_investment_activity_classification.csv")
#DEFAULT_COMBINED = Path(r"C:\Users\6559484\OneDrive - Universiteit Utrecht\Desktop\BiodiversityASSET_SODA\data\processed\assetization_features_scoring\combined_assetization_features_scoring.csv")
def normalize_text(s: str) -> str:
    """Trim and collapse inner whitespace so minor formatting differences don't break duplicate checks."""
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"\s+", " ", s)
    return s

def main():
    parser = argparse.ArgumentParser(description="Check and remove duplicates based on paragraph_text.")
    parser.add_argument("--input", type=Path, default=DEFAULT_COMBINED,
                        help="Path to the CSV to check (default: combined_with_investor_name.csv).")
    parser.add_argument("--out-dedup", type=Path, default=None,
                        help="Where to save the deduplicated CSV (default: <input> with _dedup suffix).")
    parser.add_argument("--out-dups", type=Path, default=None,
                        help="Where to save rows that were duplicates (default: <input> with _duplicates suffix).")
    parser.add_argument("--keep", choices=["first","last","none"], default="first",
                        help="Which duplicate to keep: first/last/none (none drops all dup groups). Default: first.")
    args = parser.parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    # Defaults for outputs
    if args.out_dedup is None:
        args.out_dedup = args.input.with_name(args.input.stem + "_dedup.csv")
    if args.out_dups is None:
        args.out_dups = args.input.with_name(args.input.stem + "_duplicates.csv")

    print(f"📄 Reading: {args.input}")
    df = pd.read_csv(args.input)

    if "paragraph_text" not in df.columns:
        raise ValueError("Column 'paragraph_text' not found in the input CSV.")

    # Build normalized key for duplicate detection
    df["_para_norm"] = df["paragraph_text"].apply(normalize_text)

    total_rows = len(df)
    # Identify duplicates based on normalized text
    dup_mask = df.duplicated(subset=["_para_norm"], keep=False)
    dups_df = df[dup_mask].copy()

    # Choose how to keep rows
    if args.keep == "first":
        dedup_df = df.drop_duplicates(subset=["_para_norm"], keep="first").copy()
    elif args.keep == "last":
        dedup_df = df.drop_duplicates(subset=["_para_norm"], keep="last").copy()
    else:  # none -> drop all rows that are in a duplicate group
        dedup_df = df[~dup_mask].copy()

    # Stats (compute BEFORE dropping the helper column)
    unique_norm = dedup_df.shape[0]
    dup_count_rows = dups_df.shape[0]
    dup_groups_n = df.loc[dup_mask, "_para_norm"].nunique() if dup_count_rows else 0

    # Optionally add ranks (not strictly needed)
    # dup_groups = (
    #     dups_df.assign(_grp_rank=dups_df.groupby("_para_norm").cumcount())
    #     if dup_count_rows else dups_df
    # )

    # Now it's safe to drop the helper column
    for _df in (dups_df, dedup_df):
        if "_para_norm" in _df.columns:
            _df.drop(columns=["_para_norm"], inplace=True)

    # Save outputs
    dedup_df.to_csv(args.out_dedup, index=False, encoding="utf-8")
    dups_df.to_csv(args.out_dups, index=False, encoding="utf-8")


    print("✅ Done.")
    print(f"   Total rows:            {total_rows}")
    print(f"   Duplicate rows (norm): {dup_count_rows}")
    print(f"   Duplicate groups:      {dup_groups_n}")
    print(f"   Rows after dedup:      {unique_norm}")
    print(f"💾 Saved deduplicated CSV: {args.out_dedup}")
    print(f"🧾 Saved duplicates list:  {args.out_dups}")

if __name__ == "__main__":
    main()
