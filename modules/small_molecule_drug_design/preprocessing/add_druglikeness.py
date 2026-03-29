#!/usr/bin/env python3
import pandas as pd
import sys


def add_druglikeness():
    """Add druglikeness column from enamine.csv to antibiotics_enamine.csv"""

    print("Loading enamine.csv...")
    enamine_df = pd.read_csv('enamine.csv')
    print(f"Loaded {len(enamine_df)} rows from enamine.csv")

    print("Loading antibiotics_enamine.csv...")
    antibiotics_df = pd.read_csv('antibiotics_enamine.csv')
    print(f"Loaded {len(antibiotics_df)} rows from antibiotics_enamine.csv")

    print("Merging datasets on SMILES...")
    # Merge on SMILES column, keeping all rows from antibiotics_enamine.csv
    merged_df = pd.merge(antibiotics_df, enamine_df, on='smiles', how='left')

    print(f"Merged dataset has {len(merged_df)} rows")
    print(
        f"Rows with druglikeness data: {merged_df['druglikeness'].notna().sum()}"
    )
    print(
        f"Rows missing druglikeness data: {merged_df['druglikeness'].isna().sum()}"
    )

    # Reduce float precision to save storage space
    print("Reducing float precision...")
    float_cols = merged_df.select_dtypes(include=['float64']).columns
    for col in float_cols:
        merged_df[col] = merged_df[col].round(4).astype('float32')

    # Save the merged dataset
    print("Saving updated antibiotics_enamine.csv...")
    merged_df.to_csv('antibiotics_enamine.csv', index=False)
    print("Done!")

    return merged_df


if __name__ == "__main__":
    add_druglikeness()
