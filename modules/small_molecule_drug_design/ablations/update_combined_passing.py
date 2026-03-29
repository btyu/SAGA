#!/usr/bin/env python3
"""
Update combined passing molecules CSV by reading existing passing molecule files
and checking level 2 iterations directly.
"""

import pandas as pd
from pathlib import Path
import sys

# Add evaluation script to path to import functions
sys.path.insert(0, str(Path(__file__).parent))
from evaluation_type2 import (
    count_passing,
    ACTIVITY_THRESHOLDS,
    OUTPUT_DIR,
)

TARGET = "ecoli"
PASSING_DIR = OUTPUT_DIR / "passing_molecules"
TOP100_DIR = Path("top100_diverse_molecules")


def get_passing_from_csv_files():
    """Read passing molecules from existing CSV files."""
    all_passing = []
    
    # All SciLeo iterations - read from existing passing molecule files
    scileo_files = [
        ("scileo_level1_iter1_ecoli", "level1_iter1"),
        ("scileo_level1_iter2_ecoli", "level1_iter2"),
        ("scileo_level1_iter3_ecoli", "level1_iter3"),
        ("scileo_level2_iter1_ecoli", "level2_iter1"),
        ("scileo_level2_iter2_ecoli", "level2_iter2"),
    ]
    
    print("Reading existing passing molecule files...")
    for file_prefix, method_suffix in scileo_files:
        method_name = f"SciLeo_{method_suffix}"
        for act_thresh in ACTIVITY_THRESHOLDS:
            thresh_str = str(act_thresh).replace(".", "p")
            file_path = PASSING_DIR / f"{file_prefix}_activity{thresh_str}.csv"
            
            if file_path.exists():
                df = pd.read_csv(file_path)
                if len(df) > 0:
                    df = df.copy()
                    df["method"] = method_name
                    df["activity_threshold"] = act_thresh
                    all_passing.append(df)
                    print(f"  {method_name} activity≥{act_thresh}: {len(df)} molecules")
            else:
                # Try without .backup extension if file doesn't exist
                backup_path = file_path.with_suffix(file_path.suffix + ".backup")
                if backup_path.exists():
                    df = pd.read_csv(backup_path)
                    if len(df) > 0:
                        df = df.copy()
                        df["method"] = method_name
                        df["activity_threshold"] = act_thresh
                        all_passing.append(df)
                        print(f"  {method_name} activity≥{act_thresh}: {len(df)} molecules (from backup)")
    
    return all_passing


def main():
    """Main function."""
    print(f"\n{'='*70}")
    print("Updating Combined SciLeo Passing Molecules")
    print("=" * 70)
    
    # Get passing molecules from existing CSV files (includes all levels now)
    all_passing = get_passing_from_csv_files()
    
    if not all_passing:
        print("\nNo passing molecules found!")
        return
    
    # Combine all DataFrames
    print(f"\nCombining {len(all_passing)} datasets...")
    combined = pd.concat(all_passing, ignore_index=True)
    
    # Remove duplicates based on SMILES (keep highest activity)
    print(f"Before deduplication: {len(combined)} molecules")
    combined = combined.sort_values("target_activity", ascending=False)
    combined = combined.drop_duplicates(subset=["smiles"], keep="first")
    print(f"After deduplication: {len(combined)} unique molecules")
    
    # Sort by activity (descending - highest activity first)
    combined = combined.sort_values("target_activity", ascending=False)
    
    # Reset index
    combined = combined.reset_index(drop=True)
    
    # Save to CSV
    output_file = OUTPUT_DIR / "scileo_all_passing_molecules_combined.csv"
    combined.to_csv(output_file, index=False)
    
    print(f"\n{'='*70}")
    print(f"✓ Saved {len(combined)} unique passing molecules to:")
    print(f"  {output_file}")
    print(f"\nActivity range: {combined['target_activity'].min():.4f} - {combined['target_activity'].max():.4f}")
    print(f"Methods included: {sorted(combined['method'].unique())}")
    print(f"Activity thresholds: {sorted(combined['activity_threshold'].unique())}")
    print("=" * 70)


if __name__ == "__main__":
    main()

