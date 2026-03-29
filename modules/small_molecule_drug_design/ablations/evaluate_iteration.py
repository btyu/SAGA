#!/usr/bin/env python3
"""Simple Type 2 evaluation script for iteration directories.

Takes a directory path, finds all CSV files ending with _original, _mutation, 
_selected, or _crossover, aggregates SMILES, selects top 100 diverse molecules,
computes type 2 properties, and reports pass rates.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

# Ensure project root imports work
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.small_molecule_drug_design.ablations.held_out_metrics import (  # noqa: E402
    ensure_held_out_metrics,
    HELD_OUT_FILTERS,
)
from modules.small_molecule_drug_design.ablations.process_scileo_csvs import (  # noqa: E402
    PRIMARY_METRICS,
    coerce_numeric,
    compute_aggregate_score,
    normalize_columns,
    select_diverse_smiles,
)

# Import from evaluation_type2.py
from modules.small_molecule_drug_design.ablations.evaluation_type2 import (  # noqa: E402
    ACTIVITY_THRESHOLDS,
    FILTERS,
    count_passing,
    count_per_filter,
)


def find_csv_files(directory: Path) -> List[Path]:
    """Find all CSV files ending with _original, _mutation, _selected, or _crossover."""
    suffixes = ["_original.csv", "_mutation.csv", "_selected.csv", "_crossover.csv"]
    csv_files = []
    for suffix in suffixes:
        csv_files.extend(directory.rglob(f"*{suffix}"))
    return sorted(csv_files)


def load_and_aggregate_csvs(csv_files: List[Path]) -> pd.DataFrame:
    """Load all CSV files and aggregate into single dataframe."""
    dfs = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            dfs.append(df)
        except Exception as e:
            print(f"  Warning: Failed to load {csv_file}: {e}")
            continue
    
    if not dfs:
        raise ValueError("No CSV files could be loaded")
    
    combined_df = pd.concat(dfs, ignore_index=True)
    
    # Normalize columns
    combined_df = normalize_columns(combined_df)
    
    # Deduplicate by SMILES
    if "smiles" not in combined_df.columns:
        raise ValueError("No SMILES column found after normalization")
    
    initial_count = len(combined_df)
    combined_df = combined_df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
    final_count = len(combined_df)
    
    print(f"  Loaded {initial_count} molecules from {len(csv_files)} CSV files")
    print(f"  After deduplication: {final_count} unique molecules")
    
    return combined_df


def prepare_dataframe_for_evaluation(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare dataframe for type 2 evaluation."""
    df_copy = df.copy()
    
    # Coerce primary metrics to numeric
    df_copy = coerce_numeric(df_copy, PRIMARY_METRICS)
    
    # Compute aggregate score if not present
    if "aggregate_score" not in df_copy.columns:
        df_copy = compute_aggregate_score(df_copy)
    
    # Map kp to target_activity for evaluation_type2 functions
    if "kp" in df_copy.columns and "target_activity" not in df_copy.columns:
        df_copy["target_activity"] = df_copy["kp"]
    
    return df_copy


def compute_pass_rates(df: pd.DataFrame) -> Dict:
    """Compute pass rates for each metric and total pass rate."""
    results = {}
    
    # Per-filter pass rates at each activity threshold
    for act_thresh in ACTIVITY_THRESHOLDS:
        per_filter_results = count_per_filter(df, act_thresh)
        results[f"activity_{act_thresh}"] = per_filter_results
    
    # Total pass rate (passing all filters) at each activity threshold
    total_results = []
    for act_thresh in ACTIVITY_THRESHOLDS:
        passing, total = count_passing(df, act_thresh)
        total_results.append({
            "activity_threshold": act_thresh,
            "passing": passing,
            "total": total,
            "pass_rate": (passing / total * 100) if total > 0 else 0.0,
        })
    results["total_passing"] = total_results
    
    return results


def print_summary(results: Dict, num_molecules: int):
    """Print summary to console."""
    print("\n" + "=" * 70)
    print("TYPE 2 EVALUATION SUMMARY")
    print("=" * 70)
    print(f"Total molecules evaluated: {num_molecules}")
    print()
    
    # Print per-filter pass rates for first activity threshold
    act_thresh = ACTIVITY_THRESHOLDS[0]
    print(f"Per-filter pass rates (activity threshold >= {act_thresh}):")
    print("-" * 70)
    per_filter = results[f"activity_{act_thresh}"]
    
    # Activity first
    if "activity" in per_filter:
        act = per_filter["activity"]
        print(f"  Activity (>= {act_thresh}): {act['passing']}/{act['total']} ({act['pass_rate']:.1f}%)")
    
    # Then other filters
    filter_order = ["qed", "sa", "deepdl", "mw", "pains", "brenk", 
                    "antibiotics_novelty", "antibiotics_motifs_filter", 
                    "toxicity", "ring_score"]
    for filter_name in filter_order:
        if filter_name in per_filter:
            f = per_filter[filter_name]
            threshold = FILTERS.get(filter_name, "N/A")
            print(f"  {filter_name} (>= {threshold}): {f['passing']}/{f['total']} ({f['pass_rate']:.1f}%)")
    
    print()
    print("Total pass rate (passing ALL filters):")
    print("-" * 70)
    for total_result in results["total_passing"]:
        print(f"  Activity >= {total_result['activity_threshold']}: "
              f"{total_result['passing']}/{total_result['total']} "
              f"({total_result['pass_rate']:.1f}%)")
    print("=" * 70)


def save_results_to_csv(results: Dict, output_path: Path, num_molecules: int):
    """Save detailed results to CSV."""
    rows = []
    
    # Add per-filter pass rates
    for act_thresh in ACTIVITY_THRESHOLDS:
        per_filter = results[f"activity_{act_thresh}"]
        for filter_name, filter_data in per_filter.items():
            rows.append({
                "activity_threshold": act_thresh,
                "metric": filter_name,
                "threshold": FILTERS.get(filter_name, act_thresh if filter_name == "activity" else "N/A"),
                "passing": filter_data["passing"],
                "total": filter_data["total"],
                "pass_rate": filter_data["pass_rate"],
            })
    
    # Add total pass rates
    for total_result in results["total_passing"]:
        rows.append({
            "activity_threshold": total_result["activity_threshold"],
            "metric": "all_filters",
            "threshold": "all",
            "passing": total_result["passing"],
            "total": total_result["total"],
            "pass_rate": total_result["pass_rate"],
        })
    
    df_results = pd.DataFrame(rows)
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory path to search for CSV files (e.g., iteration_1)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV file path (default: <directory>_type2_results.csv)",
    )
    parser.add_argument(
        "--recompute-held-out",
        action="store_true",
        help="Force recomputation of held-out metrics even if columns already exist",
    )
    args = parser.parse_args()
    
    directory = args.directory.resolve()
    if not directory.exists():
        raise SystemExit(f"Directory not found: {directory}")
    
    # Find CSV files
    print(f"Searching for CSV files in {directory}...")
    csv_files = find_csv_files(directory)
    if not csv_files:
        raise SystemExit(f"No matching CSV files found in {directory}")
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Load and aggregate
    print("\nLoading and aggregating CSV files...")
    df = load_and_aggregate_csvs(csv_files)
    
    # Prepare dataframe
    print("\nPreparing dataframe...")
    df = prepare_dataframe_for_evaluation(df)
    
    # Select top 100 diverse molecules
    print("\nSelecting top 100 diverse molecules...")
    top_smiles = select_diverse_smiles(
        df,
        smiles_col="smiles",
        score_col="aggregate_score",
        limit=100,
        similarity_threshold=0.6,
    )
    
    print(f"  Selected {len(top_smiles)} diverse molecules")
    
    # Filter to top 100 diverse
    df_top100 = df[df["smiles"].isin(top_smiles)].copy()
    
    # Compute held-out metrics
    print("\nComputing held-out metrics...")
    df_top100 = ensure_held_out_metrics(
        df_top100,
        smiles_col="smiles",
        prefix="",
        recompute=args.recompute_held_out,
    )
    
    # Map held-out metrics to expected column names for evaluation_type2
    # Handle both cases: columns created with prefix and without prefix
    column_mapping = {
        "held_out_qed": "qed",
        "held_out_sa": "sa",
        "held_out_mw": "mw",
        "held_out_pains": "pains",
        "held_out_brenk": "brenk",
        "held_out_deepdl": "deepdl",
        "held_out_ring_score": "ring_score",
        "held_out_toxicity": "toxicity",
        "held_out_antibiotics_novelty": "antibiotics_novelty",
        "held_out_antibiotics_motifs_filter": "antibiotics_motifs_filter",
    }
    
    # Copy columns with prefix to columns without prefix if needed
    for old_col, new_col in column_mapping.items():
        if old_col in df_top100.columns and new_col not in df_top100.columns:
            df_top100[new_col] = df_top100[old_col]
    
    # Ensure columns match what FILTERS expects
    # normalize_columns creates "novelty" but FILTERS expects "antibiotics_novelty"
    if "novelty" in df_top100.columns and "antibiotics_novelty" not in df_top100.columns:
        df_top100["antibiotics_novelty"] = df_top100["novelty"]
    
    # Ensure toxicity column exists (normalize_columns creates it from toxicity_safety_chemprop)
    if "toxicity" not in df_top100.columns:
        if "toxicity_safety_chemprop" in df_top100.columns:
            df_top100["toxicity"] = pd.to_numeric(df_top100["toxicity_safety_chemprop"], errors="coerce")
    
    # Ensure antibiotics_motifs_filter exists (normalize_columns creates "motifs")
    if "motifs" in df_top100.columns and "antibiotics_motifs_filter" not in df_top100.columns:
        df_top100["antibiotics_motifs_filter"] = pd.to_numeric(df_top100["motifs"], errors="coerce")
    
    # Ensure target_activity is set
    if "target_activity" not in df_top100.columns:
        if "kp" in df_top100.columns:
            df_top100["target_activity"] = df_top100["kp"]
        elif "klebsiella_pneumoniae_minimol" in df_top100.columns:
            df_top100["target_activity"] = df_top100["klebsiella_pneumoniae_minimol"]
    
    # Compute pass rates
    print("\nComputing pass rates...")
    results = compute_pass_rates(df_top100)
    
    # Print summary
    print_summary(results, len(df_top100))
    
    # Save to CSV
    if args.output:
        output_path = args.output
    else:
        output_path = directory / f"{directory.name}_type2_results.csv"
    
    save_results_to_csv(results, output_path, len(df_top100))


if __name__ == "__main__":
    main()

