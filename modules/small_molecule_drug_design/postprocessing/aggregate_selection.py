"""
Post-processing: aggregate generated molecules from CSV logs in a folder,
deduplicate by SMILES (keeping best aggregate score), and select top-diverse
K molecules.

Can be used as:
  - A reusable function: aggregate_and_select_folder(...)
  - A CLI script: python -m modules.small_molecule_drug_design.postprocessing.aggregate_selection --input-dir <dir>
"""

# pylint: disable=import-error,no-name-in-module,no-member

from typing import List, Tuple, Optional
import os
import glob

import pandas as pd

from modules.small_molecule_drug_design.utils.log_selection import (
    load_and_merge_csvs,
    select_top_diverse_from_df,
    save_selected,
)


def collect_csv_paths(input_dir: str, pattern: str) -> List[str]:
    """
    Recursively find CSV files under input_dir matching the given pattern.

    Args:
        input_dir: Root directory to search.
        pattern: Glob pattern relative to input_dir (e.g., "**/*selected.csv").

    Returns:
        List of absolute CSV file paths.
    """
    search_glob = os.path.join(input_dir, pattern)
    paths = glob.glob(search_glob, recursive=True)
    # Normalize to absolute paths
    return [os.path.abspath(p) for p in paths if p.lower().endswith('.csv')]


def aggergate_all_csvs(
    input_dir: str,
    pattern: str = "**/*selected.csv",
    smiles_column: str = "smiles",
    score_column: str = "aggregate",
) -> pd.DataFrame:
    """
    Aggregate all matching CSVs under a folder.
    """
    input_dir = os.path.abspath(input_dir)
    csv_paths = collect_csv_paths(input_dir=input_dir, pattern=pattern)
    if not csv_paths:
        raise FileNotFoundError(
            f"No CSV files found in '{input_dir}' matching pattern '{pattern}'"
        )
    return load_and_merge_csvs(csv_paths=csv_paths,
                               smiles_column=smiles_column,
                               score_column=score_column)


def aggregate_and_select_folder(
    input_dir: str,
    pattern: str = "**/*selected.csv",
    smiles_column: str = "smiles",
    score_column: str = "aggregate",
    k: int = 1000,
    tanimoto_threshold: float = 0.4,
    leniency: int = 0,
    output_path: Optional[str] = None,
) -> Tuple[str, int, int]:
    """
    Aggregate all matching CSVs under a folder and select top-diverse K molecules.

    Args:
        input_dir: Directory containing CSV logs (search is recursive).
        pattern: Glob pattern for CSVs to include. Default targets generation-selected files.
        smiles_column: Name of the SMILES column.
        score_column: Name of the aggregated score column.
        k: Number of molecules to select.
        tanimoto_threshold: Max allowed similarity among selected molecules.
        leniency: Allow up to this many similarities >= threshold.
        output_path: Optional path to store the selected CSV. If not provided,
            it will be created inside input_dir.

    Returns:
        (output_csv_path, total_unique_molecules, selected_count)
    """
    print(f"Post aggregation: tanimoto_threshold", tanimoto_threshold, "leniency", leniency)
    merged_df = aggergate_all_csvs(input_dir=input_dir,
                                   pattern=pattern,
                                   smiles_column=smiles_column,
                                   score_column=score_column)
    selected_df, _ = select_top_diverse_from_df(
        df=merged_df,
        smiles_column=smiles_column,
        score_column=score_column,
        k=k,
        tanimoto_threshold=tanimoto_threshold,
        leniency=leniency,
    )

    if output_path is None:
        base_name = os.path.basename(os.path.normpath(input_dir))
        output_path = os.path.join(input_dir,
                                   f"{base_name}_selected_top{k}_diverse.csv")

    save_selected(selected_df, output_path)

    return output_path, int(len(merged_df)), int(len(selected_df))


def main():
    import argparse
    # RDKit logger silent
    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')

    parser = argparse.ArgumentParser(
        description="Aggregate CSV logs and select top-diverse molecules",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir",
                        required=True,
                        type=str,
                        help="Folder to scan recursively for CSV logs")
    parser.add_argument("--pattern",
                        type=str,
                        default="**/*selected.csv",
                        help="Glob pattern for CSVs to include")
    parser.add_argument("--smiles-column",
                        type=str,
                        default="smiles",
                        help="SMILES column name")
    parser.add_argument("--score-column",
                        type=str,
                        default="aggregate",
                        help="Aggregated score column name")
    parser.add_argument("--k",
                        type=int,
                        default=1000,
                        help="Number of molecules to select")
    parser.add_argument("--tanimoto-threshold",
                        type=float,
                        default=0.4,
                        help="Max allowed Tanimoto similarity")
    parser.add_argument("--leniency",
                        type=int,
                        default=0,
                        help="Allow up to this many similarities >= threshold")
    parser.add_argument("--output", type=str, help="Optional output CSV path")

    args = parser.parse_args()

    out_path, total, selected = aggregate_and_select_folder(
        input_dir=args.input_dir,
        pattern=args.pattern,
        smiles_column=args.smiles_column,
        score_column=args.score_column,
        k=args.k,
        tanimoto_threshold=args.tanimoto_threshold,
        leniency=args.leniency,
        output_path=args.output,
    )

    print("=" * 60)
    print("AGGREGATION AND DIVERSE SELECTION")
    print("=" * 60)
    print(f"Input dir: {os.path.abspath(args.input_dir)}")
    print(f"Pattern: {args.pattern}")
    print(f"Unique molecules loaded: {total}")
    print(f"Selected molecules: {selected}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
