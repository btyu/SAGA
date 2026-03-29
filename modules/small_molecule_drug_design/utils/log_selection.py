"""
Utilities for loading generated molecules from CSV logs and selecting
top-diverse candidates by an aggregated score.

This module is independent of the optimizer loop and can be used from
CLI entry points to pre-select a diverse subset of molecules from past
runs.
"""

# pylint: disable=import-error,no-name-in-module,no-member

from typing import List, Tuple
import os

import pandas as pd

from .rdkit_utils import select_top_diverse_modes


def load_and_merge_csvs(
    csv_paths: List[str],
    smiles_column: str = "smiles",
    score_column: str = "aggregate",
) -> pd.DataFrame:
    """
    Load one or more CSV files containing molecules and their properties,
    concatenate them, and deduplicate by SMILES keeping the row with the
    highest aggregated score.

    Args:
        csv_paths: List of CSV file paths.
        smiles_column: Column name containing SMILES strings.
        score_column: Column name containing the aggregated score.

    Returns:
        A DataFrame with unique SMILES and all available properties/columns.
    """
    if not csv_paths:
        raise ValueError("csv_paths must not be empty")

    dataframes: List[pd.DataFrame] = []
    for path in csv_paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV file not found: {path}")
        df = pd.read_csv(path)
        if smiles_column not in df.columns:
            raise KeyError(
                f"Missing required column '{smiles_column}' in {path}"
            )
        if score_column not in df.columns:
            raise KeyError(
                f"Missing required column '{score_column}' in {path}"
            )
        dataframes.append(df)

    merged = pd.concat(dataframes, ignore_index=True)

    # Drop rows with missing SMILES or score
    merged = merged.dropna(subset=[smiles_column, score_column])

    # Ensure numeric score
    merged[score_column] = pd.to_numeric(merged[score_column], errors="coerce")
    merged = merged.dropna(subset=[score_column])

    # Deduplicate by SMILES, keeping the highest score row
    idx = merged.groupby(smiles_column)[score_column].idxmax()
    deduped = merged.loc[idx].reset_index(drop=True)
    return deduped


def select_top_diverse_from_df(
    df: pd.DataFrame,
    smiles_column: str = "smiles",
    score_column: str = "aggregate",
    k: int = 1000,
    tanimoto_threshold: float = 0.4,
    leniency: int = 0,
) -> Tuple[pd.DataFrame, List[int]]:
    """
    Select top-k diverse rows from a DataFrame using Tanimoto diversity
    on SMILES and greedily picking by descending score.

    Args:
        df: Input DataFrame with SMILES and score columns.
        smiles_column: Column with SMILES strings.
        score_column: Column with aggregated score to optimize.
        k: Number of molecules to select.
        tanimoto_threshold: Maximum allowed similarity between selected molecules.
        leniency: Allow up to this many similarities >= threshold in selection.

    Returns:
        (selected_df, selected_indices) where selected_df is a view (copy)
        of the top-diverse rows, and selected_indices are positions of the
        selected rows relative to the input df.
    """
    if smiles_column not in df.columns:
        raise KeyError(f"Missing required column '{smiles_column}' in DataFrame")
    if score_column not in df.columns:
        raise KeyError(f"Missing required column '{score_column}' in DataFrame")

    smiles_list = df[smiles_column].astype(str).tolist()
    scores_list = df[score_column].astype(float).tolist()

    selected_idx = select_top_diverse_modes(
        smiles_list=smiles_list,
        scores=scores_list,
        tanimoto_threshold=tanimoto_threshold,
        k=k,
        leniency=leniency,
    )

    selected_df = df.iloc[selected_idx].copy().reset_index(drop=True)
    return selected_df, selected_idx


def save_selected(
    df: pd.DataFrame,
    save_path: str,
) -> str:
    """
    Save the selected DataFrame to CSV.

    Args:
        df: DataFrame to save.
        save_path: Output CSV path.

    Returns:
        The path written to.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    return save_path


