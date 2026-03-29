"""
Analyze aggregated antibiotic design results, compute threshold-based counts,
select a top-diverse subset, and generate a UMAP visualization.

Usage:
  python -m modules.small_molecule_drug_design.experimental.antibiotics_results_analyzer \
    --input-dir /path/to/logs/run_folder \
    --tanimoto-threshold 0.4 \
    --leniency 0 \
    [--k 1000] [--pattern "**/*selected.csv"]
"""

# pylint: disable=import-error,no-name-in-module,no-member

from typing import Tuple, List, Optional
import os
import argparse

import pandas as pd

from modules.small_molecule_drug_design.postprocessing.aggregate_selection import (
    aggergate_all_csvs, )
from modules.small_molecule_drug_design.utils.log_selection import (
    select_top_diverse_from_df,
    save_selected,
)


def ensure_analysis_dir(input_dir: str) -> str:
    analysis_dir = os.path.join(os.path.abspath(input_dir), "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    return analysis_dir


def compute_threshold_counts(
    df: pd.DataFrame,
    smiles_column: str,
    score_column: str,
    k: int,
    tanimoto_threshold: float,
    leniency: int,
) -> pd.DataFrame:
    required_cols = ["staph_aureus_chemprop", "deepdl_druglikeness"]
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' for analysis")

    if smiles_column not in df.columns:
        raise KeyError(f"Missing required column '{smiles_column}' for analysis")
    if score_column not in df.columns:
        raise KeyError(f"Missing required column '{score_column}' for analysis")

    conds: List[Tuple[str, float, float]] = [
        ("staph>0.30 & deepdl>0.70", 0.30, 0.70),
        ("staph>0.25 & deepdl>0.70", 0.25, 0.70),
        ("staph>0.20 & deepdl>0.70", 0.20, 0.70),
        ("staph>0.30 & deepdl>0.60", 0.30, 0.60),
        ("staph>0.25 & deepdl>0.60", 0.25, 0.60),
        ("staph>0.20 & deepdl>0.60", 0.20, 0.60),
    ]

    rows = []
    staph = pd.to_numeric(df["staph_aureus_chemprop"], errors="coerce")
    deepdl = pd.to_numeric(df["deepdl_druglikeness"], errors="coerce")
    base = df.copy()
    base[smiles_column] = base[smiles_column].astype(str)
    base[score_column] = pd.to_numeric(base[score_column], errors="coerce")
    for name, staph_thr, deepdl_thr in conds:
        mask = (staph > staph_thr) & (deepdl > deepdl_thr)
        filtered = base[mask].copy()
        if len(filtered) == 0:
            rows.append({"filter": name, "count": 0})
            continue
        selected_df, _ = select_top_diverse_from_df(
            df=filtered,
            smiles_column=smiles_column,
            score_column=score_column,
            k=k,
            tanimoto_threshold=tanimoto_threshold,
            leniency=leniency,
        )
        rows.append({"filter": name, "count": int(len(selected_df))})
    return pd.DataFrame(rows)


def compute_umap_for_smiles(smiles: List[str]) -> pd.DataFrame:
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator as rdFPGen
    import numpy as np
    import umap

    gen = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)
    fps = []
    valid_indices = []
    for idx, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        bv = gen.GetFingerprint(m)
        arr = np.zeros((1, ), dtype=int)
        DataStructs.ConvertToNumpyArray(bv, arr)
        fps.append(arr.astype(bool))
        valid_indices.append(idx)

    if not fps:
        return pd.DataFrame(columns=["x", "y", "row_index"])

    X = np.vstack(fps)
    reducer = umap.UMAP(n_components=2, metric="jaccard", random_state=42)
    emb = reducer.fit_transform(X)
    return pd.DataFrame({
        "x": emb[:, 0],
        "y": emb[:, 1],
        "row_index": valid_indices,
    })


def save_umap_plot(df_coords: pd.DataFrame, save_path: str, colors: Optional[List[float]] = None, cmap: str = "viridis") -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    if colors is None:
        scatter = ax.scatter(df_coords["x"], df_coords["y"], s=6, alpha=0.8, c="#4C78A8")
    else:
        scatter = ax.scatter(df_coords["x"], df_coords["y"], s=6, alpha=0.85, c=colors, cmap=cmap)
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label("Aggregated score", rotation=270, labelpad=12)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("UMAP of selected molecules")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    parser = argparse.ArgumentParser(
        description="Analyze antibiotic results, compute counts, and UMAP",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir",
                        required=True,
                        type=str,
                        help="Root folder to scan for CSV logs")
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
                        help="Number of molecules to select for diversity")
    parser.add_argument("--tanimoto-threshold",
                        type=float,
                        default=0.4,
                        help="Max allowed Tanimoto similarity in selection")
    parser.add_argument("--leniency",
                        type=int,
                        default=0,
                        help="Allow up to this many similarities >= threshold")

    args = parser.parse_args()

    input_dir = os.path.abspath(args.input_dir)
    analysis_dir = ensure_analysis_dir(input_dir)

    # 1) Aggregate and deduplicate
    merged_df = aggergate_all_csvs(
        input_dir=input_dir,
        pattern=args.pattern,
        smiles_column=args.smiles_column,
        score_column=args.score_column,
    )

    # 2) Threshold-based counts using diverse selection under each filter
    counts_df = compute_threshold_counts(
        df=merged_df,
        smiles_column=args.smiles_column,
        score_column=args.score_column,
        k=args.k,
        tanimoto_threshold=args.tanimoto_threshold,
        leniency=args.leniency,
    )
    counts_path = os.path.join(analysis_dir, "threshold_counts.csv")
    counts_df.to_csv(counts_path, index=False)

    # 3) Select top-diverse from filtered set (staph>0.2 & deepdl>0.7)
    filtered = merged_df[
        (merged_df["staph_aureus_chemprop"].astype(float) > 0.2)
        & (merged_df["deepdl_druglikeness"].astype(float) > 0.6)].copy()

    if len(filtered) > 0:
        selected_df, _ = select_top_diverse_from_df(
            df=filtered,
            smiles_column=args.smiles_column,
            score_column=args.score_column,
            k=args.k,
            tanimoto_threshold=args.tanimoto_threshold,
            leniency=args.leniency,
        )
        selected_csv = os.path.join(analysis_dir, "filtered_top_diverse.csv")
        save_selected(selected_df, selected_csv)

        # 4) UMAP on selected molecules
        coords = compute_umap_for_smiles(
            filtered[args.smiles_column].tolist())
        # Attach aggregated scores aligned via row_index
        try:
            scores_series = pd.to_numeric(filtered.iloc[coords["row_index"].tolist()][args.score_column], errors="coerce")
            coords["score"] = scores_series.values
        except Exception:
            coords["score"] = float("nan")
        coords_csv = os.path.join(analysis_dir, "filtered_umap.csv")
        coords.to_csv(coords_csv, index=False)

        umap_png = os.path.join(analysis_dir, "filtered_umap.png")
        if len(coords) >= 2:
            color_values = coords["score"].tolist() if "score" in coords.columns else None
            save_umap_plot(coords, umap_png, colors=color_values)
        else:
            # Not enough points to plot meaningfully; still create empty placeholder
            with open(umap_png + ".txt", "w", encoding="utf-8") as f:
                f.write("Not enough valid molecules for UMAP plot.\n")
    else:
        # Save empty placeholders when filter yields nothing
        empty_csv = os.path.join(analysis_dir, "filtered_top_diverse.csv")
        pd.DataFrame().to_csv(empty_csv, index=False)
        with open(os.path.join(analysis_dir, "filtered_umap.png.txt"),
                  "w",
                  encoding="utf-8") as f:
            f.write("No molecules passed the staph/deepdl filter.\n")

    # 5) Also save a lightweight summary file
    summary_txt = os.path.join(analysis_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Input dir: {input_dir}\n")
        f.write(f"Pattern: {args.pattern}\n")
        f.write(f"Unique molecules (deduped by SMILES): {len(merged_df)}\n")
        f.write(f"Counts CSV: {counts_path}\n")


if __name__ == "__main__":
    main()
