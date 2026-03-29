"""
Multi-seed analyzer: given multiple run folders (seeds), aggregate molecules
from each run, filter by staph/druglikeness thresholds, compute a UMAP across
the combined set, and plot colored-by-seed scatter plots per threshold.

Usage:
  python -m modules.small_molecule_drug_design.experimental.antibiotics_multiple_seed_analyzer \
    --input-dirs /path/to/run_seedA /path/to/run_seedB \
    [--pattern "**/*selected.csv"] [--output-dir /path/to/save]

Example: 
    python -m modules.small_molecule_drug_design.experimental.antibiotics_multiple_seed_analyzer     --input-dirs  /projects/jlab/to.shen/SciLeoAgent/logs/an
tibiotics_original-1_20250907_123427 /projects/jlab/to.shen/SciLeoAgent/logs/antibiotics_origina
l-2_20250907_123427 /projects/jlab/to.shen/SciLeoAgent/logs/antibiotics_original-3_20250907_1234
26 /projects/jlab/to.shen/SciLeoAgent/logs/antibiotics_original-4_20250907_135959 /projects/jlab
/to.shen/SciLeoAgent/logs/antibiotics_original-5_20250907_135959


Notes:
  - Uses Jaccard distance over Morgan fingerprints (radius=2, fpSize=2048)
    with UMAP to embed SMILES.
  - Points in each plot are colored by their originating seed (run folder).
"""

# pylint: disable=import-error,no-name-in-module,no-member

from typing import List, Dict, Tuple
import os
import argparse

import pandas as pd

from modules.small_molecule_drug_design.postprocessing.aggregate_selection import (
    aggergate_all_csvs,
)
from modules.small_molecule_drug_design.utils.log_selection import (
    select_top_diverse_from_df,
    save_selected,
)


def ensure_output_dir(output_dir: str) -> str:
    out_dir = os.path.abspath(output_dir)
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def load_seed_data(
    input_dirs: List[str],
    pattern: str,
    smiles_column: str,
    staph_column: str,
    deepdl_column: str,
    score_column: str,
) -> Dict[str, pd.DataFrame]:
    """Load and deduplicate molecules per seed (run folder).

    Returns mapping seed_name -> DataFrame that includes at least
    [smiles_column, staph_column, deepdl_column] and other columns.
    """
    seed_to_df: Dict[str, pd.DataFrame] = {}
    for run_dir in input_dirs:
        seed_name = os.path.basename(os.path.normpath(run_dir))
        try:
            df = aggergate_all_csvs(
                input_dir=run_dir,
                pattern=pattern,
                smiles_column=smiles_column,
                score_column=score_column,
            )
        except Exception as exc:  # pragma: no cover - robust CLI behavior
            print(f"[WARN] Skipping '{run_dir}': {exc}")
            continue

        missing_cols = [c for c in [staph_column, deepdl_column] if c not in df.columns]
        if missing_cols:
            print(
                f"[WARN] Seed '{seed_name}' missing required columns {missing_cols}; skipping."
            )
            continue

        df = df.copy()
        df[smiles_column] = df[smiles_column].astype(str)
        df[staph_column] = pd.to_numeric(df[staph_column], errors="coerce")
        df[deepdl_column] = pd.to_numeric(df[deepdl_column], errors="coerce")
        df = df.dropna(subset=[smiles_column, staph_column, deepdl_column])
        df["seed"] = seed_name
        seed_to_df[seed_name] = df
    return seed_to_df


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
        arr = np.zeros((1,), dtype=int)
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


def compute_threshold_counts(
    df: pd.DataFrame,
    smiles_column: str,
    score_column: str,
    staph_column: str,
    deepdl_column: str,
    k: int,
    tanimoto_threshold: float,
    leniency: int,
) -> pd.DataFrame:
    """Compute counts after diversity selection under several thresholds.

    Mirrors single-seed analyzer behavior but parameterized by column names.
    """
    for col in [staph_column, deepdl_column, smiles_column, score_column]:
        if col not in df.columns:
            raise KeyError(f"Missing required column '{col}' for analysis")

    conds: List[Tuple[str, float, float]] = [
        ("staph>0.30 & deepdl>0.70", 0.30, 0.70),
        ("staph>0.25 & deepdl>0.70", 0.25, 0.70),
        ("staph>0.20 & deepdl>0.70", 0.20, 0.70),
        ("staph>0.30 & deepdl>0.60", 0.30, 0.60),
        ("staph>0.25 & deepdl>0.60", 0.25, 0.60),
        ("staph>0.20 & deepdl>0.60", 0.20, 0.60),
    ]

    rows = []
    staph = pd.to_numeric(df[staph_column], errors="coerce")
    deepdl = pd.to_numeric(df[deepdl_column], errors="coerce")
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


def plot_umap_colored_by_seed(
    df_coords: pd.DataFrame,
    seeds: List[str],
    save_path: str,
) -> str:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    unique_seeds = sorted(set(seeds))
    # Build a color map with enough distinct colors
    base_cmap = plt.cm.get_cmap("tab20", max(20, len(unique_seeds)))
    colors = [base_cmap(i % base_cmap.N) for i in range(len(unique_seeds))]
    seed_to_color = {seed: colors[i] for i, seed in enumerate(unique_seeds)}

    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    for seed in unique_seeds:
        mask = df_coords["seed"] == seed
        if not np.any(mask):
            continue
        ax.scatter(
            df_coords.loc[mask, "x"],
            df_coords.loc[mask, "y"],
            s=6,
            alpha=0.85,
            c=[seed_to_color[seed]],
            label=seed,
        )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=7, markerscale=2, frameon=False)
    ax.set_title("UMAP across seeds")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return save_path


def main():
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")

    parser = argparse.ArgumentParser(
        description="Multi-seed UMAP analyzer colored by seed",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dirs",
        nargs="+",
        required=True,
        help="List of run folders (seeds)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*selected.csv",
        help="Glob pattern for CSVs to include",
    )
    parser.add_argument(
        "--smiles-column",
        type=str,
        default="smiles",
        help="SMILES column name",
    )
    parser.add_argument(
        "--staph-column",
        type=str,
        default="staph_aureus_chemprop",
        help="Staph score column name",
    )
    parser.add_argument(
        "--deepdl-column",
        type=str,
        default="deepdl_druglikeness",
        help="Druglikeness score column name",
    )
    parser.add_argument(
        "--score-column",
        type=str,
        default="aggregate",
        help="Aggregated score column name (for loading)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs (defaults next to first input)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="Number of molecules to select for diversity",
    )
    parser.add_argument(
        "--tanimoto-threshold",
        type=float,
        default=0.4,
        help="Max allowed Tanimoto similarity in selection",
    )
    parser.add_argument(
        "--leniency",
        type=int,
        default=0,
        help="Allow up to this many similarities >= threshold",
    )

    args = parser.parse_args()

    input_dirs = [os.path.abspath(p) for p in args.input_dirs]
    if args.output_dir is None:
        base = os.path.dirname(os.path.abspath(input_dirs[0]))
        out_dir = os.path.join(base, "analysis_multi_seed")
    else:
        out_dir = args.output_dir
    out_dir = ensure_output_dir(out_dir)
    print(f"[INFO] Outputs will be saved to: {out_dir}")

    seed_to_df = load_seed_data(
        input_dirs=input_dirs,
        pattern=args.pattern,
        smiles_column=args.smiles_column,
        staph_column=args.staph_column,
        deepdl_column=args.deepdl_column,
        score_column=args.score_column,
    )

    if not seed_to_df:
        raise SystemExit("No seeds loaded with required columns; nothing to do.")

    # Combine across seeds and deduplicate by SMILES using best score
    combined = pd.concat(list(seed_to_df.values()), ignore_index=True)
    combined[args.score_column] = pd.to_numeric(combined[args.score_column], errors="coerce")
    combined[args.staph_column] = pd.to_numeric(combined[args.staph_column], errors="coerce")
    combined[args.deepdl_column] = pd.to_numeric(combined[args.deepdl_column], errors="coerce")
    combined[args.smiles_column] = combined[args.smiles_column].astype(str)
    combined = combined.dropna(subset=[args.smiles_column])
    combined = combined.sort_values(by=args.score_column, ascending=False)
    combined = combined.drop_duplicates(subset=[args.smiles_column], keep="first").reset_index(drop=True)

    # 1) Threshold-based counts using diverse selection under each filter
    counts_df = compute_threshold_counts(
        df=combined,
        smiles_column=args.smiles_column,
        score_column=args.score_column,
        staph_column=args.staph_column,
        deepdl_column=args.deepdl_column,
        k=args.k,
        tanimoto_threshold=args.tanimoto_threshold,
        leniency=args.leniency,
    )
    counts_path = os.path.join(out_dir, "threshold_counts.csv")
    counts_df.to_csv(counts_path, index=False)
    print(f"[INFO] Wrote threshold counts: {counts_path}")

    # 2) Select top-diverse from filtered combined set (staph>0.2 & deepdl>0.6)
    filtered = combined[
        (combined[args.staph_column].astype(float) > 0.2)
        & (combined[args.deepdl_column].astype(float) > 0.6)
    ].copy()
    if len(filtered) > 0:
        selected_df, _ = select_top_diverse_from_df(
            df=filtered,
            smiles_column=args.smiles_column,
            score_column=args.score_column,
            k=args.k,
            tanimoto_threshold=args.tanimoto_threshold,
            leniency=args.leniency,
        )
        selected_csv = os.path.join(out_dir, "filtered_top_diverse.csv")
        save_selected(selected_df, selected_csv)
        print(f"[INFO] Wrote filtered top-diverse: {selected_csv}")
    else:
        selected_csv = os.path.join(out_dir, "filtered_top_diverse.csv")
        pd.DataFrame().to_csv(selected_csv, index=False)
        print(f"[INFO] No filtered molecules; wrote empty: {selected_csv}")

    # Define thresholds to evaluate
    conds: List[Tuple[str, float, float]] = [
        ("staph>0.30 & deepdl>0.70", 0.30, 0.70),
        ("staph>0.25 & deepdl>0.70", 0.25, 0.70),
        ("staph>0.20 & deepdl>0.70", 0.20, 0.70),
        ("staph>0.30 & deepdl>0.60", 0.30, 0.60),
        ("staph>0.25 & deepdl>0.60", 0.25, 0.60),
        ("staph>0.20 & deepdl>0.60", 0.20, 0.60),
    ]

    # Per-threshold UMAP and plot
    for name, staph_thr, deepdl_thr in conds:
        pool_rows: List[Dict[str, str]] = []
        for seed, df in seed_to_df.items():
            mask = (df[args.staph_column] > staph_thr) & (df[args.deepdl_column] > deepdl_thr)
            if mask.any():
                sub = df.loc[mask, [args.smiles_column]].copy()
                sub["seed"] = seed
                pool_rows.append(sub)
        if not pool_rows:
            # Create a placeholder note
            safe_name = f"umap_staph{staph_thr:.2f}_deepdl{deepdl_thr:.2f}"
            with open(os.path.join(out_dir, safe_name + ".txt"), "w", encoding="utf-8") as f:
                f.write("No molecules passed the thresholds across all seeds.\n")
            continue

        pool = pd.concat(pool_rows, ignore_index=True)

        coords = compute_umap_for_smiles(pool[args.smiles_column].tolist())
        if len(coords) < 2:
            safe_name = f"umap_staph{staph_thr:.2f}_deepdl{deepdl_thr:.2f}"
            with open(os.path.join(out_dir, safe_name + ".txt"), "w", encoding="utf-8") as f:
                f.write("Not enough valid molecules for UMAP plot.\n")
            continue

        # Map seeds back by row_index
        coords["seed"] = pool.loc[coords["row_index"].astype(int), "seed"].values
        coords["smiles"] = pool.loc[coords["row_index"].astype(int), args.smiles_column].values

        safe_name = f"umap_staph{staph_thr:.2f}_deepdl{deepdl_thr:.2f}"
        csv_path = os.path.join(out_dir, safe_name + ".csv")
        coords[["x", "y", "smiles", "seed"]].to_csv(csv_path, index=False)

        png_path = os.path.join(out_dir, safe_name + ".png")
        plot_umap_colored_by_seed(coords, coords["seed"].tolist(), png_path)

    # Save a short summary
    summary_txt = os.path.join(out_dir, "summary.txt")
    with open(summary_txt, "w", encoding="utf-8") as f:
        f.write(f"Seeds: {', '.join(sorted(seed_to_df.keys()))}\n")
        f.write(f"Pattern: {args.pattern}\n")
        f.write(f"Output dir: {out_dir}\n")
        f.write(f"Unique molecules (deduped by SMILES): {len(combined)}\n")
        f.write(f"Counts CSV: {counts_path}\n")
        f.write(f"Filtered top-diverse CSV: {selected_csv}\n")
    print(f"[INFO] Wrote summary: {summary_txt}")


if __name__ == "__main__":
    main()



