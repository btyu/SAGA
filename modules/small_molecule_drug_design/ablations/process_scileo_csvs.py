#!/usr/bin/env python3
"""Process every CSV in `scileo_ablations` and emit unified evaluation artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

# Ensure project root imports work when running this file directly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from modules.small_molecule_drug_design.ablations.held_out_metrics import (  # noqa: E402
    ensure_held_out_metrics,
    summarize_pass_rates,
)


PRIMARY_METRICS = ("kp", "novelty", "toxicity", "motifs", "similarity")
PRIMARY_THRESHOLDS: Dict[str, float] = {
    "kp": 0.05,  # Match evaluation activity threshold (lower threshold)
    "novelty": 0.6,
    "toxicity": 0.5,
    "motifs": 1.0,
    "similarity": 0.5,
}

DEFAULT_TOP_K = 100
DEFAULT_SIMILARITY_THRESHOLD = 0.6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=SCRIPT_DIR / "scileo_ablations",
        help="Directory that contains the raw ablation CSV files (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=SCRIPT_DIR / "results",
        help="Directory where annotated CSVs, top SMILES, and summaries will be written",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of diverse molecules to emit per dataset (default: %(default)s)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=DEFAULT_SIMILARITY_THRESHOLD,
        help="Maximum Tanimoto similarity allowed when picking diverse SMILES",
    )
    parser.add_argument(
        "--recompute-held-out",
        action="store_true",
        help="Force recomputation of held-out metrics even if columns already exist",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop immediately if any CSV fails to process",
    )
    parser.add_argument(
        "--include-subdirs",
        action="store_true",
        help="Also recurse into subdirectories of the data directory (default: only top-level CSVs)",
    )
    parser.add_argument(
        "--limit",
        nargs="*",
        default=None,
        help="Optional list of CSV basenames to process (e.g., reinvent4_kp_output...).",
    )
    return parser.parse_args()


def discover_csvs(data_dir: Path, include_subdirs: bool) -> List[Path]:
    if include_subdirs:
        return sorted(path for path in data_dir.rglob("*.csv") if path.is_file())
    return sorted(path for path in data_dir.glob("*.csv") if path.is_file())


def sanitize_method_name(csv_path: Path, data_dir: Path) -> str:
    relative = csv_path.relative_to(data_dir)
    name = relative.with_suffix("").as_posix()
    return name.replace("/", "__")


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names and map to PRIMARY_METRICS."""
    df_copy = df.copy()

    # Normalize SMILES column
    if "smiles" not in df_copy.columns:
        for candidate in ["SMILES", "Smiles", "SMILE"]:
            if candidate in df_copy.columns:
                df_copy = df_copy.rename(columns={candidate: "smiles"})
                break

    # Map columns to PRIMARY_METRICS
    # KP: klebsiella_pneumoniae_minimol or klebsiella_pneumoniae -> kp
    if "kp" not in df_copy.columns:
        for col in [
            "klebsiella_pneumoniae_minimol",
            "klebsiella_pneumoniae",
            "escherichia_coli_minimol",
            "escherichia_coli",
        ]:
            if col in df_copy.columns:
                df_copy["kp"] = df_copy[col]
                break

    # Novelty: antibiotics_novelty -> novelty
    if "novelty" not in df_copy.columns and "antibiotics_novelty" in df_copy.columns:
        df_copy["novelty"] = df_copy["antibiotics_novelty"]

    # Toxicity: toxicity_safety_chemprop -> toxicity
    if "toxicity" not in df_copy.columns:
        for col in ["toxicity_safety_chemprop", "toxicity"]:
            if col in df_copy.columns:
                df_copy["toxicity"] = df_copy[col]
                break

    # Motifs: antibiotics_motifs_filter -> motifs (convert boolean to float)
    if (
        "motifs" not in df_copy.columns
        and "antibiotics_motifs_filter" in df_copy.columns
    ):
        df_copy["motifs"] = pd.to_numeric(
            df_copy["antibiotics_motifs_filter"], errors="coerce"
        ).replace({True: 1.0, False: 0.0})

    # Similarity: arthor_similarity or similarity -> similarity
    if "similarity" not in df_copy.columns:
        for col in ["arthor_similarity", "similarity"]:
            if col in df_copy.columns:
                df_copy["similarity"] = df_copy[col]
                break

    return df_copy


def coerce_numeric(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    df_copy = df.copy()
    for col in columns:
        if col in df_copy.columns:
            df_copy[col] = pd.to_numeric(df_copy[col], errors="coerce")
    return df_copy


def compute_aggregate_score(df: pd.DataFrame) -> pd.DataFrame:
    """Compute aggregate score from PRIMARY_METRICS. Exclude missing metrics."""
    df_copy = df.copy()
    available_metrics = [col for col in PRIMARY_METRICS if col in df_copy.columns]
    if not available_metrics:
        print(f"  Warning: No primary metrics available, setting aggregate_score to 0")
        df_copy["aggregate_score"] = 0.0
        return df_copy

    missing = [col for col in PRIMARY_METRICS if col not in df_copy.columns]
    if missing:
        print(
            f"  Warning: Missing primary metrics: {missing}, excluding from aggregate_score"
        )

    # Use geometric mean (product) of available metrics
    df_copy["aggregate_score"] = df_copy[available_metrics].prod(axis=1)
    return df_copy


def select_diverse_smiles(
    df: pd.DataFrame,
    *,
    smiles_col: str,
    score_col: str,
    limit: int,
    similarity_threshold: float,
) -> List[str]:
    selected: List[str] = []
    fingerprints: List = []
    ranked = df.dropna(subset=[smiles_col, score_col]).sort_values(
        score_col, ascending=False
    )

    for _, row in ranked.iterrows():
        smi = row[smiles_col]
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue

        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        if all(
            DataStructs.TanimotoSimilarity(fp, existing_fp) < similarity_threshold
            for existing_fp in fingerprints
        ):
            selected.append(smi)
            fingerprints.append(fp)

        if len(selected) >= limit:
            break

    return selected


def summarize_primary_metrics(df: pd.DataFrame) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for metric in PRIMARY_METRICS:
        if metric not in df.columns:
            continue
        series = df[metric].dropna()
        total = int(len(series))
        if total == 0:
            continue
        threshold = PRIMARY_THRESHOLDS.get(metric, 0.5)
        pass_count = int((series >= threshold).sum())
        rows.append(
            {
                "metric_name": metric,
                "column_name": metric,
                "metric_group": "primary",
                "threshold": float(threshold),
                "pass_count": pass_count,
                "total": total,
                "pass_rate": pass_count / total if total else 0.0,
            }
        )
    return rows


def summarize_all_metrics(df: pd.DataFrame) -> List[Dict[str, float]]:
    rows = summarize_primary_metrics(df)
    held_out_rows = summarize_pass_rates(df, prefix="held_out_")
    for entry in held_out_rows:
        rows.append(
            {
                **entry,
                "metric_group": "held_out",
            }
        )
    return rows


def ensure_output_dirs(base: Path) -> Dict[str, Path]:
    dirs = {
        "annotated": base / "annotated",
        "top_smiles": base / "top_smiles",
        "summaries": base / "summaries",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def write_top_smiles(smiles: List[str], path: Path) -> None:
    with path.open("w") as handle:
        for smi in smiles:
            handle.write(f"{smi}\n")


def process_csv(
    csv_path: Path,
    *,
    data_dir: Path,
    output_dirs: Dict[str, Path],
    args: argparse.Namespace,
) -> List[Dict[str, float]]:
    print(f"\nProcessing {csv_path} ...")
    method_name = sanitize_method_name(csv_path, data_dir)
    df = pd.read_csv(csv_path)
    df = normalize_columns(df)
    if "smiles" not in df.columns:
        raise ValueError(f"No SMILES column found in {csv_path}")

    df = df.dropna(subset=["smiles"]).drop_duplicates(subset=["smiles"])
    df = coerce_numeric(df, PRIMARY_METRICS)
    df = compute_aggregate_score(df)
    df = ensure_held_out_metrics(
        df,
        smiles_col="smiles",
        prefix="held_out_",
        recompute=args.recompute_held_out,
    )

    annotated_path = output_dirs["annotated"] / f"{method_name}_annotated.csv"
    df.to_csv(annotated_path, index=False)
    print(f"  ✓ Annotated CSV saved to {annotated_path}")

    top_smiles = select_diverse_smiles(
        df,
        smiles_col="smiles",
        score_col="aggregate_score",
        limit=args.top_k,
        similarity_threshold=args.similarity_threshold,
    )
    top_file = output_dirs["top_smiles"] / f"{method_name}_top{args.top_k}_smiles.smi"
    write_top_smiles(top_smiles, top_file)
    print(f"  ✓ Saved {len(top_smiles)} diverse SMILES to {top_file}")

    # Filter dataframe to top 100 diverse molecules for pass rate computation
    df_top100 = df[df["smiles"].isin(top_smiles)].copy()
    summary_rows = summarize_all_metrics(df_top100)
    summary_df = pd.DataFrame(summary_rows)
    summary_df.insert(0, "method", method_name)
    summary_df.insert(1, "source_csv", str(csv_path.relative_to(data_dir)))
    summary_path = output_dirs["summaries"] / f"{method_name}_pass_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  ✓ Pass summary saved to {summary_path}")

    return summary_df.to_dict(orient="records")


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not data_dir.exists():
        raise SystemExit(f"Data directory not found: {data_dir}")

    output_dirs = ensure_output_dirs(output_dir)
    csv_paths = discover_csvs(data_dir, include_subdirs=args.include_subdirs)

    if args.limit:
        selected = []
        for path in csv_paths:
            if path.name in args.limit or path.stem in args.limit:
                selected.append(path)
        csv_paths = selected

    if not csv_paths:
        print("No CSV files found to process.")
        return

    combined_rows: List[Dict[str, float]] = []
    for csv_path in csv_paths:
        try:
            combined_rows.extend(
                process_csv(
                    csv_path,
                    data_dir=data_dir,
                    output_dirs=output_dirs,
                    args=args,
                )
            )
        except Exception as exc:
            print(f"  ✗ Failed to process {csv_path}: {exc}")
            if args.fail_fast:
                raise

    if not combined_rows:
        print("No summaries were generated.")
        return

    combined_df = pd.DataFrame(combined_rows)
    combined_df = combined_df.sort_values(
        ["metric_group", "method", "metric_name"]
    ).reset_index(drop=True)
    combined_path = output_dir / "held_out_pass_rates.csv"
    combined_df.to_csv(combined_path, index=False)
    print(f"\n✓ Combined pass-rate table saved to {combined_path}")


if __name__ == "__main__":
    main()
