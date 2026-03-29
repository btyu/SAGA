#!/usr/bin/env python3
"""
Pipeline to filter combined antibiotics by heavy-atom count, fragment molecules (BRICS by default),
and score fragments with the Klebsiella pneumoniae (KP) predictor.
"""

import argparse
import os
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

# Ensure project root is on sys.path for absolute imports
PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import datamol as dm
from rdkit import Chem

# Reuse the existing scoring stack
from modules.small_molecule_drug_design.evaluation import score_smiles as score_module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Filter antibiotics, fragment them (BRICS by default), and score fragments with KP predictor.",
    )
    parser.add_argument(
        "--input",
        default="modules/small_molecule_drug_design/data/molecules/combined_antibiotics.txt",
        help="Input SMILES file with one molecule per line (default: combined antibiotics list).",
    )
    parser.add_argument(
        "--output-dir",
        default="modules/small_molecule_drug_design/analysis/combined_antibiotics",
        help="Directory where the filtered SMILES, fragments, and KP scores will be written.",
    )
    parser.add_argument(
        "--max-heavy-atoms",
        type=int,
        default=80,
        help="Maximum heavy-atom count threshold (default: 80).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for KP scoring (default: 256).",
    )
    parser.add_argument(
        "--fragment-mode",
        default="brics",
        choices=["anybreak", "brics"],
        help="Fragmentation mode passed to datamol.break_mol (default: brics).",
    )
    parser.add_argument(
        "--skip-kp",
        action="store_true",
        help="Skip KP scoring (useful for debugging).",
    )
    return parser.parse_args()


def iter_smiles(path: Path) -> Iterable[str]:
    with path.open() as handle:
        for line in handle:
            smi = line.strip()
            if smi:
                yield smi.split()[0]


def canonicalize(mol: Chem.Mol) -> Optional[str]:
    try:
        return dm.standardize_smiles(dm.to_smiles(mol))
    except Exception:
        return None


def filter_by_heavy_atoms(
    smiles: Iterable[str], max_heavy: int
) -> Tuple[List[str], int, int, int]:
    filtered: List[str] = []
    invalid = 0
    too_large = 0
    sanitized = 0

    for smi in smiles:
        mol = dm.to_mol(smi, ordered=True)
        if mol is None:
            invalid += 1
            continue
        sanitized += 1
        try:
            mol = dm.sanitize_mol(mol)
        except Exception:
            invalid += 1
            continue

        heavy_atoms = mol.GetNumHeavyAtoms()
        if heavy_atoms > max_heavy:
            too_large += 1
            continue

        canon = canonicalize(mol)
        if canon:
            filtered.append(canon)
        else:
            invalid += 1

    return filtered, sanitized, invalid, too_large


def generate_fragments(smiles: Iterable[str], mode: str) -> Set[str]:
    fragments: Set[str] = set()
    for smi in smiles:
        mol = dm.to_mol(smi, ordered=True)
        if mol is None:
            continue
        try:
            parts, _ = dm.fragment.break_mol(
                mol,
                mode=mode,
                randomize=False,
                minFragmentSize=1,
                silent=True,
            )
        except Exception:
            continue

        for frag in parts:
            frag_mol = dm.to_mol(frag)
            if frag_mol is None:
                continue
            canon = canonicalize(frag_mol)
            if canon:
                fragments.add(canon)
    return fragments


def write_list(items: Iterable[str], path: Path) -> None:
    with path.open("w") as handle:
        for item in items:
            handle.write(f"{item}\n")


def score_fragments(
    fragments: List[str],
    output_csv: Path,
    batch_size: int,
) -> Optional["pd.DataFrame"]:
    score_module.configure_logging()
    df = score_module.score_smiles(
        fragments,
        ["klebsiella_pneumoniae"],
        batch_size=batch_size,
    )
    df.to_csv(output_csv, index=False)
    return df


def write_summary(
    summary_path: Path,
    *,
    total_input: int,
    sanitized: int,
    invalid: int,
    too_large: int,
    filtered_count: int,
    fragment_count: int,
    fragment_mode: str,
    kp_stats: Optional[Tuple[float, float, float]] = None,
) -> None:
    summary_lines = [
        "# Combined Antibiotics Fragment Analysis",
        "",
        f"- Total input molecules: {total_input}",
        f"- Sanitized molecules: {sanitized}",
        f"- Invalid molecules: {invalid}",
        f"- Molecules filtered due to heavy atom threshold: {too_large}",
        f"- Molecules passing heavy atom filter: {filtered_count}",
        f"- Unique {fragment_mode} fragments: {fragment_count}",
    ]
    if kp_stats:
        min_score, mean_score, max_score = kp_stats
        summary_lines.extend(
            [
                "- KP score statistics for fragments:",
                f"  - min: {min_score:.4f}",
                f"  - mean: {mean_score:.4f}",
                f"  - max: {max_score:.4f}",
            ]
        )

    summary_path.write_text("\n".join(summary_lines) + "\n")


def main():
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading SMILES from {input_path}")
    all_smiles = list(iter_smiles(input_path))
    total_input = len(all_smiles)
    print(f"[INFO] Loaded {total_input} SMILES")

    print(f"[INFO] Filtering molecules with heavy atom count <= {args.max_heavy_atoms}")
    filtered, sanitized, invalid, too_large = filter_by_heavy_atoms(
        all_smiles, args.max_heavy_atoms
    )
    filtered_path = output_dir / "filtered_smiles.smi"
    write_list(filtered, filtered_path)
    print(f"[INFO] Wrote {len(filtered)} filtered SMILES to {filtered_path}")

    print(f"[INFO] Generating fragments using mode='{args.fragment_mode}'")
    fragments = sorted(generate_fragments(filtered, args.fragment_mode))
    fragments_path = output_dir / "fragments.smi"
    write_list(fragments, fragments_path)
    print(f"[INFO] Wrote {len(fragments)} unique fragments to {fragments_path}")

    kp_stats = None
    kp_csv = output_dir / "kp_scores.csv"
    if fragments and not args.skip_kp:
        print("[INFO] Scoring fragments with KP predictor")
        df = score_fragments(fragments, kp_csv, args.batch_size)
        if df is not None and "klebsiella_pneumoniae" in df.columns:
            kp_scores = df["klebsiella_pneumoniae"].dropna()
            if len(kp_scores) > 0:
                kp_stats = (
                    kp_scores.min(),
                    kp_scores.mean(),
                    kp_scores.max(),
                )
        print(f"[INFO] KP scores saved to {kp_csv}")
    elif args.skip_kp:
        print("[INFO] Skipping KP scoring as requested.")
    else:
        print("[WARN] No fragments generated; skipping KP scoring.")

    summary_path = output_dir / "summary.md"
    write_summary(
        summary_path,
        total_input=total_input,
        sanitized=sanitized,
        invalid=invalid,
        too_large=too_large,
        filtered_count=len(filtered),
        fragment_count=len(fragments),
        fragment_mode=args.fragment_mode,
        kp_stats=kp_stats,
    )
    print(f"[INFO] Summary written to {summary_path}")


if __name__ == "__main__":
    sys.exit(main())

