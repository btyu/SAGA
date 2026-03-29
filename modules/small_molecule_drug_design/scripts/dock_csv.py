"""Dock a CSV of SMILES with UniDock and save docked poses.

Usage example:
  # Use a predefined target ID (recommended)
  python -m modules.small_molecule_drug_design.scripts.dock_csv \
    --input logs/20250817_175230_mpro_v0/20250817_175230_mpro_v0_selected_top1000_diverse.csv \
    --output logs/20250817_175230_mpro_v0/docked_top1000 \
    --target MPRO
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import List, Tuple

from rdkit import Chem

from modules.small_molecule_drug_design.docking.unidock import docking


def parse_center(center_str: str) -> Tuple[float, float, float]:
    parts = center_str.replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError("Center must have exactly 3 numbers")
    return float(parts[0]), float(parts[1]), float(parts[2])


def load_smiles_from_csv(csv_path: Path, smiles_column: str = "smiles") -> List[str]:
    smiles_list: List[str] = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        if smiles_column not in (reader.fieldnames or []):
            raise ValueError(
                f"Column '{smiles_column}' not found in CSV. Found: {reader.fieldnames}")
        for row in reader:
            smi = (row.get(smiles_column) or "").strip()
            if smi:
                smiles_list.append(smi)
    return smiles_list


def main(argv: List[str] | None = None) -> int:
    curr_file = Path(__file__).resolve()
    data_pdb_dir = curr_file.parent.parent / "data" / "pdb"
    base_dir = curr_file.parent.parent / "data"

    # Hardcoded targets (paths and centers) to match scorer defaults
    protein_targets: dict[str, tuple[Path, Tuple[float, float, float]]] = {
        "DRD2": (data_pdb_dir / "DRD2.pdb", (9.925, 5.846, -9.582)),
        "GSK3B": (data_pdb_dir / "GSK3B.pdb", (-14.782, -17.079, -3.559)),
        "JNK3": (data_pdb_dir / "JNK3.pdb", (23.167, 8.921, 31.848)),
        "BRD4": (data_pdb_dir / "BRD4.pdb", (28.751, 15.826, -2.335)),
        "MPRO": (data_pdb_dir / "MPRO.pdb", (9.050, 8.898, -1.508)),
        "MARS1": (base_dir / "sars_mers_combined" / "mers_test_1_protein.pdb",
                   (7.813, -0.981, 22.566)),
    }

    # Defaults target SARS-CoV-2 Mpro if not specified
    default_protein = protein_targets["MPRO"][0]
    default_center = protein_targets["MPRO"][1]

    parser = argparse.ArgumentParser(
        description="Dock SMILES from a CSV with UniDock and save poses.")
    parser.add_argument(
        "--input",
        required=True,
        type=Path,
        help="Path to input CSV with a 'smiles' column.",
    )
    parser.add_argument(
        "--output",
        required=False,
        type=Path,
        default=None,
        help="Directory to save docked poses (SDF files). Defaults to <csv_dir>/docked_poses",
    )
    parser.add_argument(
        "--target",
        required=False,
        type=str,
        choices=sorted(protein_targets.keys()),
        help="Predefined protein target ID (e.g., MPRO, DRD2, BRD4). Overrides --protein/--center.",
    )
    parser.add_argument(
        "--protein",
        required=False,
        type=Path,
        default=default_protein,
        help=f"Protein PDB path (default: {default_protein})",
    )
    parser.add_argument(
        "--center",
        required=False,
        type=str,
        default=f"{default_center[0]} {default_center[1]} {default_center[2]}",
        help="Pocket center as three numbers: 'x y z'",
    )
    parser.add_argument(
        "--seed",
        required=False,
        type=int,
        default=42,
        help="Random seed for embedding/docking.",
    )
    parser.add_argument(
        "--size",
        required=False,
        type=float,
        default=20.0,
        help="Dock box size (edge length).",
    )
    parser.add_argument(
        "--search_mode",
        required=False,
        type=str,
        default="balance",
        choices=["fast", "balance", "detailed"],
        help="UniDock search mode.",
    )
    parser.add_argument(
        "--smiles_column",
        required=False,
        type=str,
        default="smiles",
        help="SMILES column name in the CSV.",
    )
    parser.add_argument(
        "--batch_size",
        required=False,
        type=int,
        default=100,
        help="Number of molecules to dock per batch (enables progress bar and avoids filename collisions).",
    )
    parser.add_argument(
        "--no_progress",
        action="store_true",
        help="Disable progress bar output.",
    )

    args = parser.parse_args(argv)

    csv_path: Path = args.input
    output_dir: Path = args.output or (csv_path.parent / "docked_poses")
    if args.target:
        protein_path, center = protein_targets[args.target]
    else:
        protein_path = args.protein
        center = parse_center(args.center)

    if not csv_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {csv_path}")
    if not protein_path.exists():
        raise FileNotFoundError(f"Protein PDB not found: {protein_path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    smiles_list = load_smiles_from_csv(csv_path, smiles_column=args.smiles_column)
    if not smiles_list:
        raise ValueError("No SMILES found in input CSV.")

    rdmols = []
    for idx, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            # Skip invalid SMILES but keep going for the rest
            continue
        rdmols.append(mol)

    if not rdmols:
        raise ValueError("All SMILES failed to parse into RDKit molecules.")

    # Dock in batches to provide a progress bar and stable filenames
    total = len(rdmols)
    use_pbar = (tqdm is not None) and (not args.no_progress)
    pbar = tqdm(total=total, desc="Docking", unit="mol") if use_pbar else None

    global_index = 0
    batch_size = max(1, int(args.batch_size))
    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        batch = rdmols[start:end]

        results = docking(
            rdmols=batch,
            protein_path=protein_path,
            center=center,
            seed=args.seed,
            size=args.size,
            search_mode=args.search_mode,
            output_path=None,  # we will save to unique filenames ourselves
        )

        for local_idx, (mol, _score) in enumerate(results):
            if mol is not None:
                out_file = output_dir / f"docked_{global_index}.sdf"
                with Chem.SDWriter(str(out_file)) as w:
                    w.write(mol)
            global_index += 1

        if pbar:
            pbar.update(end - start)

    if pbar:
        pbar.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


