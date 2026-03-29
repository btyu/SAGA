# from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, cast

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.ML.Cluster import Butina

# Import filtering function
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.rdkit_utils import filter_smiles_preserves_existing_hits

MORGAN_GEN = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)


def _load_antibiotic_smiles() -> List[str]:
    try:
        data_path = (Path(__file__).resolve().parent / "combined_antibiotics.txt")
        return [
            line.strip()
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, FileNotFoundError, UnicodeDecodeError):
        return []


def _load_hts_smiles() -> List[str]:
    try:
        data_path = (Path(__file__).resolve().parent / "high_throughput_screen_positives.txt")
        return [
            line.strip()
            for line in data_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, FileNotFoundError, UnicodeDecodeError):
        return []


def _mol_from_smiles(smiles: str):
    fn_any: Any = getattr(Chem, "MolFromSmiles", None)
    if not callable(fn_any):
        return None
    return fn_any(smiles)


def _bulk_tanimoto_similarity(query_fp, ref_fps) -> List[float]:
    fn_any: Any = getattr(DataStructs, "BulkTanimotoSimilarity", None)
    if not callable(fn_any):
        return []
    return fn_any(query_fp, list(ref_fps))


def _ensure_antibiotic_fps() -> List:
    smiles_list = _load_antibiotic_smiles()
    fps: List = []
    for s in smiles_list:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        fps.append(MORGAN_GEN.GetFingerprint(mol))
    return fps


def novelty_against_antibiotics(smiles: str,
                                ref_fps: Iterable) -> Optional[float]:
    """Compute novelty based on whole molecule similarity."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    if not ref_fps:
        return 1.0
    query_fp = MORGAN_GEN.GetFingerprint(mol)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_fps))
    max_sim = float(max(sims)) if sims else 0.0
    score = 1.0 - max_sim
    if score < 0.0:
        score = 0.0
    return score


def scaffold_novelty_against_antibiotics(smiles: str,
                                         ref_scaffold_fps: Iterable) -> Optional[float]:
    """Compute novelty based on Murcko scaffold similarity."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    if not ref_scaffold_fps:
        return 1.0
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
    except Exception:
        return None
    query_fp = MORGAN_GEN.GetFingerprint(scaffold)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_scaffold_fps))
    max_sim = float(max(sims)) if sims else 0.0
    score = 1.0 - max_sim
    if score < 0.0:
        score = 0.0
    return score


def novelty_against_hts_hits(smiles: str,
                             ref_fps: Iterable) -> Optional[float]:
    """Compute novelty based on whole molecule similarity to HTS hits."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    if not ref_fps:
        return 1.0
    query_fp = MORGAN_GEN.GetFingerprint(mol)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_fps))
    max_sim = float(max(sims)) if sims else 0.0
    score = 1.0 - max_sim
    if score < 0.0:
        score = 0.0
    return score


def scaffold_novelty_against_hts_hits(smiles: str,
                                      ref_scaffold_fps: Iterable) -> Optional[float]:
    """Compute novelty based on Murcko scaffold similarity to HTS hits."""
    mol = _mol_from_smiles(smiles)
    if mol is None:
        return None
    if not ref_scaffold_fps:
        return 1.0
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
    except Exception:
        return None
    query_fp = MORGAN_GEN.GetFingerprint(scaffold)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_scaffold_fps))
    max_sim = float(max(sims)) if sims else 0.0
    score = 1.0 - max_sim
    if score < 0.0:
        score = 0.0
    return score


def find_closest_antibiotic(smiles: str, ref_fps: List, ref_smiles: List) -> Optional[str]:
    """Find closest antibiotic by whole molecule similarity."""
    mol = _mol_from_smiles(smiles)
    if mol is None or not ref_fps:
        return None
    query_fp = MORGAN_GEN.GetFingerprint(mol)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_fps))
    if not sims:
        return None
    max_idx = sims.index(max(sims))
    return ref_smiles[max_idx]


def find_closest_antibiotic_by_scaffold(smiles: str, ref_scaffold_fps: List, ref_smiles: List) -> Optional[str]:
    """Find closest antibiotic by Murcko scaffold similarity."""
    mol = _mol_from_smiles(smiles)
    if mol is None or not ref_scaffold_fps:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
    except Exception:
        return None
    query_fp = MORGAN_GEN.GetFingerprint(scaffold)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_scaffold_fps))
    if not sims:
        return None
    max_idx = sims.index(max(sims))
    return ref_smiles[max_idx]


def find_closest_hts_hit(smiles: str, ref_fps: List, ref_smiles: List) -> Optional[str]:
    """Find closest HTS hit by whole molecule similarity."""
    mol = _mol_from_smiles(smiles)
    if mol is None or not ref_fps:
        return None
    query_fp = MORGAN_GEN.GetFingerprint(mol)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_fps))
    if not sims:
        return None
    max_idx = sims.index(max(sims))
    return ref_smiles[max_idx]


def find_closest_hts_hit_by_scaffold(smiles: str, ref_scaffold_fps: List, ref_smiles: List) -> Optional[str]:
    """Find closest HTS hit by Murcko scaffold similarity."""
    mol = _mol_from_smiles(smiles)
    if mol is None or not ref_scaffold_fps:
        return None
    try:
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None or scaffold.GetNumAtoms() == 0:
            return None
    except Exception:
        return None
    query_fp = MORGAN_GEN.GetFingerprint(scaffold)
    sims = _bulk_tanimoto_similarity(query_fp, list(ref_scaffold_fps))
    if not sims:
        return None
    max_idx = sims.index(max(sims))
    return ref_smiles[max_idx]


def add_novelty_column(in_csv: Path, out_csv: Optional[Path] = None) -> Path:
    in_csv = in_csv.resolve()
    if out_csv is None:
        out_csv = in_csv

    # Load both reference databases
    antibiotic_smiles = _load_antibiotic_smiles()
    hts_smiles = _load_hts_smiles()

    # Generate whole-molecule fingerprints for both databases
    antibiotic_fps = []
    for s in antibiotic_smiles:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        antibiotic_fps.append(MORGAN_GEN.GetFingerprint(mol))

    hts_fps = []
    for s in hts_smiles:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        hts_fps.append(MORGAN_GEN.GetFingerprint(mol))

    # Generate scaffold fingerprints for both databases
    antibiotic_scaffold_fps = []
    for s in antibiotic_smiles:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                antibiotic_scaffold_fps.append(MORGAN_GEN.GetFingerprint(scaffold))
        except Exception:
            continue

    hts_scaffold_fps = []
    for s in hts_smiles:
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            if scaffold is not None and scaffold.GetNumAtoms() > 0:
                hts_scaffold_fps.append(MORGAN_GEN.GetFingerprint(scaffold))
        except Exception:
            continue

    # Read entire CSV first
    original_rows: List[List[str]] = []
    with in_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            original_rows.append(row)

    if not original_rows:
        return in_csv

    header = original_rows[0] if original_rows else []
    header = header[:] if header else []

    # Determine or create indices
    smiles_idx = header.index("smiles") if "smiles" in header else 0
    known_antibiotics_novelty_idx = header.index("known_antibiotics_novelty") if "known_antibiotics_novelty" in header else None
    hts_hit_novelty_idx = header.index("hts_hit_novelty") if "hts_hit_novelty" in header else None
    scaffold_antibiotics_novelty_idx = header.index("scaffold_antibiotics_novelty") if "scaffold_antibiotics_novelty" in header else None
    scaffold_hts_hit_novelty_idx = header.index("scaffold_hts_hit_novelty") if "scaffold_hts_hit_novelty" in header else None
    closest_antibiotic_idx = header.index("closest_antibiotic") if "closest_antibiotic" in header else None
    closest_hts_hit_idx = header.index("closest_hts_hit") if "closest_hts_hit" in header else None
    closest_antibiotic_scaffold_idx = header.index("closest_antibiotic_scaffold") if "closest_antibiotic_scaffold" in header else None
    closest_hts_hit_scaffold_idx = header.index("closest_hts_hit_scaffold") if "closest_hts_hit_scaffold" in header else None
    cluster_idx = header.index("cluster_id") if "cluster_id" in header else None
    filter_status_idx = header.index("filter_status") if "filter_status" in header else None
    filter_reason_idx = header.index("filter_reason") if "filter_reason" in header else None

    if known_antibiotics_novelty_idx is None:
        header.append("known_antibiotics_novelty")
        known_antibiotics_novelty_idx = len(header) - 1
    if hts_hit_novelty_idx is None:
        header.append("hts_hit_novelty")
        hts_hit_novelty_idx = len(header) - 1
    if scaffold_antibiotics_novelty_idx is None:
        header.append("scaffold_antibiotics_novelty")
        scaffold_antibiotics_novelty_idx = len(header) - 1
    if scaffold_hts_hit_novelty_idx is None:
        header.append("scaffold_hts_hit_novelty")
        scaffold_hts_hit_novelty_idx = len(header) - 1
    if closest_antibiotic_idx is None:
        header.append("closest_antibiotic")
        closest_antibiotic_idx = len(header) - 1
    if closest_hts_hit_idx is None:
        header.append("closest_hts_hit")
        closest_hts_hit_idx = len(header) - 1
    if closest_antibiotic_scaffold_idx is None:
        header.append("closest_antibiotic_scaffold")
        closest_antibiotic_scaffold_idx = len(header) - 1
    if closest_hts_hit_scaffold_idx is None:
        header.append("closest_hts_hit_scaffold")
        closest_hts_hit_scaffold_idx = len(header) - 1
    if cluster_idx is None:
        header.append("cluster_id")
        cluster_idx = len(header) - 1
    if filter_status_idx is None:
        header.append("filter_status")
        filter_status_idx = len(header) - 1
    if filter_reason_idx is None:
        header.append("filter_reason")
        filter_reason_idx = len(header) - 1

    # Prepare fingerprints for clustering (valid SMILES only)
    smiles_list: List[str] = []
    row_positions: List[int] = []  # positions within original_rows[1:]
    for idx, r in enumerate(original_rows[1:]):
        if not r:
            continue
        smiles_list.append(r[smiles_idx])
        row_positions.append(idx)
    fps: List = []
    valid_row_positions: List[int] = [
    ]  # corresponding positions in original_rows[1:]
    for idx, s in enumerate(smiles_list):
        mol = _mol_from_smiles(s)
        if mol is None:
            continue
        fps.append(MORGAN_GEN.GetFingerprint(mol))
        valid_row_positions.append(row_positions[idx])

    # Compute Butina clusters with Tanimoto distance threshold
    cluster_labels_for_valid: List[int] = []
    if fps:
        n = len(fps)
        dists: List[float] = []
        for i in range(1, n):
            sims = _bulk_tanimoto_similarity(fps[i], fps[:i])
            dists.extend([1.0 - s for s in sims])
        clusters = Butina.ClusterData(dists, n, 0.6, isDistData=True)
        labels = [-1] * n
        for cid, cluster in enumerate(clusters):
            for vi in cluster:
                labels[vi] = cid
        cluster_labels_for_valid = labels

    # Get filter information for all SMILES
    filter_results = {}
    if smiles_list:
        _, filter_reasons = filter_smiles_preserves_existing_hits(smiles_list)
        for i, smiles in enumerate(smiles_list):
            if smiles in filter_reasons:
                filter_results[smiles] = ("FILTERED", filter_reasons[smiles])
            else:
                filter_results[smiles] = ("KEPT", "")

    # Build output rows
    rows: List[List[str]] = [header]
    # Map from original_rows index (starting at 1) to cluster label string
    rowpos_to_label = {
        valid_row_positions[i]: ("" if cluster_labels_for_valid[i] < 0 else
                                 str(cluster_labels_for_valid[i]))
        for i in range(len(valid_row_positions))
    }

    for row_idx, row in enumerate(original_rows[1:]):
        if not row:
            continue
        # Extend/trim row to header length
        new_row = row[:]
        if len(new_row) < len(header):
            new_row.extend([""] * (len(header) - len(new_row)))
        # Compute novelty scores and find closest matches
        smiles = new_row[smiles_idx]

        # Whole molecule scores
        antibiotic_score = novelty_against_antibiotics(smiles, antibiotic_fps)
        hts_score = novelty_against_hts_hits(smiles, hts_fps)
        closest_antibiotic = find_closest_antibiotic(smiles, antibiotic_fps, antibiotic_smiles)
        closest_hts_hit = find_closest_hts_hit(smiles, hts_fps, hts_smiles)

        # Scaffold-based scores
        scaffold_antibiotic_score = scaffold_novelty_against_antibiotics(smiles, antibiotic_scaffold_fps)
        scaffold_hts_score = scaffold_novelty_against_hts_hits(smiles, hts_scaffold_fps)
        closest_antibiotic_scaffold = find_closest_antibiotic_by_scaffold(smiles, antibiotic_scaffold_fps, antibiotic_smiles)
        closest_hts_hit_scaffold = find_closest_hts_hit_by_scaffold(smiles, hts_scaffold_fps, hts_smiles)

        new_row[known_antibiotics_novelty_idx] = "" if antibiotic_score is None else f"{antibiotic_score:.6f}"
        new_row[hts_hit_novelty_idx] = "" if hts_score is None else f"{hts_score:.6f}"
        new_row[scaffold_antibiotics_novelty_idx] = "" if scaffold_antibiotic_score is None else f"{scaffold_antibiotic_score:.6f}"
        new_row[scaffold_hts_hit_novelty_idx] = "" if scaffold_hts_score is None else f"{scaffold_hts_score:.6f}"
        new_row[closest_antibiotic_idx] = closest_antibiotic or ""
        new_row[closest_hts_hit_idx] = closest_hts_hit or ""
        new_row[closest_antibiotic_scaffold_idx] = closest_antibiotic_scaffold or ""
        new_row[closest_hts_hit_scaffold_idx] = closest_hts_hit_scaffold or ""
        # Assign cluster id if available for this row
        label = rowpos_to_label.get(row_idx, "")
        new_row[cluster_idx] = label
        # Add filter information
        filter_status, filter_reason = filter_results.get(smiles, ("UNKNOWN", ""))
        new_row[filter_status_idx] = filter_status
        new_row[filter_reason_idx] = filter_reason
        rows.append(new_row)

    tmp_out = out_csv.with_suffix(out_csv.suffix + ".tmp")
    with tmp_out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        for r in rows:
            writer.writerow(r)
    tmp_out.replace(out_csv)
    return out_csv


def main() -> None:
    base = Path(__file__).resolve().parent
    files = [
        base / "filtered_A_baumanii_avg.csv",
        base / "filtered_E_coli_avg.csv",
        base / "filtered_P_aeruginosa_avg.csv",
        base / "filtered_K_pneumoniae_avg.csv",
        base / "filtered_N_gonorrhoeae_avg.csv",
    ]
    for p in files:
        if p.exists():
            add_novelty_column(p)


if __name__ == "__main__":
    main()
