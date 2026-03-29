from pathlib import Path
from typing import List, Optional, Sequence

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen

_MORGAN_GEN = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)

from scileo_agent.core.registry import register_scorer
from scileo_agent.core.data_models import Candidate

# Import the SMARTS patterns and PAINS catalog from antibiotics_scorer_mcp
from modules.small_molecule_drug_design.scorer_mcp.antibiotics_scorer_mcp.base import (
    BAD_SMARTS_LIST,
    PAINS_CATALOG,
)

_ANTIBIOTIC_WHOLE_MOLECULE_FPS: Optional[
    List
] = None  # RDKit bit vectors for whole molecules


def _load_antibiotic_smiles() -> List[str]:
    """Load SMILES strings of existing antibiotics from the data files.

    Loads from both combined_antibiotics.txt and broad_hts_coadd_hits.txt.
    Returns an empty list if both files are missing or unreadable.
    """
    smiles_list: List[str] = []
    data_dir = Path(__file__).resolve().parent.parent / "data" / "molecules"

    # Load from combined_antibiotics.txt
    try:
        data_path = data_dir / "combined_antibiotics.txt"
        lines = data_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            s = line.strip()
            if s:
                smiles_list.append(s)
    except Exception:
        pass

    # Load from broad_hts_coadd_hits.txt
    try:
        data_path = data_dir / "broad_hts_coadd_hits.txt"
        lines = data_path.read_text(encoding="utf-8").splitlines()
        for line in lines:
            s = line.strip()
            if s:
                smiles_list.append(s)
    except Exception:
        pass

    return smiles_list


def _ensure_antibiotic_whole_molecule_fingerprints() -> Sequence:
    """Ensure whole molecule fingerprints for the antibiotics reference set are computed and cached."""
    global _ANTIBIOTIC_WHOLE_MOLECULE_FPS
    if _ANTIBIOTIC_WHOLE_MOLECULE_FPS is not None:
        return _ANTIBIOTIC_WHOLE_MOLECULE_FPS

    smiles_list = _load_antibiotic_smiles()
    fps: List = []
    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        try:
            # Use whole molecule fingerprint
            fp = _MORGAN_GEN.GetFingerprint(mol)
            fps.append(fp)
        except Exception:
            # Skip molecules where fingerprint generation fails
            continue

    _ANTIBIOTIC_WHOLE_MOLECULE_FPS = fps
    return _ANTIBIOTIC_WHOLE_MOLECULE_FPS


def _novelty_score_against_antibiotics(smiles: str) -> Optional[float]:
    """Compute novelty score based on whole molecule Tanimoto similarity to known antibiotics.

    Returns (1 - maximum Tanimoto similarity) to any known antibiotic.
    Returns 0.0 if input SMILES is invalid. If no reference antibiotics are
    available, returns 1.0.
    """
    # Handle non-string inputs
    if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
        return 0.0

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0

    ref_fps = _ensure_antibiotic_whole_molecule_fingerprints()
    if not ref_fps:
        return 1.0

    query_fp = _MORGAN_GEN.GetFingerprint(mol)
    sims = DataStructs.BulkTanimotoSimilarity(query_fp, list(ref_fps))
    max_sim = float(max(sims)) if sims else 0.0

    return max(0.0, 1.0 - max_sim)


@register_scorer(
    name="antibiotics_novelty",
    population_wise=False,
    description="Antibiotics novelty score based on whole molecule dissimilarity (value range: 0.0 to 1.0). "
    "This score is computed as (1 - maximum Tanimoto similarity) using Morgan fingerprints (radius=2, 2048 bits) on complete molecules against a reference set of existing marketed antibiotics. "
    "Score interpretation: 1.0 = completely different from all known antibiotics (maximum novelty), 0.0 = identical to a known antibiotic (no novelty). "
    "High scores (>0.7) indicate high structural novelty that may circumvent existing resistance mechanisms, while low scores (<0.3) suggest close similarity to known antibiotics. "
    "Novel antibiotics are crucial for combating antimicrobial resistance.",
)
def score_antibiotics_novelty(candidates: List[Candidate]) -> List[float]:
    """Batch scorer: 1 - max whole molecule similarity to any known antibiotic for each candidate.

    Optimized for batch processing: computes all fingerprints upfront and processes
    similarities in batch for better performance.

    - Returns 0.0 for invalid SMILES.
    - Returns 1.0 if the antibiotics reference set is empty or unavailable.
    """
    if not candidates:
        return []

    ref_fps = _ensure_antibiotic_whole_molecule_fingerprints()
    if not ref_fps:
        # If no reference antibiotics, return maximum novelty for all candidates
        return [1.0] * len(candidates)

    # Batch compute fingerprints for all valid candidates
    query_fps: List = []
    valid_indices: List[int] = []
    results: List[float] = [0.0] * len(candidates)

    for idx, candidate in enumerate(candidates):
        smiles = candidate.representation  # type: ignore[attr-defined]
        if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
            continue
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        fp = _MORGAN_GEN.GetFingerprint(mol)
        query_fps.append(fp)
        valid_indices.append(idx)

    if not query_fps:
        return results

    # Batch compute similarities: for each query fingerprint, compute similarity to all references
    # BulkTanimotoSimilarity is already optimized, but we batch the fingerprint computation
    ref_fps_list = list(ref_fps)
    for query_idx, query_fp in enumerate(query_fps):
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, ref_fps_list)
        max_sim = float(max(sims)) if sims else 0.0
        results[valid_indices[query_idx]] = max(0.0, 1.0 - max_sim)

    return results


@register_scorer(
    name="antibiotics_motifs_filter",
    population_wise=False,
    description=(
        "Binary filter for known antibiotic structural motifs (value: 0.0 or 1.0). "
        "This scorer identifies molecules containing structural patterns commonly found in existing antibiotics, "
        "including sulfonamides, aminoglycosides, beta-lactams, tetracyclines, quinolones, and pyrimidine derivatives. "
        "It also flags molecules matching PAINS (Pan-Assay Interference Compounds) alerts. "
        "A score of 1.0 indicates the molecule does NOT contain any known antibiotic motifs or PAINS alerts, "
        "suggesting structural novelty and reduced risk of assay interference. "
        "A score of 0.0 indicates the molecule contains one or more known antibiotic motifs or PAINS alerts, "
        "which may indicate similarity to existing antibiotics or potential assay interference issues. "
        "This filter is useful for identifying structurally novel candidates that escape known antibiotic classes "
        "while avoiding problematic structural patterns."
    ),
)
def score_antibiotics_motifs_filter(
    candidates: List[Candidate],
) -> List[float]:
    """Score molecules based on absence of known antibiotic motifs.

    Args:
        candidates: List of Candidate objects with SMILES representations.

    Returns:
        List of scores where:
        - 1.0: Molecule does NOT contain any known antibiotic motifs (passes filter)
        - 0.0: Molecule contains one or more known antibiotic motifs (fails filter) OR invalid SMILES string

    The scorer checks against:
    - Custom SMARTS patterns for known antibiotic scaffolds (sulfonamides,
      beta-lactams, quinolones, tetracyclines, aminoglycosides, etc.)
    - RDKit PAINS catalog (classes A, B, and C)
    """
    results: List[float] = []

    for candidate in candidates:
        smiles = candidate.representation

        # Handle non-string or empty SMILES
        if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
            results.append(0.0)
            continue

        # Parse SMILES
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            results.append(0.0)
            continue

        # Check against custom SMARTS patterns for antibiotic motifs
        has_motif = False
        for smarts_pattern in BAD_SMARTS_LIST:
            try:
                if mol.HasSubstructMatch(smarts_pattern):
                    has_motif = True
                    break
            except Exception:
                # Skip patterns that cause matching errors
                continue

        # If already found a motif, no need to check PAINS
        if has_motif:
            results.append(0.0)
            continue

        # Check against PAINS catalog
        try:
            if PAINS_CATALOG.HasMatch(mol):
                has_motif = True
        except Exception:
            # If PAINS check fails, be conservative and don't flag
            pass

        # Return score: 1.0 if no motifs found, 0.0 if any motif found
        results.append(0.0 if has_motif else 1.0)

    return results


if __name__ == "__main__":
    from scileo_agent.core.registry import get_scorer, list_scorers

    print("Available scorers:\n", list_scorers())
    scorer = get_scorer("antibiotics_novelty")
    test = [
        Candidate(representation="CC(=O)NC1=CC=C(O)C=C1"),
        Candidate(representation="CCOC(=O)C1=CC=CC=C1"),
    ]
    print(scorer(test))
