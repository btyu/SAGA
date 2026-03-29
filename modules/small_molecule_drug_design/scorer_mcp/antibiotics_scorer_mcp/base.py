import os
from pathlib import Path
from typing import List, Optional, Sequence, Dict, Any, Tuple, Set
from loguru import logger

from .scorer_utils import BaseScorer, scorer

from rdkit import Chem
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator as rdFPGen
from rdkit.Chem import FilterCatalog

_MORGAN_GEN = rdFPGen.GetMorganGenerator(radius=2, fpSize=2048)

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


# -------- Helpers for SMARTS patterns and PAINS catalog --------
def _s(pattern: str, name: str):
    """Compile SMARTS with a friendly name; returns (name, MolPattern)."""
    m = Chem.MolFromSmarts(pattern)
    if m is None:
        raise ValueError(f"Invalid SMARTS for `{name}`: {pattern}")
    return name, m


def _build_pains_catalog():
    """Build the PAINS (Pan-Assay Interference Compounds) catalog."""
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
    return FilterCatalog.FilterCatalog(params)


# -------- SMARTS registry for antibiotic motifs --------
SMARTS_GROUPS = {
    # Sulfonamides
    "sulfonamides": [
        _s("[#6][SX4](=O)(=O)[#6]", "sulfone_general"),
        _s("[#6][SX4](=O)(=O)O[#6]", "sulfonate_ester"),
        _s("[*][SX4](=O)(=O)[NX3H2,NX3H1,NX3H0,NX2-]", "sulfonamide_h1"),
    ],
    "aminoglycosides": [
        _s("[NX3H2,NX3H1,NX3H0,NX2-]c1ccc(N)cc1", "aminoglycoside_h1"),
    ],
    # Tetracyclic skeletons (tetracycline family)
    "tetracyclic_skeletons": [
        _s("[#6;R2]~[#6;R]~[#6;R2]~[#6;R]~[#6;R2]", "tetracyclic_core"),
    ],
    "beta_lactams": [
        _s("[#7;r4]1C(=O)[#6;r4][#6;r4]1", "beta_lactams"),
    ],
    # Ring filters - refined to avoid over-filtering legitimate drug-like molecules
    "ring_filters": [
        _s("[r3,r4;!a]", "small_strained"),
        _s("[r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r20]", "macrocycle"),
        # Only filter problematic hetero-hetero adjacencies, not legitimate ones
        _s(
            "[O,S]-[O,S]", "OO_SS_adjacency"
        ),  # O-O, S-S adjacency (rare and problematic)
        _s(
            "[r;N,O,S]:[r;Cl,Br,I]", "hetero_halogen_adjacency"
        ),  # hetero-halogen adjacency inside ring
        _s("[R3,R4,R5,R6;!a]@[R3,R4,R5,R6;!a]@[R3,R4,R5,R6;!a]", "polyfused_hetero"),
    ],
    # Pyrimidine derivatives
    "pyrimidine_derivatives": [
        _s("[NH2]c1nc([NH2])ccn1", "diaminopyrimidine"),
    ],
    "quinolone": [
        _s(
            "[#8]=[#6]1:[#7H]:[#6]:[#6]:[#6]2:[#6]:1:[#6]:[#6]:[#6]:[#6]:2",
            "quinolone_deriv_1",
        ),
        _s(
            "[#8]=[#6]1:[#6]:[#6]:[#6]2:[#6](:[#7H]:1):[#6]:[#6]:[#6]:[#6]:2",
            "quinolone_deriv_2",
        ),
        _s(
            "[#8]=[#6]1:[#6]:[#6]:[#7H]:[#6]2:[#6]:1:[#6]:[#6]:[#6]:[#6]:2",
            "quinolone_deriv_3",
        ),
        _s(
            "[#6]12:[#6]:[#6]:[#6]:[#7]:[#6]:1:[#6]:[#6]:[#6]:[#6]:2",
            "quinolone_deriv_4",
        ),
        _s(
            "[#6]12:[#6]:[#7]:[#6]:[#6]:[#6]:1:[#6]:[#6]:[#6]:[#6]:2",
            "quinolone_deriv_5",
        ),
        _s(
            "[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#6]:[#6]:[#7]:[#7]:2",
            "quinolone_deriv_6",
        ),
        _s(
            "[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#6]:[#7]:[#6]:[#7]:2",
            "quinolone_deriv_7",
        ),
        _s(
            "[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#7]:[#6]:[#6]:[#7]:2",
            "quinolone_deriv_8",
        ),
        _s(
            "[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#6]:[#6]1:[#6](:[#6]:[#6]:[#6]:[#6]:1):[#7]:2",
            "quinolone_deriv_9",
        ),
        _s("[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#7H]:[#6]:[#6]:2", "quinolone_deriv_10"),
        _s("[#6]12:[#6]:[#6]:[#6]:[#6]:[#6]:1:[#7H]:[#6]:[#7]:2", "quinolone_deriv_11"),
        _s(
            "[#8]=[#6]1:[#6]:[#6]:[#7H]:[#6]2:[#7]:[#6]:[#6]:[#6]:[#6]:1:2",
            "quinolone_deriv_12",
        ),
    ],
}

# Flatten into an ordered list
BAD_SMARTS_LIST = [mol for group in SMARTS_GROUPS.values() for _, mol in group]

# Build PAINS catalog
PAINS_CATALOG = _build_pains_catalog()


class Scorer(BaseScorer):
    def __init__(self):
        # Call parent constructor to set up registry
        super().__init__()

        # If needed, load and initialize any resources (e.g., models, data) here
        # so that they can be used directly

        self.antibiotics_fps = self._ensure_antibiotic_fingerprints()

    def _load_antibiotic_smiles(self) -> List[str]:
        """Load SMILES strings of existing antibiotics from the data files.

        Loads from both combined_antibiotics.txt and broad_hts_coadd_hits.txt.
        Returns an empty list if both files are missing or unreadable.
        """
        smiles_list: List[str] = []
        scorer_data_dir = Path(os.path.join(CURRENT_FILE_DIR, "scorer_data"))

        # Load from combined_antibiotics.txt
        try:
            data_path = scorer_data_dir / "combined_antibiotics.txt"
            lines = data_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                s = line.strip()
                if s:
                    smiles_list.append(s)
        except Exception:
            pass

        # Load from broad_hts_coadd_hits.txt
        try:
            data_path = scorer_data_dir / "broad_hts_coadd_hits.txt"
            lines = data_path.read_text(encoding="utf-8").splitlines()
            for line in lines:
                s = line.strip()
                if s:
                    smiles_list.append(s)
        except Exception:
            pass

        return smiles_list

    def _ensure_antibiotic_fingerprints(self) -> Sequence:
        """Ensure fingerprints for the antibiotics reference set are computed and cached."""
        if getattr(self, "antibiotics_fps", None) is not None:
            return self.antibiotics_fps

        smiles_list = self._load_antibiotic_smiles()
        fps: List = []
        for s in smiles_list:
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            fp = _MORGAN_GEN.GetFingerprint(mol)
            fps.append(fp)

        return fps

    def _novelty_score_against_antibiotics(self, smiles: str) -> float:
        """Compute 1 - max Tanimoto similarity to any known antibiotic.

        Returns 0.0 if input SMILES is invalid. If no reference antibiotics are
        available, returns 1.0.
        """
        if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
            return 0.0
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0.0

        ref_fps = self.antibiotics_fps
        if not ref_fps:
            return 1.0

        query_fp = _MORGAN_GEN.GetFingerprint(mol)
        sims = DataStructs.BulkTanimotoSimilarity(query_fp, list(ref_fps))
        max_sim = float(max(sims)) if sims else 0.0
        return max(0.0, 1.0 - max_sim)

    @scorer(
        name="antibiotics_novelty",
        population_wise=False,
        description="Antibiotics novelty score based on whole molecule dissimilarity (value range: 0.0 to 1.0). "
        "This score is computed as (1 - maximum Tanimoto similarity) using Morgan fingerprints (radius=2, 2048 bits) on complete molecules against a reference set of existing marketed antibiotics. "
        "Score interpretation: 1.0 = completely different from all known antibiotics (maximum novelty), 0.0 = identical to a known antibiotic (no novelty). "
        "High scores (>0.7) indicate high structural novelty that may circumvent existing resistance mechanisms, while low scores (<0.3) suggest close similarity to known antibiotics. "
        "Novel antibiotics are crucial for combating antimicrobial resistance.",
    )
    def score_antibiotics_novelty(self, samples: List[str]) -> List[float]:
        """
        Calculate antibiotics novelty score based on molecular dissimilarity.

        This function computes novelty scores by:
        1. Converting SMILES to molecular fingerprints using Morgan fingerprints (radius=2, 2048 bits)
        2. Computing maximum Tanimoto similarity against reference marketed antibiotics
        3. Calculating novelty as (1 - maximum similarity)

        Optimized for batch processing: computes all fingerprints upfront and processes
        similarities in batch for better performance.

        Args:
            samples: List of input samples, where each sample is a SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; returns 0.0 for invalid samples
        """
        if not samples:
            return []

        ref_fps = self.antibiotics_fps
        if not ref_fps:
            # If no reference antibiotics, return maximum novelty for all samples
            return [1.0] * len(samples)

        # Batch compute fingerprints for all valid samples
        query_fps: List = []
        valid_indices: List[int] = []
        results: List[float] = [0.0] * len(samples)

        for idx, sample in enumerate(samples):
            if not isinstance(sample, str) or not sample or sample.strip() == "":
                continue
            mol = Chem.MolFromSmiles(sample)
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

    @scorer(
        name="antibiotics_motifs_filter",
        population_wise=False,
        description="Binary filter for known antibiotic structural motifs (value: 0.0 or 1.0). "
        "This scorer identifies molecules containing structural patterns commonly found in existing antibiotics, "
        "including sulfonamides, aminoglycosides, beta-lactams, tetracyclines, quinolones, and pyrimidine derivatives. "
        "It also flags molecules matching PAINS (Pan-Assay Interference Compounds) alerts. "
        "A score of 1.0 indicates the molecule does NOT contain any known antibiotic motifs or PAINS alerts, "
        "suggesting structural novelty and reduced risk of assay interference. "
        "A score of 0.0 indicates the molecule contains one or more known antibiotic motifs or PAINS alerts, "
        "which may indicate similarity to existing antibiotics or potential assay interference issues. "
        "This filter is useful for identifying structurally novel candidates that escape known antibiotic classes "
        "while avoiding problematic structural patterns.",
    )
    def score_antibiotics_motifs_filter(
        self, samples: List[str]
    ) -> List[float]:
        """
        Score molecules based on absence of known antibiotic motifs.

        Args:
            samples: List of input samples, where each sample is a SMILES string of a molecule

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

        for sample in samples:
            # Handle empty SMILES
            if not isinstance(sample, str) or not sample or sample.strip() == "":
                results.append(0.0)
                continue

            # Parse SMILES
            mol = Chem.MolFromSmiles(sample)
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
