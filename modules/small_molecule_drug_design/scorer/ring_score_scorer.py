from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate
from typing import List, Optional
import logging
import math
from pathlib import Path
from rdkit import Chem

from modules.small_molecule_drug_design.scorer.ring_systems import RingSystemLookup, ring_systems_min_score

# silence rdkit warnings
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')


@register_scorer_class
class RingScoreScorers:
    """Collection of ring score scoring functions with 0-1 scoring."""

    def __init__(self):
        # Initialize RingSystemLookup with reference data
        # Find the CSV file relative to the module
        module_root = Path(__file__).resolve().parent.parent.parent
        ring_system_csv = module_root / "scorer_mcp" / "ring_score_scorer_mcp" / "scorer_data" / "chembl_ring_systems.csv"
        
        if not ring_system_csv.exists():
            # Fallback: try alternative path
            ring_system_csv = Path(__file__).resolve().parent.parent.parent.parent / "scorer_mcp" / "ring_score_scorer_mcp" / "scorer_data" / "chembl_ring_systems.csv"
        
        self._ring_lookup = RingSystemLookup(ring_system_csv=str(ring_system_csv))

    def _normalize_ring_score(self, min_freq: int) -> float:
        """Normalize ring frequency to 0-1 scale where 1 = normal, 0 = weird.

        Args:
            min_freq: Minimum frequency of any ring system in the molecule

        Returns:
            Normalized score between 0.0 and 1.0
        """
        if min_freq == -1:
            # No rings found - no penalty, score is best
            return 1.0
        if min_freq == 0:
            # Not in database = weird
            return 0.0
        if min_freq >= 100:
            # Reasonably common = normal (no penalty)
            return 1.0
        if min_freq <= 10:
            # Very rare = weird (penalize)
            return 0.0

        # Linear interpolation on log scale between 10-100
        # Only penalize reasonably rare rings (frequency < 100)
        log_freq = math.log(min_freq + 1)
        log_max = math.log(100 + 1)
        log_min = math.log(10 + 1)
        return (log_freq - log_min) / (log_max - log_min)

    @register_scorer(
        name="ring_score",
        population_wise=False,
        description=
        "Ring system frequency score (value range: 0.0 to 1.0). "
        "This score identifies unusual or 'weird' ring systems by comparing them to ChEMBL database frequencies. "
        "Ring systems are extracted from molecules by cleaving linker bonds, and each ring system's frequency in ChEMBL is looked up. "
        "The score uses the minimum frequency (most unusual ring) to assess overall ring system quality. "
        "High scores (>0.8) indicate molecules with no rings (no penalty) or reasonably common ring systems (frequency ≥100) that are frequently found in known drugs and drug-like compounds, while low scores (<0.3) suggest molecules containing very rare or novel ring systems (frequency ≤10) that may have unknown properties or synthetic challenges. "
        "A score of 0.0 indicates the presence of ring systems not found in the ChEMBL database, which may represent highly novel or problematic structures. "
        "This metric helps identify compounds with unusual ring systems that may require additional evaluation or optimization.",
    )
    def score_ring_score(
            self, candidates: List[Candidate]) -> List[float]:
        """
        Calculate ring system frequency score.

        Returns normalized scores between 0.0 and 1.0 based on ring system frequencies:
        - Score = 1.0: No rings found (no penalty) or all ring systems have frequency ≥100 (reasonably common)
        - Score = 0.0: Contains ring systems with frequency ≤10 or not in database (very rare/weird) OR invalid SMILES

        The score identifies unusual ring systems by extracting all ring systems from the molecule
        and looking up their frequency in ChEMBL. The minimum frequency (most unusual ring) is used
        to assess overall ring system quality.

        Returns:
            List of scores (0.0-1.0), returns 0.0 for invalid SMILES
        """
        results = []
        for candidate in candidates:
            smiles = candidate.representation
            if not isinstance(smiles, str) or not smiles or smiles.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                results.append(0.0)
            else:
                # Get ring systems and their frequencies
                freq_list = self._ring_lookup.process_mol(mol)
                # Get minimum frequency (most unusual ring)
                min_freq = ring_systems_min_score(freq_list)
                # Normalize to 0-1 scale
                score = self._normalize_ring_score(min_freq)
                results.append(score)
        return results


if __name__ == "__main__":
    from scileo_agent.core.registry import get_scorer, list_scorers
    from scileo_agent.core.data_models import Candidate

    # List all available scorers
    print("Available scorers:\n", list_scorers())
    print()

    # Test the scorer with diverse molecules
    test_smiles = [
        "c1ccccc1",  # Benzene - common ring, should score well
        "CCO",  # Ethanol - no rings, should score 0.5
        "C1CCC2CCCCC2C1",  # Decalin - should check frequency
        "invalid_smiles",  # Invalid SMILES to test error handling
    ]
    candidates = [Candidate(representation=smiles) for smiles in test_smiles]

    # Get the scorer
    ring_scorer = get_scorer("ring_score")

    # Test the scorer
    print("Test molecules:")
    for i, smiles in enumerate(test_smiles):
        print(f"{i+1}. {smiles}")
    print()

    print("Ring scores:", ring_scorer(candidates))

