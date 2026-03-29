import os
import math
from typing import List, Optional
from pathlib import Path
from loguru import logger

from rdkit import Chem
import pandas as pd

# silence rdkit warnings
from rdkit import RDLogger

RDLogger.DisableLog('rdApp.*')

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class RingSystemFinder:
    """Extract ring systems from molecules by cleaving linker bonds."""
    
    def __init__(self):
        """Initialize substructure search objects to identify key functionality."""
        self.ring_db_pat = Chem.MolFromSmarts("[#6R,#18R]=[OR0,SR0,CR0,NR0]")
        self.ring_atom_pat = Chem.MolFromSmarts("[R]")

    def tag_bonds_to_preserve(self, mol):
        """Assign the property "protected" to all ring carbonyls, etc.
        
        Args:
            mol: input molecule
        """
        for bnd in mol.GetBonds():
            bnd.SetBoolProp("protected", False)
        for match in mol.GetSubstructMatches(self.ring_db_pat):
            bgn, end = match
            bnd = mol.GetBondBetweenAtoms(bgn, end)
            bnd.SetBoolProp("protected", True)

    @staticmethod
    def cleave_linker_bonds(mol):
        """Cleave bonds that are not in rings and not protected.
        
        Args:
            mol: input molecule
            
        Returns:
            Fragmented molecule with linker bonds cleaved
        """
        frag_bond_list = []
        for bnd in mol.GetBonds():
            if not bnd.IsInRing() and not bnd.GetBoolProp("protected") and bnd.GetBondType() == Chem.BondType.SINGLE:
                frag_bond_list.append(bnd.GetIdx())

        if len(frag_bond_list):
            frag_mol = Chem.FragmentOnBonds(mol, frag_bond_list)
            Chem.SanitizeMol(frag_mol)
            return frag_mol
        else:
            return mol

    def cleanup_fragments(self, mol):
        """Split a molecule containing multiple ring systems into individual ring systems.
        
        Args:
            mol: input molecule
            
        Returns:
            A list of SMILES corresponding to individual ring systems
        """
        frag_list = Chem.GetMolFrags(mol, asMols=True)
        ring_system_smiles_list = []
        for frag in frag_list:
            if frag.HasSubstructMatch(self.ring_atom_pat):
                for atm in frag.GetAtoms():
                    if atm.GetAtomicNum() == 0:
                        atm.SetAtomicNum(1)
                        atm.SetIsotope(0)
                # Convert explicit Hs to implicit
                frag = Chem.RemoveAllHs(frag)
                ring_system_smiles_list.append(Chem.MolToSmiles(frag))
        return ring_system_smiles_list

    def find_ring_systems(self, mol):
        """Find the ring systems for an RDKit molecule.
        
        Args:
            mol: input molecule
            
        Returns:
            A list of SMILES corresponding to individual ring systems
        """
        self.tag_bonds_to_preserve(mol)
        frag_mol = self.cleave_linker_bonds(mol)
        ring_system_smiles_list = self.cleanup_fragments(frag_mol)
        return ring_system_smiles_list


class RingSystemLookup:
    """Lookup ring system frequencies from ChEMBL database."""
    
    def __init__(self, ring_system_csv: Optional[str] = None):
        """
        Initialize the lookup table.
        
        Args:
            ring_system_csv: Path to CSV file with ring smiles and frequency.
                            If None, uses default path relative to module.
        """
        if ring_system_csv is None:
            # Default to scorer_data directory
            ring_system_csv = os.path.join(CURRENT_FILE_DIR, "scorer_data", "chembl_ring_systems.csv")
            if not os.path.exists(ring_system_csv):
                # Fallback: try alternative path
                ring_system_csv = os.path.join(CURRENT_FILE_DIR, "..", "..", "scorer_mcp", "ring_score_scorer_mcp", "scorer_data", "chembl_ring_systems.csv")
        
        ring_df = pd.read_csv(ring_system_csv)
        self.ring_dict = dict(ring_df[["ring_system", "count"]].values)

    def process_mol(self, mol):
        """Find ring systems in an RDKit molecule.
        
        Args:
            mol: input molecule
            
        Returns:
            List of tuples (ring_system_smiles, frequency) for each ring system
        """
        if mol:
            ring_system_finder = RingSystemFinder()
            ring_system_list = ring_system_finder.find_ring_systems(mol)
            return [(x, self.ring_dict.get(x) or 0) for x in ring_system_list]
        else:
            return []

    def process_smiles(self, smi: str):
        """Find ring systems from a SMILES string.
        
        Args:
            smi: input SMILES
            
        Returns:
            List of tuples (ring_system_smiles, frequency) for each ring system
        """
        mol = Chem.MolFromSmiles(smi)
        return self.process_mol(mol)


def ring_systems_min_score(freq_list):
    """Get the minimum frequency (most unusual ring) from a list of ring systems.
    
    Args:
        freq_list: List of tuples (ring_system_smiles, frequency)
        
    Returns:
        Minimum frequency, or -1 if no rings found
    """
    if len(freq_list):
        res = min([x[1] for x in freq_list])
    else:
        res = -1
    return res


class Scorer(BaseScorer):
    """Collection of ring score scoring functions with 0-1 scoring."""

    def __init__(self):
        super().__init__()

        # Initialize RingSystemLookup with reference data
        ring_system_csv = os.path.join(CURRENT_FILE_DIR, "scorer_data", "chembl_ring_systems.csv")
        if not os.path.exists(ring_system_csv):
            # Fallback: try alternative path
            ring_system_csv = os.path.join(CURRENT_FILE_DIR, "..", "..", "scorer_mcp", "ring_score_scorer_mcp", "scorer_data", "chembl_ring_systems.csv")
        
        self._ring_lookup = RingSystemLookup(ring_system_csv=ring_system_csv)

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

    @scorer(
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
    def score_ring_score(self, samples: List[str]) -> List[float]:
        """
        Calculate ring system frequency score.

        Returns normalized scores between 0.0 and 1.0 based on ring system frequencies:
        - Score = 1.0: No rings found (no penalty) or all ring systems have frequency ≥100 (reasonably common)
        - Score = 0.0: Contains ring systems with frequency ≤10 or not in database (very rare/weird) OR invalid SMILES

        The score identifies unusual ring systems by extracting all ring systems from the molecule
        and looking up their frequency in ChEMBL. The minimum frequency (most unusual ring) is used
        to assess overall ring system quality.

        Args:
            samples: List of input samples, where each sample is a SMILES string of a molecule

        Returns:
            List of float scores, returns 0.0 for invalid samples
        """
        results = []
        for sample in samples:
            if not isinstance(sample, str) or not sample or sample.strip() == "":
                results.append(0.0)
                continue
            mol = Chem.MolFromSmiles(sample)
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

