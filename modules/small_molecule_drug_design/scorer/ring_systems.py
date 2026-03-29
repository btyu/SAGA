from rdkit import Chem
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Optional


class RingSystemFinder:
    """Extract ring systems from molecules by cleaving linker bonds."""
    
    def __init__(self):
        """
        Initialize substructure search objects to identify key functionality.
        """
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
            # Default to scorer_mcp directory structure
            module_root = Path(__file__).resolve().parent.parent.parent
            ring_system_csv = module_root / "scorer_mcp" / "ring_score_scorer_mcp" / "scorer_data" / "chembl_ring_systems.csv"
            if not ring_system_csv.exists():
                # Fallback: try relative to scorer directory
                ring_system_csv = Path(__file__).resolve().parent.parent.parent.parent / "scorer_mcp" / "ring_score_scorer_mcp" / "scorer_data" / "chembl_ring_systems.csv"
        
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


def ring_systems_min_score(freq_list: List[Tuple[str, int]]) -> int:
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


