from pathlib import Path

from Bio.PDB import PDBParser
from rdkit import Chem
import numpy as np
from typing import List, Tuple, Dict, Any

def get_ligand_coords_from_sdf(sdf_file: str) -> List[Tuple[str, np.ndarray]]:
    coords = []
    try:
        suppl = Chem.SDMolSupplier(sdf_file, removeHs=False)
        mol = suppl[0]  # first conformer
        conf = mol.GetConformer()
        for i, atom in enumerate(mol.GetAtoms()):
            pos = conf.GetAtomPosition(i)
            coords.append((atom.GetSymbol(), np.array([pos.x, pos.y, pos.z])))
    except Exception as e:
        print(f"Error reading SDF file {sdf_file}: {e}")
    return coords  # list of (atom_symbol, [x,y,z])

def generate_residue_map(
    pdb_file: str,
    ligand_atoms: List[Tuple[str, np.ndarray]],
    distance_cutoff: float = 5.0
) -> List[Dict[str, Any]]:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)

    residue_map = []
    
    if len(ligand_atoms) == 0:
        return residue_map
    try:
        for model in structure:
            for chain in model:
                for residue in chain:
                    if residue.id[0] != " ":  # skip heteroatoms/water
                        continue
                    
                    # Track nearest ligand atom
                    nearest_atom = None
                    nearest_dist = 1e6

                    # Find the nearest docked ligand atom for each protein residue
                    for prot_atom in residue:
                        for lig_atom, lig_coord in ligand_atoms:
                            dist = np.linalg.norm(lig_coord - prot_atom.get_coord())
                            if dist < nearest_dist:
                                nearest_dist = dist
                                nearest_atom = (lig_atom, lig_coord)

                    if nearest_dist < distance_cutoff:  # only keep docking pocket residues
                        residue_map.append({
                            "residue_id": residue.id[1],
                            "residue_name": f"{residue.get_resname()}{residue.id[1]}",
                            "residue_chain": chain.id,
                            "docked_ligand_atom": nearest_atom[0],
                            "docked_ligand_atom_distance_angstrom": nearest_dist,
                            "docked_ligand_atom_xyz": nearest_atom[1].tolist(),
                        })
        residue_map = sorted(residue_map, key=lambda x: x['residue_id'])
    except Exception as e:
        print(f"Error generating residue map: {e}")
        
    return residue_map

if __name__ == "__main__":
    protein_pdb_dirpath = Path(__file__).resolve().parent.parent / 'data' / 'pdb'
    protein_pdb_filepath = str(protein_pdb_dirpath / 'DRD2.pdb')

    ligand_sdf_dirpath = Path(__file__).resolve().parent.parent / 'docked_poses'
    ligand_sdf_filepath = str(ligand_sdf_dirpath  / 'docked_0.sdf')

    ligand_coords = get_ligand_coords_from_sdf(ligand_sdf_filepath)
    print(f"Ligand coordinates for {ligand_sdf_filepath}: {ligand_coords}")

    residue_map = generate_residue_map(protein_pdb_filepath, ligand_coords, distance_cutoff=12.0)
    print(f"Residue map for {protein_pdb_filepath}: {residue_map}")
    
    print(f"Number of ligand atoms: {len(ligand_coords)}")
    print(f"Number of residues in docking pocket: {len(residue_map)}")