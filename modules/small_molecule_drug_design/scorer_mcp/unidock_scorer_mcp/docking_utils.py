"""Docking utilities for UniDock scoring - self-contained for Docker execution."""

import os
import sys
from pathlib import Path
from io import StringIO
from typing import Dict, List, Tuple, Optional
import xml.etree.ElementTree as ET
import logging

import numpy as np
import parmed
from Bio.PDB import PDBParser
from rdkit import Chem
from plip.structure.preparation import PDBComplex, logger as PLIP_LOGGER
from plip.exchange.report import StructureReport
from plip.basic import config as PLIP_CONFIG

PLIP_LOGGER.setLevel(logging.ERROR)
PLIP_LOGGER.propagate = False

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def get_ligand_coords_from_sdf(sdf_file: str) -> List[Tuple[str, np.ndarray]]:
    """Extract ligand coordinates from SDF file.

    Args:
        sdf_file: Path to SDF file

    Returns:
        List of (atom_symbol, coordinates) tuples
    """
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
) -> List[Dict]:
    """Generate residue map for protein-ligand interactions.

    Args:
        pdb_file: Path to protein PDB file
        ligand_atoms: List of (atom_symbol, coordinates) tuples
        distance_cutoff: Distance cutoff for identifying pocket residues

    Returns:
        List of dictionaries containing residue information
    """
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


def plip_analyze_single_frame(
    pdbpath: str,
    mol_name: str = "MOL",
    add_hydrogen: bool = False,
    resnr_renum: Optional[Dict[int, int]] = None
) -> Dict[str, int]:
    """Analyze protein-ligand interactions using PLIP.

    Args:
        pdbpath: Path to PDB file
        mol_name: Ligand molecule name
        add_hydrogen: Whether to add hydrogens
        resnr_renum: Residue number renumbering dictionary

    Returns:
        Dictionary with interaction type as key and count as value
    """
    if add_hydrogen:
        PLIP_CONFIG.NOHYDRO = False
    else:
        PLIP_CONFIG.NOHYDRO = True

    pdb = PDBComplex()
    pdb.load_pdb(str(pdbpath))
    pdb.analyze()
    report = StructureReport(pdb)

    tmp = sys.stdout
    xmlstr = StringIO()
    sys.stdout = xmlstr
    report.write_xml(True)
    sys.stdout = tmp
    xmlstr.seek(0)

    xmlobj = ET.fromstring(xmlstr.read())

    binding_sites = xmlobj.findall("./bindingsite")
    try:
        bs = [bs for bs in binding_sites if bs.findall("identifiers/longname")[0].text == mol_name][0]
        itypes = bs.findall("interactions/")
    except IndexError:
        itypes = []
        bs = [bs for bs in binding_sites]
        for site in bs:
            itypes += list(site.findall("interactions/"))

    interact_count_frame = {}
    for itype in itypes:
        for item in itype:
            name = item.tag
            restype = item.find("restype").text
            resnr = item.find("resnr").text
            if resnr_renum is not None:
                resnr = resnr_renum[int(resnr)]
            chain = item.find("reschain").text
            sig = f"{name}/{restype}/{resnr}/{chain}"
            cnt = interact_count_frame.get(sig, 0)
            if cnt == 0:
                interact_count_frame.update({sig: cnt+1})

    return interact_count_frame


def mpro_plip_score_computation(sdf_path: str, debug: bool = False) -> Dict[str, int]:
    """Analyze MPRO-ligand complex and return interaction scores.

    Args:
        sdf_path: Path to docked ligand SDF file
        debug: Whether to print debug information

    Returns:
        Dictionary with three interaction identifiers mapped to 1 (present) or 0 (absent):
        "hydrogen_bond/HIS/161/A", "hydrogen_bond/GLU/164/A", "pi_stack/HIS/39/A"
    """
    # Load MPRO protein structure
    mpro_pdb_path = os.path.join(CURRENT_FILE_DIR, "scorer_data", "pdb", "MPRO.pdb")
    protein_structure = parmed.load_file(mpro_pdb_path)

    supplier = Chem.SDMolSupplier(str(sdf_path))

    targets = [
        "hydrogen_bond/HIS/161/A",
        "hydrogen_bond/GLU/164/A",
        "pi_stack/HIS/39/A",
    ]

    best_presence: Optional[Dict[str, int]] = None
    best_count = -1
    best_report: Optional[Dict] = None
    found_any = False

    for i, mol in enumerate(supplier):
        if mol is None:
            continue
        found_any = True

        ligand_pdb_path = str(sdf_path).replace('.sdf', f'_{i}.pdb')
        Chem.MolToPDBFile(mol, ligand_pdb_path)

        ligand_structure = parmed.load_file(ligand_pdb_path)
        complex_structure = protein_structure + ligand_structure
        complex_pdb_path = ligand_pdb_path.replace('.pdb', '_complex.pdb')
        complex_structure.save(complex_pdb_path, overwrite=True)

        report = plip_analyze_single_frame(complex_pdb_path, mol_name='UNL')

        presence = {key: (1 if key in report else 0) for key in targets}
        count = sum(presence.values())

        if debug:
            present = [k for k, v in presence.items() if v]
            print(
                f"[PLIP] mol_index={i} interaction_count={count} present={present} presence={presence}"
            )

        if count > best_count:
            best_count = count
            best_presence = presence
            best_report = report

    if not found_any:
        if debug:
            raise ValueError("No ligand found in SDF file")
        return {
            "hydrogen_bond/HIS/161/A": 0,
            "hydrogen_bond/GLU/164/A": 0,
            "pi_stack/HIS/39/A": 0,
        }

    if debug and best_report is not None:
        print(best_report)

    assert best_presence is not None
    return best_presence
