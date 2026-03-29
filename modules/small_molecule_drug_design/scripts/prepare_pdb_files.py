#!/usr/bin/env python3
"""
Script to download PDB files, compute pocket centers, and prepare clean structures for docking.
"""

import os
import subprocess
import urllib.request
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser, PDBIO, Select

# Target definitions: PDB ID -> (local_name, ligand_resname)
TARGETS = {
    "6CM4": ("DRD2.pdb", "8NU"),  # Risperidone
    "6TCU": ("GSK3B.pdb", "N1Q"),  # Ligand "1" 
    "2O0U": ("JNK3.pdb", "C0M"),  # N-{3-cyano...}
    "3MXF": ("BRD4.pdb", "JQ1"),  # JQ1
    "7VU6": ("MPRO.pdb", "7YY")
}


class ProteinSelect(Select):
    """Select only protein atoms, excluding ligands, waters, and ions."""

    def accept_residue(self, residue):
        # Keep standard amino acids only
        if residue.get_resname() in [
                'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS',
                'ILE', 'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP',
                'TYR', 'VAL'
        ]:
            return True
        # Exclude ligands, waters (HOH), and other heteroatoms
        return False


class LigandSelect(Select):
    """Select only the first instance of the specified ligand residue."""

    def __init__(self, ligand_resname):
        self.ligand_resname = ligand_resname
        self.found_first = False

    def accept_residue(self, residue):
        if residue.get_resname().strip() == self.ligand_resname:
            if not self.found_first:
                self.found_first = True
                return True
        return False


def create_directories():
    """Create necessary directories."""
    Path("data/pdb").mkdir(parents=True, exist_ok=True)
    Path("data/ligands").mkdir(parents=True, exist_ok=True)
    print("Created data/pdb and data/ligands directories")


def download_pdb_file(pdb_id: str, local_filename: str):
    """Download PDB file from RCSB."""
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    # Download original file with 'original_' prefix to avoid conflicts
    original_filename = f"original_{local_filename}"
    local_path = f"data/pdb/{original_filename}"

    print(f"Downloading {pdb_id} -> {original_filename}")
    urllib.request.urlretrieve(url, local_path)
    return local_path


def extract_ligand_to_sdf(pdb_path: str, ligand_resname: str, output_sdf: str):
    """Extract ligand from PDB and save as SDF file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)
    
    # Create selector for the specific ligand
    selector = LigandSelect(ligand_resname)
    
    # Write ligand to temporary PDB file
    temp_pdb = f"temp_ligand_{ligand_resname}.pdb"
    io = PDBIO()
    io.set_structure(structure)
    io.save(temp_pdb, selector)
    
    # Convert PDB to SDF using Open Babel
    try:
        cmd = [
            "obabel",
            temp_pdb,
            "-O",
            output_sdf
        ]
        subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"Extracted ligand {ligand_resname} to {output_sdf}")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not convert ligand to SDF: {e}")
        print("Open Babel may not be installed. Install with: conda install -c conda-forge openbabel")
    finally:
        # Clean up temporary file
        if os.path.exists(temp_pdb):
            os.remove(temp_pdb)


def add_hydrogens(input_pdb: str, output_pdb: str):
    """Add missing atoms and hydrogens using pdb4amber."""
    cmd = [
        "pdb4amber",
        # "--add-missing-atoms",
        "--reduce",  # place hydrogens via Reduce
        "-i",
        input_pdb,
        "-o",
        output_pdb
    ]
    print(f"Adding hydrogens to {input_pdb} -> {output_pdb}")
    try:
        subprocess.run(
            cmd,
            capture_output=True,  # capture stdout/stderr in Python
            text=True,
            check=True)
        print(f"✅ Successfully added hydrogens: {output_pdb}")
    except subprocess.CalledProcessError as e:
        print(f"❌ pdb4amber failed (exit {e.returncode}):")
        print(e.stderr)


def compute_pocket_center(pdb_path: str,
                          ligand_resname: str) -> tuple[float, float, float]:
    """Compute geometric center of the first ligand instance."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("", pdb_path)
    coords = []
    first_residue = None

    for atom in structure.get_atoms():
        res = atom.get_parent()
        if res.get_resname().strip() == ligand_resname:
            if first_residue is None:
                # First instance of this ligand
                first_residue = res
                coords.append(atom.get_coord())
            elif res == first_residue:
                # Still in the first residue, continue collecting atoms
                coords.append(atom.get_coord())
            else:
                # We've moved to a different residue (next instance), stop
                break

    if not coords:
        raise ValueError(f"No ligand {ligand_resname} found in {pdb_path}")

    coords = np.stack(coords, axis=0)
    center = coords.mean(axis=0)
    return float(center[0]), float(center[1]), float(center[2])


def clean_protein_structure(input_pdb: str, output_pdb: str):
    """Remove ligands, waters, and ions, keeping only protein atoms."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", input_pdb)

    # Create selector to keep only protein atoms
    selector = ProteinSelect()

    # Write cleaned protein structure
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, selector)
    print(f"Cleaned protein saved: {output_pdb}")


def main():
    """Main function to download files, compute centers, and prepare clean structures."""
    create_directories()

    centers = {}

    for pdb_id, (local_name, ligand_resname) in TARGETS.items():
        target_name = local_name.replace('.pdb', '')

        # Download original PDB file
        original_pdb_path = download_pdb_file(pdb_id, local_name)

        # Extract ligand to SDF file
        ligand_sdf_path = f"data/ligands/{target_name}.sdf"
        extract_ligand_to_sdf(original_pdb_path, ligand_resname, ligand_sdf_path)

        # Compute pocket center from original structure (with ligand)
        try:
            center = compute_pocket_center(original_pdb_path, ligand_resname)
            centers[target_name] = center
            print(f"Computed center for {local_name}: {center}")
        except Exception as e:
            print(f"Error computing center for {local_name}: {e}")
            continue

        # Create clean protein structure (remove ligands and waters)
        clean_pdb_path = f"data/pdb/{target_name}.pdb"
        clean_protein_structure(original_pdb_path, clean_pdb_path)

        # Add hydrogens to the clean protein structure
        try:
            hydrogenated_pdb_path = f"data/pdb/{target_name}_H.pdb"
            add_hydrogens(clean_pdb_path, hydrogenated_pdb_path)
            # move the hydrogenated pdb to the original pdb file
            os.rename(hydrogenated_pdb_path, clean_pdb_path)

            # Clean up intermediate files created by pdb4amber
            cleanup_files = [
                f"data/pdb/{target_name}_H_sslink",
                f"data/pdb/{target_name}_H_nonprot.pdb",
                f"data/pdb/{target_name}_H_renum.txt"
            ]
            for cleanup_file in cleanup_files:
                if os.path.exists(cleanup_file):
                    os.remove(cleanup_file)
                    print(f"Removed temporary file: {cleanup_file}")

        except Exception as e:
            print(f"Warning: Could not add hydrogens to {target_name}: {e}")
            print("Continuing with original cleaned structure...")

        # Clean up original downloaded PDB file
        if os.path.exists(original_pdb_path):
            os.remove(original_pdb_path)
            print(f"Removed original file: {original_pdb_path}")

    # Print results for unidock_scorer.py
    print("\n" + "=" * 50)
    print("Pocket centers for unidock_scorer.py:")
    print("=" * 50)

    for target_name, center in centers.items():
        x, y, z = center
        print(
            f'ProteinTarget(protein_name="{target_name}", pdb_path="data/pdb/{target_name}.pdb", pocket_center=({x:.3f}, {y:.3f}, {z:.3f})),'
        )


if __name__ == "__main__":
    main()
