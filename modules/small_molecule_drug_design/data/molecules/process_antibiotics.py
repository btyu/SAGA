#!/usr/bin/env python3
"""
Process existing_antibiotics.txt to clean up the SMILES:
1. Split multi-component entries (separated by '.')
2. Remove ions and small counterions
3. Remove very small compounds
"""

from rdkit import Chem
import sys

def is_ion_or_salt(smiles):
    """Check if SMILES is an ion or common salt/counterion"""
    # Common ions and small molecules to remove
    ions_and_salts = [
        'O',  # water
        'OO',  # hydrogen peroxide
        'Cl',  # chloride
        'Br',  # bromide
        'I',  # iodide
        '[Na+]', '[K+]', '[Ca+2]', '[Mg+2]', '[NH4+]', '[Li+]',  # cations
        '[Cl-]', '[Br-]', '[I-]', '[F-]',  # halide anions
        '[O-]', '[OH-]',  # hydroxide/oxide
        'O=S(=O)(O)O', 'O=S(=O)([O-])O', 'O=S(=O)([O-])[O-]',  # sulfuric acid/sulfate
        'O=P(=O)(O)O', 'O=P(=O)([O-])O', 'O=P(=O)([O-])[O-]',  # phosphoric acid/phosphate
        'O=C(O)O', 'O=C([O-])[O-]',  # carbonic acid/carbonate
        'C(C(=O)O)C(CC(=O)O)(C(=O)O)O',  # citric acid
        'O=C(O)C(O)C(O)C(O)C(O)C(O)CO',  # gluconic acid
        'O=C(O)/C=C/C(=O)O',  # fumaric acid
        'CC(O)C(=O)O',  # lactic acid
        'CCO',  # ethanol
        'CCOC',  # diethyl ether fragments
    ]

    smiles = smiles.strip()

    # Check exact matches
    if smiles in ions_and_salts:
        return True

    # Check for charge brackets (ions)
    if '[' in smiles and ('+' in smiles or '-' in smiles):
        # Count if it's purely an ion (small bracketed charged species)
        if smiles.count('[') <= 2 and len(smiles) < 15:
            return True

    return False

def count_heavy_atoms(smiles):
    """Count non-hydrogen atoms in molecule"""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return 0
        return mol.GetNumHeavyAtoms()
    except:
        return 0

def process_smiles(smiles_line):
    """Process a SMILES line: split components and filter"""
    smiles_line = smiles_line.strip()
    if not smiles_line:
        return []

    # Split by '.' to get individual components
    components = smiles_line.split('.')

    valid_components = []
    for comp in components:
        comp = comp.strip()
        if not comp:
            continue

        # Skip ions and common salts
        if is_ion_or_salt(comp):
            continue

        # Check heavy atom count (skip very small molecules)
        heavy_atoms = count_heavy_atoms(comp)
        if heavy_atoms < 5:  # threshold for "very small"
            continue

        # Validate SMILES
        try:
            mol = Chem.MolFromSmiles(comp)
            if mol is not None:
                # Canonicalize SMILES
                canonical = Chem.MolToSmiles(mol)
                valid_components.append(canonical)
        except:
            # Skip invalid SMILES
            continue

    return valid_components

def main():
    input_file = '/home/tsa87/SciLeoAgent/modules/small_molecule_drug_design/data/molecules/temp_combined_all.txt'
    output_file = '/home/tsa87/SciLeoAgent/modules/small_molecule_drug_design/data/molecules/combined_antibiotics.txt'

    print(f"Reading from: {input_file}")

    all_smiles = []
    unique_smiles = set()

    with open(input_file, 'r') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue

            # Process the line
            valid_components = process_smiles(line)

            for smiles in valid_components:
                if smiles not in unique_smiles:
                    unique_smiles.add(smiles)
                    all_smiles.append(smiles)

    print(f"Processed {line_num} lines")
    print(f"Found {len(all_smiles)} unique valid molecules")

    # Write output
    with open(output_file, 'w') as f:
        for smiles in all_smiles:
            f.write(smiles + '\n')

    print(f"Wrote cleaned SMILES to: {output_file}")
    print(f"Removed duplicates, ions, and small compounds")

if __name__ == '__main__':
    main()
