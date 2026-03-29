#!/usr/bin/env python3
"""
Merge all antibiotic databases:
1. Original 92 antibiotics from names CSV
2. Extended 163 antibiotics from PubChem searches  
3. 566 antibiotics from existing_antibiotics.txt
4. Create comprehensive database for novelty analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem
import pubchempy as pcp


def load_existing_antibiotics():
    """Load the 566 antibiotics from existing_antibiotics.txt"""
    print("Loading existing antibiotics from file...")

    with open("combined_antibiotics.txt", "r") as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    print(f"Found {len(smiles_list)} antibiotics in existing file")

    # Create DataFrame with basic info
    data = []
    for i, smiles in enumerate(smiles_list):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                data.append({
                    'name': f'antibiotic_{i+1}',
                    'class': 'existing_antibiotic',
                    'smiles': smiles,
                    'status': 'ok',
                    'source': 'existing_file'
                })
        except Exception as e:
            print(f"Error processing SMILES {i+1}: {e}")

    return pd.DataFrame(data)


def merge_all_databases():
    """Merge all antibiotic databases"""
    print("Merging all antibiotic databases...")

    # Load existing databases
    try:
        original_df = pd.read_csv("antibiotics_library_complete.csv")
        print(f"Loaded {len(original_df)} from original database")
    except:
        print("Original database not found, skipping...")
        original_df = pd.DataFrame()

    try:
        extended_df = pd.read_csv(
            "extended_antibiotics_database_with_class.csv")
        print(f"Loaded {len(extended_df)} from extended database")
    except:
        print("Extended database not found, skipping...")
        extended_df = pd.DataFrame()

    # Load existing antibiotics
    existing_df = load_existing_antibiotics()
    print(f"Loaded {len(existing_df)} from existing antibiotics file")

    # Merge all databases
    all_dfs = []
    if not original_df.empty:
        original_df['source'] = 'original'
        all_dfs.append(original_df)

    if not extended_df.empty:
        extended_df['source'] = 'extended'
        all_dfs.append(extended_df)

    if not existing_df.empty:
        all_dfs.append(existing_df)

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)

        # Remove duplicates based on SMILES
        print(f"Before deduplication: {len(combined_df)} compounds")
        combined_df = combined_df.drop_duplicates(subset=['smiles'],
                                                  keep='first')
        print(f"After deduplication: {len(combined_df)} unique compounds")

        # Add molecular properties
        print("Adding molecular properties...")
        combined_df = add_molecular_properties(combined_df)

        return combined_df
    else:
        print("No databases found!")
        return pd.DataFrame()


def add_molecular_properties(df):
    """Add molecular properties to the dataframe"""
    print("Calculating molecular properties...")

    properties = []
    for i, row in df.iterrows():
        if i % 100 == 0:
            print(f"Processing {i}/{len(df)}...")

        try:
            mol = Chem.MolFromSmiles(row['smiles'])
            if mol:
                props = {
                    'molecular_weight':
                    Chem.rdMolDescriptors.CalcExactMolWt(mol),
                    'logp':
                    Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
                    'tpsa':
                    Chem.rdMolDescriptors.CalcTPSA(mol),
                    'hbd':
                    Chem.rdMolDescriptors.CalcNumHBD(mol),
                    'hba':
                    Chem.rdMolDescriptors.CalcNumHBA(mol),
                    'rotatable_bonds':
                    Chem.rdMolDescriptors.CalcNumRotatableBonds(mol),
                    'aromatic_rings':
                    Chem.rdMolDescriptors.CalcNumAromaticRings(mol),
                    'heavy_atoms':
                    mol.GetNumHeavyAtoms()
                }
            else:
                props = {
                    key: np.nan
                    for key in [
                        'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
                        'rotatable_bonds', 'aromatic_rings', 'heavy_atoms'
                    ]
                }
        except Exception as e:
            print(
                f"Error calculating properties for {row.get('name', 'unknown')}: {e}"
            )
            props = {
                key: np.nan
                for key in [
                    'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba',
                    'rotatable_bonds', 'aromatic_rings', 'heavy_atoms'
                ]
            }

        properties.append(props)

    # Add properties to dataframe
    for prop in properties[0].keys():
        df[prop] = [p[prop] for p in properties]

    return df


def create_comprehensive_analysis():
    """Create comprehensive analysis of the merged database"""
    print("Creating comprehensive analysis...")

    # Load merged database
    df = pd.read_csv("comprehensive_antibiotics_database.csv")

    # Basic statistics
    print(f"\n=== COMPREHENSIVE ANTIBIOTICS DATABASE ===")
    print(f"Total compounds: {len(df)}")
    print(f"Unique SMILES: {df['smiles'].nunique()}")
    print(f"Sources: {df['source'].value_counts().to_dict()}")
    print(f"Classes: {df['class'].value_counts().to_dict()}")

    # Molecular property statistics
    print(f"\n=== MOLECULAR PROPERTIES ===")
    numeric_cols = [
        'molecular_weight', 'logp', 'tpsa', 'hbd', 'hba', 'rotatable_bonds',
        'aromatic_rings', 'heavy_atoms'
    ]
    for col in numeric_cols:
        if col in df.columns:
            print(
                f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}"
            )

    # Lipinski's Rule of Five analysis
    print(f"\n=== LIPINSKI'S RULE OF FIVE ===")
    lipinski_violations = 0
    for _, row in df.iterrows():
        violations = 0
        if row['molecular_weight'] > 500:
            violations += 1
        if row['logp'] > 5:
            violations += 1
        if row['hbd'] > 5:
            violations += 1
        if row['hba'] > 10:
            violations += 1

        if violations > 0:
            lipinski_violations += 1

    print(
        f"Compounds violating Lipinski's Rule: {lipinski_violations}/{len(df)} ({lipinski_violations/len(df)*100:.1f}%)"
    )

    # Save analysis
    analysis_file = "comprehensive_analysis.txt"
    with open(analysis_file, "w") as f:
        f.write("COMPREHENSIVE ANTIBIOTICS DATABASE ANALYSIS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total compounds: {len(df)}\n")
        f.write(f"Unique SMILES: {df['smiles'].nunique()}\n")
        f.write(f"Sources: {df['source'].value_counts().to_dict()}\n")
        f.write(f"Classes: {df['class'].value_counts().to_dict()}\n\n")

        f.write("MOLECULAR PROPERTIES\n")
        f.write("-" * 20 + "\n")
        for col in numeric_cols:
            if col in df.columns:
                f.write(
                    f"{col}: mean={df[col].mean():.2f}, std={df[col].std():.2f}, min={df[col].min():.2f}, max={df[col].max():.2f}\n"
                )

        f.write(
            f"\nLipinski's Rule violations: {lipinski_violations}/{len(df)} ({lipinski_violations/len(df)*100:.1f}%)\n"
        )

    print(f"Analysis saved to {analysis_file}")


def main():
    print("Creating comprehensive antibiotics database...")

    # Merge all databases
    combined_df = merge_all_databases()

    if not combined_df.empty:
        # Save comprehensive database
        output_file = "comprehensive_antibiotics_database.csv"
        combined_df.to_csv(output_file, index=False)
        print(f"Saved comprehensive database to {output_file}")

        # Create analysis
        create_comprehensive_analysis()

        print(f"\n=== SUMMARY ===")
        print(f"Total antibiotics: {len(combined_df)}")
        print(f"Sources: {combined_df['source'].value_counts().to_dict()}")
        print(f"Classes: {combined_df['class'].value_counts().to_dict()}")

        # Test with novelty script
        print(f"\nTesting with novelty script...")
        test_smiles = "CC1=CC(=O)C2=C(C1=O)C(=O)C(=C(O2)C3=CC=CC=C3)C4=CC=CC=C4"
        print(f"Query SMILES: {test_smiles}")

    else:
        print("No data to process!")


if __name__ == "__main__":
    main()

