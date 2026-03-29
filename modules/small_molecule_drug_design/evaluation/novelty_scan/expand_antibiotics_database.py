#!/usr/bin/env python3
"""
Script to expand antibiotics database using multiple sources:
1. ChEMBL API - for comprehensive drug database
2. DrugBank - for FDA-approved drugs
3. WHO Essential Medicines List
4. Additional PubChem searches with broader terms
"""

import requests
import pandas as pd
import time
import json
from pathlib import Path
import pubchempy as pcp
from rdkit import Chem


def get_chembl_antibiotics():
    """Get antibiotics from ChEMBL database"""
    print("Fetching antibiotics from ChEMBL...")

    # ChEMBL API endpoint for antibiotics
    url = "https://www.ebi.ac.uk/chembl/api/data/molecule"

    # Search for antibiotics using various terms
    search_terms = [
        "antibiotic", "antimicrobial", "bactericidal", "bacteriostatic",
        "fluoroquinolone", "penicillin", "cephalosporin", "tetracycline",
        "macrolide", "aminoglycoside"
    ]

    all_compounds = []

    for term in search_terms:
        try:
            params = {
                'molecule_synonyms__molecule_synonym__icontains': term,
                'limit': 1000,
                'format': 'json'
            }

            response = requests.get(url, params=params)
            if response.status_code == 200:
                data = response.json()
                compounds = data.get('molecules', [])
                print(f"Found {len(compounds)} compounds for '{term}'")
                all_compounds.extend(compounds)
            else:
                print(
                    f"Failed to fetch data for '{term}': {response.status_code}"
                )

            time.sleep(1)  # Rate limiting

        except Exception as e:
            print(f"Error fetching '{term}': {e}")

    # Remove duplicates based on chembl_id
    unique_compounds = {}
    for compound in all_compounds:
        chembl_id = compound.get('molecule_chembl_id')
        if chembl_id and chembl_id not in unique_compounds:
            unique_compounds[chembl_id] = compound

    print(f"Total unique compounds from ChEMBL: {len(unique_compounds)}")
    return list(unique_compounds.values())


def get_drugbank_antibiotics():
    """Get antibiotics from DrugBank (requires API key)"""
    print("Note: DrugBank requires API key. Skipping for now...")
    return []


def get_who_essential_medicines():
    """Get WHO Essential Medicines List antibiotics"""
    print("Fetching WHO Essential Medicines List...")

    # WHO EML API endpoint
    url = "https://list.essentialmeds.org/api/medicines"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
            medicines = data.get('medicines', [])

            # Filter for antibiotics
            antibiotics = []
            for medicine in medicines:
                if medicine.get('category') and 'antibiotic' in medicine.get(
                        'category', '').lower():
                    antibiotics.append(medicine)

            print(f"Found {len(antibiotics)} antibiotics in WHO EML")
            return antibiotics
        else:
            print(f"Failed to fetch WHO EML: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error fetching WHO EML: {e}")
        return []


def get_broad_pubchem_searches():
    """Get antibiotics using broader PubChem searches"""
    print("Performing broader PubChem searches...")

    # Broader search terms
    search_terms = [
        "antibiotic", "antimicrobial", "bactericidal", "bacteriostatic",
        "fluoroquinolone", "quinolone", "penicillin", "cephalosporin",
        "carbapenem", "monobactam", "tetracycline", "macrolide",
        "aminoglycoside", "glycopeptide", "lipopeptide", "oxazolidinone",
        "lincosamide", "rifamycin", "polymyxin", "sulfonamide",
        "nitroimidazole", "nitrofuran", "amphenicol", "pleuromutilin"
    ]

    all_compounds = []

    for term in search_terms:
        try:
            compounds = pcp.get_compounds(term, 'name', listkey_count=50)
            print(f"Found {len(compounds)} compounds for '{term}'")

            for compound in compounds:
                if compound.cid:
                    all_compounds.append({
                        'name':
                        term,
                        'cid':
                        compound.cid,
                        'molecular_formula':
                        compound.molecular_formula,
                        'molecular_weight':
                        compound.molecular_weight
                    })

            time.sleep(0.5)  # Rate limiting

        except Exception as e:
            print(f"Error searching '{term}': {e}")

    print(f"Total compounds from PubChem: {len(all_compounds)}")
    return all_compounds


def resolve_smiles_for_compounds(compounds):
    """Resolve SMILES for a list of compounds"""
    print("Resolving SMILES for compounds...")

    results = []
    for i, compound in enumerate(compounds):
        if i % 10 == 0:
            print(f"Processing {i}/{len(compounds)}...")

        try:
            if 'cid' in compound:
                # PubChem compound
                c = pcp.Compound.from_cid(compound['cid'])
                smiles = None

                # Try direct SMILES
                if c.canonical_smiles:
                    smiles = c.canonical_smiles
                elif c.isomeric_smiles:
                    smiles = c.isomeric_smiles

                # Try InChI conversion
                if not smiles and c.inchi:
                    mol = Chem.MolFromInchi(c.inchi)
                    if mol:
                        smiles = Chem.MolToSmiles(mol)

                if smiles:
                    results.append({
                        'name':
                        compound.get('name', 'Unknown'),
                        'cid':
                        compound['cid'],
                        'smiles':
                        smiles,
                        'molecular_formula':
                        compound.get('molecular_formula', ''),
                        'molecular_weight':
                        compound.get('molecular_weight', ''),
                        'source':
                        'PubChem'
                    })

        except Exception as e:
            print(
                f"Error resolving compound {compound.get('name', 'Unknown')}: {e}"
            )

        time.sleep(0.1)  # Rate limiting

    return results


def main():
    print("Expanding antibiotics database...")

    # Get data from multiple sources
    chembl_data = get_chembl_antibiotics()
    who_data = get_who_essential_medicines()
    pubchem_data = get_broad_pubchem_searches()

    # Combine all data
    all_data = []

    # Add PubChem data
    if pubchem_data:
        all_data.extend(pubchem_data)

    print(f"Total compounds collected: {len(all_data)}")

    # Resolve SMILES for compounds
    if all_data:
        resolved_data = resolve_smiles_for_compounds(all_data)

        # Save to CSV
        df = pd.DataFrame(resolved_data)
        output_file = "expanded_antibiotics_database.csv"
        df.to_csv(output_file, index=False)
        print(f"Saved {len(resolved_data)} compounds to {output_file}")

        # Show summary
        print(f"\nSummary:")
        print(f"- Total compounds: {len(resolved_data)}")
        print(f"- Unique names: {df['name'].nunique()}")
        print(f"- With SMILES: {len(df[df['smiles'].notna()])}")

    else:
        print("No data collected")


if __name__ == "__main__":
    main()
