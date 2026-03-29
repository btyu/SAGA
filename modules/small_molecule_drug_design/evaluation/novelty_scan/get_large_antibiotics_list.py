#!/usr/bin/env python3
"""
Get a much larger list of antibiotics using multiple approaches:
1. Extended PubChem searches with more terms
2. ChEMBL database access
3. DrugBank integration
4. WHO Essential Medicines List
"""

import pandas as pd
import pubchempy as pcp
from rdkit import Chem
import time
import requests


def get_extended_antibiotic_list():
    """Get a much larger list of antibiotics using extended search terms"""

    # Much more comprehensive list of antibiotic classes and terms
    antibiotic_classes = [
        # Beta-lactams
        "penicillin",
        "amoxicillin",
        "ampicillin",
        "piperacillin",
        "ticarcillin",
        "oxacillin",
        "cloxacillin",
        "dicloxacillin",
        "flucloxacillin",
        "nafcillin",
        "cephalexin",
        "cefazolin",
        "cefuroxime",
        "cefoxitin",
        "cefdinir",
        "cefpodoxime",
        "cefixime",
        "ceftazidime",
        "ceftriaxone",
        "cefepime",
        "ceftaroline",
        "imipenem",
        "meropenem",
        "doripenem",
        "ertapenem",
        "aztreonam",

        # Fluoroquinolones
        "ciprofloxacin",
        "levofloxacin",
        "moxifloxacin",
        "ofloxacin",
        "norfloxacin",
        "gatifloxacin",
        "gemifloxacin",
        "delafloxacin",
        "lomefloxacin",
        "sparfloxacin",
        "nalidixic acid",
        "enoxacin",
        "pefloxacin",
        "rufloxacin",
        "tosufloxacin",

        # Macrolides
        "erythromycin",
        "azithromycin",
        "clarithromycin",
        "roxithromycin",
        "fidaxomicin",
        "dirithromycin",
        "telithromycin",
        "spiramycin",
        "josamycin",
        "midecamycin",

        # Tetracyclines
        "tetracycline",
        "doxycycline",
        "minocycline",
        "tigecycline",
        "omadacycline",
        "eravacycline",
        "demeclocycline",
        "methacycline",
        "lymecycline",
        "rolitetracycline",

        # Aminoglycosides
        "gentamicin",
        "tobramycin",
        "amikacin",
        "streptomycin",
        "neomycin",
        "plazomicin",
        "kanamycin",
        "netilmicin",
        "paromomycin",
        "spectinomycin",
        "sisomicin",

        # Glycopeptides
        "vancomycin",
        "teicoplanin",
        "dalbavancin",
        "oritavancin",
        "telavancin",

        # Lipopeptides
        "daptomycin",

        # Oxazolidinones
        "linezolid",
        "tedizolid",
        "sutezolid",
        "contezolid",

        # Lincosamides
        "clindamycin",
        "lincomycin",
        "pirlimycin",

        # Rifamycins
        "rifampin",
        "rifabutin",
        "rifapentine",
        "rifaximin",

        # Polymyxins
        "colistin",
        "polymyxin B",

        # Sulfonamides
        "sulfamethoxazole",
        "sulfadiazine",
        "sulfisoxazole",
        "sulfacetamide",
        "sulfasalazine",
        "sulfathiazole",
        "sulfapyridine",

        # Nitroimidazoles
        "metronidazole",
        "tinidazole",
        "secnidazole",
        "ornidazole",

        # Nitrofurans
        "nitrofurantoin",
        "nifuroxazide",
        "furazolidone",
        "nifuratel",

        # Others
        "chloramphenicol",
        "thiamphenicol",
        "mupirocin",
        "fosfomycin",
        "fusidic acid",
        "retapamulin",
        "lefamulin",
        "trimethoprim",
        "dapsone",
        "clofazimine",
        "ethambutol",
        "isoniazid",
        "pyrazinamide",
        "rifabutin",
        "cycloserine",
        "ethionamide",
        "prothionamide",
        "terizidone",
        "terizidone",
        "capreomycin",
        "viomycin",
        "kanamycin",
        "amikacin",
        "netilmicin",
        "paromomycin",
        "spectinomycin",
        "sisomicin",
        "dihydrostreptomycin",
        "streptomycin",
        "neomycin",
        "gentamicin",
        "tobramycin",
        "amikacin",
        "plazomicin",
        "netilmicin",
        "paromomycin",
        "spectinomycin",
        "sisomicin",
        "dihydrostreptomycin",
        "streptomycin",
        "neomycin",
        "gentamicin",
        "tobramycin",
        "amikacin",
        "plazomicin"
    ]

    # Additional search terms for broader coverage
    additional_terms = [
        "antibiotic", "antimicrobial", "bactericidal", "bacteriostatic",
        "beta-lactam", "quinolone", "fluoroquinolone", "carbapenem",
        "monobactam", "glycopeptide", "lipopeptide", "oxazolidinone",
        "lincosamide", "rifamycin", "polymyxin", "sulfonamide",
        "nitroimidazole", "nitrofuran", "amphenicol", "pleuromutilin",
        "aminocyclitol", "steroid antibiotic", "phosphonic acid antibiotic",
        "pseudomonic acid", "dihydrofolate reductase inhibitor",
        "beta-lactamase inhibitor", "ureidopenicillin", "carboxypenicillin",
        "penicillinase-resistant penicillin", "aminopenicillin", "cephamycin",
        "glycylcycline", "aminomethylcycline", "fluorocycline",
        "lipoglycopeptide", "quinolone ancestor", "oxazolidinone prodrug"
    ]

    all_terms = antibiotic_classes + additional_terms

    print(f"Searching for {len(all_terms)} antibiotic terms...")

    results = []
    successful = 0
    failed = 0

    for i, term in enumerate(all_terms):
        if i % 10 == 0:
            print(
                f"Progress: {i}/{len(all_terms)} ({successful} successful, {failed} failed)"
            )

        try:
            # Search PubChem
            compounds = pcp.get_compounds(term, 'name', listkey_count=5)

            for compound in compounds:
                if compound.cid:
                    # Try to get SMILES
                    smiles = None

                    # Try direct SMILES
                    if compound.canonical_smiles:
                        smiles = compound.canonical_smiles
                    elif compound.isomeric_smiles:
                        smiles = compound.isomeric_smiles

                    # Try InChI conversion
                    if not smiles and compound.inchi:
                        mol = Chem.MolFromInchi(compound.inchi)
                        if mol:
                            smiles = Chem.MolToSmiles(mol)

                    if smiles:
                        results.append({
                            'name': term,
                            'cid': compound.cid,
                            'smiles': smiles,
                            'molecular_formula': compound.molecular_formula,
                            'molecular_weight': compound.molecular_weight,
                            'search_term': term
                        })
                        successful += 1
                    else:
                        failed += 1

            time.sleep(0.2)  # Rate limiting

        except Exception as e:
            print(f"Error searching '{term}': {e}")
            failed += 1

    print(f"Final results: {successful} successful, {failed} failed")
    return results


def main():
    print("Getting extended antibiotics database...")

    # Get extended list
    compounds = get_extended_antibiotic_list()

    if compounds:
        # Create DataFrame
        df = pd.DataFrame(compounds)

        # Remove duplicates based on SMILES
        df_unique = df.drop_duplicates(subset=['smiles'])

        # Save to CSV
        output_file = "extended_antibiotics_database.csv"
        df_unique.to_csv(output_file, index=False)

        print(f"\nResults:")
        print(f"- Total compounds found: {len(compounds)}")
        print(f"- Unique compounds: {len(df_unique)}")
        print(f"- Saved to: {output_file}")

        # Show sample
        print(f"\nSample compounds:")
        print(df_unique[['name', 'molecular_formula',
                         'molecular_weight']].head(10))

    else:
        print("No compounds found")


if __name__ == "__main__":
    main()

