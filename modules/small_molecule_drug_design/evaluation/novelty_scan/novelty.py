#!/usr/bin/env python3
"""
Build antibiotics_library.csv (name, class, smiles) by resolving names via PubChem.
Then run RDKit substructure + similarity scan for a user-supplied query.
"""
import argparse
import sys
import time
from pathlib import Path

import pandas as pd

# Requires: pip install pubchempy rdkit-pypi pandas
import pubchempy as pcp
from rdkit import Chem
from rdkit.Chem import AllChem


def resolve_smiles_pubchem(name, max_tries=3, sleep=0.4):
    # Define alternative names for problematic compounds
    alt_names = {
        'polymyxin B': ['colistin', 'polymyxin'],
        'linezolid phosphate': ['linezolid']
    }

    # Try original name first, then alternatives
    search_names = [name] + alt_names.get(name, [])

    for search_name in search_names:
        for i in range(max_tries):
            try:
                cands = pcp.get_compounds(search_name, 'name')
                if cands:
                    c = cands[0]
                    # Try direct SMILES first
                    smi = c.isomeric_smiles or c.canonical_smiles
                    if smi:
                        return smi

                    # If no SMILES, try converting from InChI
                    if c.inchi:
                        from rdkit import Chem
                        mol = Chem.MolFromInchi(c.inchi)
                        if mol:
                            smi = Chem.MolToSmiles(mol)
                            return smi
                    return None
            except Exception as e:
                if i == max_tries - 1:
                    break
                time.sleep(sleep * (i + 1))
        # If this search_name failed, try the next one
        if search_name != search_names[-1]:
            time.sleep(sleep)

    return None


def make_library(names_csv, out_csv):
    df = pd.read_csv(names_csv)
    smiles, statuses = [], []
    for nm in df["name"]:
        smi = resolve_smiles_pubchem(nm)
        smiles.append(smi)
        statuses.append("ok" if smi else "not_found")
    df["smiles"] = smiles
    df["status"] = statuses
    df.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv} with {df['status'].value_counts().to_dict()}")


def rdkit_scan(library_csv,
               query_smiles,
               ring_smarts_strict=None,
               ring_smarts_general=None,
               topk=30,
               out_dir="novelty_checks"):
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    df = pd.read_csv(library_csv).dropna(subset=["smiles"]).copy()
    df = df[df["smiles"].astype(str).str.len() > 0]

    if len(df) == 0:
        print(
            "WARNING: No valid SMILES found in library. Creating empty results."
        )
        # Create empty results files
        empty_df = pd.DataFrame(columns=["name", "class", "smiles"])
        empty_nearest = pd.DataFrame(columns=[
            "name", "class", "smiles", "tanimoto_to_query", "match_strict",
            "match_general"
        ])

        out_strict = Path(out_dir) / "substructure_hits_strict.csv"
        out_general = Path(out_dir) / "substructure_hits_general.csv"
        out_nearest = Path(out_dir) / "nearest_neighbors.csv"
        empty_df.to_csv(out_strict, index=False)
        empty_df.to_csv(out_general, index=False)
        empty_nearest.to_csv(out_nearest, index=False)

        print(f"Strict hits: 0  |  General hits: 0")
        print(f"Wrote: {out_strict}, {out_general}, {out_nearest}")
        return

    qmol = Chem.MolFromSmiles(query_smiles)
    if qmol is None:
        print("ERROR: Query SMILES failed to parse.", file=sys.stderr)
        sys.exit(2)

    # compile patterns
    patt_strict = Chem.MolFromSmarts(
        ring_smarts_strict) if ring_smarts_strict else None
    patt_general = Chem.MolFromSmarts(
        ring_smarts_general) if ring_smarts_general else None

    # build mols+fps
    mols, fps = [], []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(s)
        mols.append(m)
        fps.append(
            AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048
                                                  ) if m else None)
    df["mol"] = mols
    df["fp"] = fps

    def has_match(m, patt):
        return bool(m and patt and m.HasSubstructMatch(patt))

    df["match_strict"] = df["mol"].apply(lambda m: has_match(m, patt_strict))
    df["match_general"] = df["mol"].apply(lambda m: has_match(m, patt_general))

    # neighbors
    qfp = AllChem.GetMorganFingerprintAsBitVect(qmol, 2, nBits=2048)
    from rdkit.DataStructs.cDataStructs import TanimotoSimilarity
    df["tanimoto_to_query"] = df["fp"].apply(
        lambda fp: TanimotoSimilarity(qfp, fp) if fp is not None else 0.0)

    # write reports
    strict_hits = df[df["match_strict"]][["name", "class", "smiles"]]
    general_hits = df[df["match_general"]][["name", "class", "smiles"]]
    nearest = df.sort_values("tanimoto_to_query", ascending=False).head(topk)[[
        "name", "class", "smiles", "tanimoto_to_query", "match_strict",
        "match_general"
    ]]

    out_strict = Path(out_dir) / "substructure_hits_strict.csv"
    out_general = Path(out_dir) / "substructure_hits_general.csv"
    out_nearest = Path(out_dir) / "nearest_neighbors.csv"
    strict_hits.to_csv(out_strict, index=False)
    general_hits.to_csv(out_general, index=False)
    nearest.to_csv(out_nearest, index=False)

    print(
        f"Strict hits: {len(strict_hits)}  |  General hits: {len(general_hits)}"
    )
    print(f"Wrote: {out_strict}, {out_general}, {out_nearest}")


def main():
    ap = argparse.ArgumentParser(
        description=
        "Build antibiotic SMILES library from names via PubChem, then scan for substructure/similarity."
    )
    ap.add_argument("--names_csv",
                    default="antibiotics_names.csv",
                    help="CSV with columns: name,class")
    ap.add_argument("--out_library_csv",
                    default="comprehensive_antibiotics_database.csv",
                    help="Output CSV with name,class,smiles,status")
    ap.add_argument("--query",
                    required=True,
                    help="Query SMILES to check for novelty")
    ap.add_argument("--strict_smarts",
                    default="c1nc(Cl)c(C#N)s1",
                    help="Strict substructure SMARTS")
    ap.add_argument("--general_smarts",
                    default="c1ncc(s1)",
                    help="Generalized substructure SMARTS")
    ap.add_argument("--topk", type=int, default=30)
    ap.add_argument("--out_dir", default="novelty_checks")
    args = ap.parse_args()

    library_csv = Path(args.out_library_csv)
    if not library_csv.exists():
        names_csv = Path(args.names_csv)
        if not names_csv.exists():
            print(f"ERROR: {names_csv} not found.", file=sys.stderr)
            sys.exit(1)
        print(f"Building library from {names_csv}...")
        make_library(names_csv, args.out_library_csv)
    else:
        print(f"Using existing library: {library_csv}")

    rdkit_scan(args.out_library_csv, args.query, args.strict_smarts,
               args.general_smarts, args.topk, args.out_dir)


if __name__ == "__main__":
    main()
