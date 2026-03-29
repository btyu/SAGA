#!/usr/bin/env python3
import argparse
import datamol as dm
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import os

dm.disable_rdkit_log()

def clean_smiles(
    smi: str,
    *,
    sanifix: bool = True,
    charge_neutral: bool = False,
    disconnect_metals: bool = False,
    normalize: bool = True,
    reionize: bool = True,
    uncharge: bool = False,
    stereo: bool = True,
    largest_fragment: bool = True,
):
    try:
        mol = dm.to_mol(smi, ordered=True)
        if mol is None:
            return None

        mol = dm.fix_mol(mol)
        mol = dm.sanitize_mol(mol, sanifix=sanifix, charge_neutral=charge_neutral)
        mol = dm.standardize_mol(
            mol,
            disconnect_metals=disconnect_metals,
            normalize=normalize,
            reionize=reionize,
            uncharge=uncharge,
            stereo=stereo,
        )

        if largest_fragment:
            mol = dm.keep_largest_fragment(mol)

        return dm.standardize_smiles(dm.to_smiles(mol))
    except Exception:
        return None


def process_smiles_chunk(smiles_data):
    """Process a chunk of SMILES data"""
    results = []
    for smi, name in smiles_data:
        cleaned = clean_smiles(smi)
        if cleaned is not None:
            results.append((cleaned, name))
    return results


def main():
    parser = argparse.ArgumentParser(description="Clean SMILES with datamol")
    parser.add_argument("input", help="Path to input SMILES file")
    parser.add_argument("output", help="Path to output cleaned SMILES file")
    parser.add_argument("--workers", type=int, default=cpu_count(), help="Number of worker processes")
    parser.add_argument("--chunk-size", type=int, default=1000, help="Chunk size for multiprocessing")
    args = parser.parse_args()

    # Read all SMILES data
    smiles_data = []
    print("Reading SMILES data...")
    with open(args.input, "r") as fin:
        for line in tqdm(fin, desc="Loading SMILES"):
            parts = line.strip().split()
            if not parts:
                continue
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else ""
            smiles_data.append((smi, name))
    print(f"Read {len(smiles_data)} SMILES entries")
            

    # Create chunks for multiprocessing
    chunks = [smiles_data[i:i + args.chunk_size] for i in range(0, len(smiles_data), args.chunk_size)]
    
    # Process chunks with multiprocessing and progress bar
    with Pool(min(args.workers, 16)) as pool:
        with open(args.output, "w") as fout:
            for results in tqdm(pool.imap(process_smiles_chunk, chunks), total=len(chunks), desc="Processing SMILES"):
                for cleaned, name in results:
                    fout.write(f"{cleaned}\t{name}\n")


if __name__ == "__main__":
    main()
