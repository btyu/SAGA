#!/usr/bin/env python3
import argparse
import pandas as pd
from tqdm import tqdm
import torch

def main():
    parser = argparse.ArgumentParser(description="Add druglikeness scores to SMILES file")
    parser.add_argument("input", help="Path to input SMILES file (.smi)")
    parser.add_argument("output", help="Path to output CSV file")
    parser.add_argument("--batch-size", type=int, default=1000, help="Batch size for processing")
    args = parser.parse_args()

    # Initialize DeepDL model
    from druglikeness.deepdl import DeepDL
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DeepDL.from_pretrained('extended', device=device)
    print(f"Using {device} for scoring")
    
    # Read SMILES
    smiles_list = []
    with open(args.input, "r") as f:
        for line in tqdm(f, desc="Loading SMILES"):
            parts = line.strip().split()
            if parts:
                smiles_list.append(parts[0])
    
    print(f"Processing {len(smiles_list)} SMILES...")
    
    # Score in batches with progress bar
    all_scores = []
    for i in tqdm(range(0, len(smiles_list), args.batch_size), desc="Scoring batches"):
        batch = smiles_list[i:i + args.batch_size]
        batch_scores = model.screening(smiles_list=batch, naive=True, batch_size=64)
        all_scores.extend(batch_scores)
    
    # Normalize scores (0-100 -> 0-1)
    normalized_scores = [s/100.0 if s is not None else None for s in all_scores]
    
    # Save results
    df = pd.DataFrame({
        'smiles': smiles_list,
        'druglikeness': normalized_scores
    })
    df.to_csv(args.output, index=False)
    
    print(f"Results saved to {args.output}")
    valid_scores = [s for s in normalized_scores if s is not None]
    if valid_scores:
        print(f"Valid: {len(valid_scores)}/{len(smiles_list)}")
        print(f"Mean: {sum(valid_scores)/len(valid_scores):.3f}")

if __name__ == "__main__":
    main()
