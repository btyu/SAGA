#!/usr/bin/env python3
"""Analyze which SMARTS patterns each failing SMILES matched."""

import pandas as pd
import sys
from pathlib import Path

# Add the project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from modules.small_molecule_drug_design.ablations.evaluation_type2 import load_scileo_seed
from modules.small_molecule_drug_design.utils.rdkit_utils import filter_smiles_preserves_existing_hits

print("Loading data...")
# Load data
data, _ = load_scileo_seed('kp_level1_iter2_20251105131705_all_molecules.csv', 'kpneumoniae')

# Get failing molecules (antibiotics_motifs_filter == False)
failing = data[data['antibiotics_motifs_filter'] == False].copy()

print(f"Total molecules: {len(data)}")
print(f"Failing antibiotics_motifs_filter: {len(failing)}")
print(f"\nAnalyzing which SMARTS rule each failing SMILES matched...\n")

# Get reasons for each failing SMILES - check individually to get detailed reasons
smiles_list = failing['smiles'].tolist()
reasons_map = {}
kept_list = []

# Check each molecule individually to get detailed reasons
for smi in smiles_list:
    kept, reasons = filter_smiles_preserves_existing_hits([smi])
    if len(kept) == 0:  # Dropped - should have a reason
        if smi in reasons:
            reasons_map[smi] = reasons[smi]
        else:
            reasons_map[smi] = "UNKNOWN (filtered but no reason found)"
    else:  # Kept - shouldn't happen since antibiotics_motifs_filter == False
        reasons_map[smi] = "KEPT (should not be here!)"

# Create a detailed report
results = []
for idx, row in failing.iterrows():
    smi = row['smiles']
    reason = reasons_map.get(smi, "UNKNOWN (no reason found)")
    
    # Parse reason to get individual patterns
    patterns = []
    if reason:
        for part in reason.split('; '):
            patterns.append(part.strip())
    
    results.append({
        'smiles': smi,
        'filter_reason': reason,
        'matched_patterns': ', '.join(patterns) if patterns else 'UNKNOWN'
    })

results_df = pd.DataFrame(results)

# Count by pattern
pattern_counts = {}
for _, row in results_df.iterrows():
    if row['filter_reason']:
        for part in row['filter_reason'].split('; '):
            pattern = part.strip()
            if pattern:
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

print("=" * 80)
print("SUMMARY: Which SMARTS rules are causing failures")
print("=" * 80)
for pattern, count in sorted(pattern_counts.items(), key=lambda x: -x[1]):
    print(f"  {pattern}: {count} molecules")

print("\n" + "=" * 80)
print("DETAILED BREAKDOWN: Each failing SMILES and its matched pattern(s)")
print("=" * 80)
for i, (_, row) in enumerate(results_df.iterrows(), 1):
    print(f"\n{i}. SMILES: {row['smiles']}")
    print(f"   Matched pattern(s): {row['matched_patterns']}")

