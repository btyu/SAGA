import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.DataStructs import TanimotoSimilarity
from rdkit.Chem import AllChem
from typing import List, Tuple

def compute_aggregate_score(row, df):
    """Compute aggregate score as product of kp, novelty, toxicity, motifs, similarity."""
    # Handle different column names
    if 'kp' in df.columns:
        kp = row['kp'] if pd.notna(row['kp']) else 0.0
    elif 'klebsiella_pneumoniae' in df.columns:
        kp = row['klebsiella_pneumoniae'] if pd.notna(row['klebsiella_pneumoniae']) else 0.0
    else:
        kp = 0.0
    
    if 'novelty' in df.columns:
        novelty = row['novelty'] if pd.notna(row['novelty']) else 0.0
    elif 'antibiotics_novelty' in df.columns:
        novelty = row['antibiotics_novelty'] if pd.notna(row['antibiotics_novelty']) else 0.0
    else:
        novelty = 0.0
    
    toxicity = row['toxicity'] if pd.notna(row['toxicity']) else 0.0
    
    if 'motifs' in df.columns:
        motifs = row['motifs'] if pd.notna(row['motifs']) else 0.0
    elif 'antibiotics_motifs_filter' in df.columns:
        motifs = row['antibiotics_motifs_filter'] if pd.notna(row['antibiotics_motifs_filter']) else 0.0
    else:
        motifs = 0.0
    
    if 'similarity' in df.columns:
        similarity = row['similarity'] if pd.notna(row['similarity']) else 0.0
    else:
        # If similarity column doesn't exist, use 1.0 (don't penalize)
        similarity = 1.0
    
    return kp * novelty * toxicity * motifs * similarity

def get_fingerprint(smiles: str):
    """Get Morgan fingerprint for a SMILES string."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    except:
        return None

def select_diverse_molecules(df: pd.DataFrame, n: int = 100) -> pd.DataFrame:
    """Select top n diverse molecules based on aggregate score and diversity."""
    # Sort by aggregate score descending
    df_sorted = df.sort_values('aggregate_score', ascending=False).reset_index(drop=True)
    
    # Get SMILES column name
    smiles_col = 'smiles' if 'smiles' in df.columns else 'SMILES'
    
    selected = []
    fingerprints = []
    
    for idx, row in df_sorted.iterrows():
        smiles = row[smiles_col]
        if pd.isna(smiles):
            continue
        fp = get_fingerprint(smiles)
        
        if fp is None:
            continue
        
        # Check diversity: minimum Tanimoto similarity to already selected molecules
        is_diverse = True
        for selected_fp in fingerprints:
            similarity = TanimotoSimilarity(fp, selected_fp)
            if similarity > 0.7:  # Threshold for diversity
                is_diverse = False
                break
        
        if is_diverse:
            selected.append(idx)
            fingerprints.append(fp)
            
            if len(selected) >= n:
                break
    
    return df_sorted.loc[selected].reset_index(drop=True)

def process_csv(filepath: str):
    """Process a CSV file and compute top 100 diverse molecules."""
    print(f"\nProcessing {filepath}...")
    df = pd.read_csv(filepath)
    
    # Compute aggregate score
    df['aggregate_score'] = df.apply(lambda row: compute_aggregate_score(row, df), axis=1)
    
    # Select top 100 diverse molecules
    top100_diverse = select_diverse_molecules(df, n=100)
    
    # Get kp column name
    kp_col = 'kp' if 'kp' in df.columns else 'klebsiella_pneumoniae'
    
    # Compute percentage with kp > 0.05
    kp_above_threshold = (top100_diverse[kp_col] > 0.05).sum()
    percentage = (kp_above_threshold / len(top100_diverse)) * 100
    
    print(f"Total molecules: {len(df)}")
    print(f"Top 100 diverse molecules selected: {len(top100_diverse)}")
    print(f"Molecules with kp > 0.05: {kp_above_threshold}")
    print(f"Percentage with kp > 0.05: {percentage:.2f}%")
    
    return top100_diverse, percentage

if __name__ == "__main__":
    files = [
        "naturelm_kp_antibiot_jointscorer_cleaned_withscore.csv",
        "textgrad_kp_output_new_epoch_merged_updatebaseline_withscore.csv",
        "molt5_KP_antibiot_final_scored_results.csv",
        "reinvent4_kp_output_new_epoch_merged_updatebaseline_withscore.csv"
    ]
    
    results = {}
    for filepath in files:
        print("\n" + "=" * 60)
        print(f"File: {filepath}")
        print("=" * 60)
        top100, pct = process_csv(filepath)
        results[filepath] = pct
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for filepath, pct in results.items():
        filename = filepath.split('/')[-1]
        print(f"{filename} - Percentage with kp > 0.05: {pct:.2f}%")

