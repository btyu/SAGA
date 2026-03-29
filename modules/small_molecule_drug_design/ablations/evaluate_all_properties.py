#!/usr/bin/env python3
"""
Type 2 Evaluation for ALL molecules with ALL properties computed properly.
NO FAKE FILTERS - everything is computed for real.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, FilterCatalog
import sys

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable

# Add project root to path
script_dir = Path(__file__).resolve().parent
project_root = script_dir.parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Activity thresholds to test
ACTIVITY_THRESHOLDS = [0.05, 0.1]

# Held-out filter thresholds
FILTERS = {
    "qed": 0.5,
    "sa": 0.5,
    "mw": 0.5,
    "pains": 1.0,
    "brenk": 1.0,
    "antibiotics_novelty": 0.6,
    "antibiotics_motifs_filter": 1.0,
    "toxicity": 0.5,
    "ring_score": 1.0,
}

# Global catalogs
_pains_catalog = None
_brenk_catalog = None


def _init_filters():
    """Initialize PAINS and BRENK catalogs."""
    global _pains_catalog, _brenk_catalog
    if _pains_catalog is None:
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        _pains_catalog = FilterCatalog.FilterCatalog(pains_params)

    if _brenk_catalog is None:
        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)


def _compute_qed(smiles_list):
    """Compute QED scores."""
    scores = []
    for smi in tqdm(smiles_list, desc="Computing QED"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            scores.append(QED.qed(mol))
        else:
            scores.append(np.nan)
    return scores


def _compute_sa(smiles_list):
    """Compute synthetic accessibility scores."""
    from rdkit.Chem import RDConfig
    import os
    sys.path.append(os.path.join(RDConfig.RDContribDir, 'SA_Score'))
    try:
        import sascorer
        scores = []
        for smi in tqdm(smiles_list, desc="Computing SA"):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                sa = sascorer.calculateScore(mol)
                scores.append(1.0 - (sa / 10.0))
            else:
                scores.append(np.nan)
        return scores
    except ImportError:
        print("  Warning: SA Score module not available, returning NaN")
        return [np.nan] * len(smiles_list)


def _compute_mw(smiles_list):
    """Compute molecular weight scores."""
    scores = []
    for smi in tqdm(smiles_list, desc="Computing MW"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mw = Descriptors.MolWt(mol)
            if mw <= 500:
                scores.append(1.0)
            elif mw <= 600:
                scores.append(1.0 - (mw - 500) / 100)
            else:
                scores.append(0.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_pains(smiles_list):
    """Compute PAINS filter (1.0 = pass, 0.0 = fail)."""
    _init_filters()
    scores = []
    for smi in tqdm(smiles_list, desc="Computing PAINS"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            has_pains = _pains_catalog.HasMatch(mol)
            scores.append(0.0 if has_pains else 1.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_brenk(smiles_list):
    """Compute BRENK filter (1.0 = pass, 0.0 = fail)."""
    _init_filters()
    scores = []
    for smi in tqdm(smiles_list, desc="Computing BRENK"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            has_brenk = _brenk_catalog.HasMatch(mol)
            scores.append(0.0 if has_brenk else 1.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_ring_score(smiles_list):
    """Compute ring score filter (1.0 = pass, 0.0 = fail)."""
    scores = []
    for smi in tqdm(smiles_list, desc="Computing Ring Score"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings == 0:
                scores.append(0.0)
            else:
                max_ring_size = max([len(ring) for ring in ring_info.AtomRings()], default=0)
                if max_ring_size > 7:
                    scores.append(0.0)
                else:
                    scores.append(1.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_toxicity(smiles_list):
    """Compute toxicity using chemprop model."""
    print("  Loading chemprop toxicity model...")
    try:
        from modules.small_molecule_drug_design.scorer_mcp.chemprop_scorers_mcp.base import Scorer as ChempropScorer
        scorer = ChempropScorer()
        scores = scorer.score_primary_cell_toxicity(smiles_list)
        return scores
    except Exception as e:
        print(f"    Warning: Toxicity computation failed: {e}")
        print(f"    Returning NaN for all molecules")
        return [np.nan] * len(smiles_list)


def _compute_antibiotics_novelty(smiles_list):
    """Compute antibiotics novelty score."""
    print("  Computing novelty against known antibiotics...")
    try:
        from modules.small_molecule_drug_design.scorer_mcp.antibiotics_scorer_mcp.base import Scorer as AntibioticsScorer
        scorer = AntibioticsScorer()
        scores = scorer.score_antibiotics_novelty(smiles_list)
        return scores
    except Exception as e:
        print(f"    Warning: Novelty computation failed: {e}")
        print(f"    Returning NaN for all molecules")
        return [np.nan] * len(smiles_list)


def _compute_antibiotics_motifs_filter(smiles_list):
    """Compute antibiotics motifs filter (1.0 = no known motifs, 0.0 = has motifs)."""
    print("  Computing antibiotics motifs filter...")
    try:
        from modules.small_molecule_drug_design.utils.rdkit_utils import filter_smiles_preserves_existing_hits
        
        # Batch process for efficiency
        results = [None] * len(smiles_list)
        valid_smiles = []
        valid_indices = []

        for i, smi in enumerate(smiles_list):
            if smi and smi.strip():
                valid_smiles.append(smi)
                valid_indices.append(i)

        if not valid_smiles:
            return results

        # Process in batch - function returns (kept_list, reasons_dict)
        kept_list, reasons_dict = filter_smiles_preserves_existing_hits(valid_smiles)
        kept_set = set(kept_list)
        for idx, smi in zip(valid_indices, valid_smiles):
            results[idx] = 1.0 if smi in kept_set else 0.0
        
        return results
    except Exception as e:
        print(f"    Warning: Motifs filter computation failed: {e}")
        print(f"    Returning NaN for all molecules")
        return [np.nan] * len(smiles_list)


def compute_properties(df):
    """Compute ALL required properties for molecules."""
    print("\nComputing molecular properties for ALL molecules...")
    print("NO FAKE FILTERS - everything is computed properly")
    
    df = df.copy()
    
    valid_smiles = df["smiles"].dropna()
    valid_indices = valid_smiles.index.tolist()
    smiles_list = valid_smiles.tolist()
    
    if not smiles_list:
        raise ValueError("No valid SMILES found")
    
    print(f"\nProcessing {len(smiles_list)} molecules...")
    
    prop_computers = {
        "qed": _compute_qed,
        "sa": _compute_sa,
        "mw": _compute_mw,
        "pains": _compute_pains,
        "brenk": _compute_brenk,
        "ring_score": _compute_ring_score,
        "toxicity": _compute_toxicity,
        "antibiotics_novelty": _compute_antibiotics_novelty,
        "antibiotics_motifs_filter": _compute_antibiotics_motifs_filter,
    }
    
    for prop, computer in prop_computers.items():
        if prop not in df.columns or df[prop].isna().all():
            print(f"\n{'='*60}")
            print(f"Computing {prop}...")
            print(f"{'='*60}")
            scores = computer(smiles_list)
            df.loc[valid_indices, prop] = scores
            
            # Report statistics
            non_null = df[prop].notna().sum()
            if non_null > 0:
                mean_val = df[prop].mean()
                pass_count = (df[prop] >= FILTERS.get(prop, 0.5)).sum()
                print(f"  ✓ Computed: {non_null}/{len(smiles_list)}")
                print(f"  Mean: {mean_val:.3f}")
                print(f"  Passing (>= {FILTERS.get(prop, 0.5)}): {pass_count} ({100*pass_count/len(df):.1f}%)")
            else:
                print(f"  ⚠ Warning: All values are NaN")
    
    return df


def evaluate_filters(df, activity_col, activity_threshold):
    """Evaluate filter pass rates at given activity threshold."""
    results = {}
    
    # Start with molecules passing activity threshold
    df_filtered = df[df[activity_col] >= activity_threshold].copy()
    results["Initial (Activity)"] = len(df_filtered)
    
    if len(df_filtered) == 0:
        return results, df_filtered
    
    # Apply each filter
    filter_order = [
        ("antibiotics_novelty", "Novelty"),
        ("toxicity", "Toxicity"),
        ("antibiotics_motifs_filter", "Motifs"),
        ("qed", "QED"),
        ("sa", "SA"),
        ("mw", "MW"),
        ("pains", "PAINS"),
        ("brenk", "BRENK"),
        ("ring_score", "Ring"),
    ]
    
    for col, name in filter_order:
        if col in df_filtered.columns:
            threshold = FILTERS.get(col, 0.5)
            before = len(df_filtered)
            df_filtered = df_filtered[df_filtered[col].notna() & (df_filtered[col] >= threshold)]
            after = len(df_filtered)
            results[name] = after
            if before > 0:
                pct = 100 * after / before
                print(f"  {name}: {after}/{before} ({pct:.1f}% pass)")
    
    # Dedup
    before_dedup = len(df_filtered)
    df_filtered = df_filtered.drop_duplicates(subset=["smiles"])
    after_dedup = len(df_filtered)
    results["Dedup"] = after_dedup
    if before_dedup > 0:
        print(f"  Dedup: {after_dedup}/{before_dedup} ({100*after_dedup/before_dedup:.1f}% unique)")
    
    return results, df_filtered


def main():
    if len(sys.argv) < 2:
        print("Usage: python evaluate_all_properties.py <input_csv> [activity_column] [output_name]")
        print("  activity_column: column name for activity scores (default: kp_activity)")
        print("  output_name: prefix for output files (default: kp_all_real)")
        sys.exit(1)
    
    input_csv = sys.argv[1]
    activity_col = sys.argv[2] if len(sys.argv) > 2 else "kp_activity"
    output_name = sys.argv[3] if len(sys.argv) > 3 else "kp_all_real"
    
    print("=" * 70)
    print(f"Type 2 Evaluation (ALL MOLECULES, ALL REAL PROPERTIES)")
    print(f"Input: {input_csv}")
    print("=" * 70)
    print("\nNO FAKE FILTERS - everything is computed properly!")
    
    # Load CSV
    df = pd.read_csv(input_csv)
    print(f"\nLoaded {len(df)} molecules")
    print(f"Columns: {list(df.columns)}")
    
    if activity_col not in df.columns:
        print(f"\nError: Activity column '{activity_col}' not found in CSV")
        print(f"Available columns: {list(df.columns)}")
        sys.exit(1)
    
    # Compute properties
    df_processed = compute_properties(df)
    
    # Save the processed data with all properties
    output_dir = Path("type2_results")
    output_dir.mkdir(exist_ok=True)
    
    processed_file = output_dir / f"{output_name}_all_properties.csv"
    df_processed.to_csv(processed_file, index=False)
    print(f"\n✓ Saved all properties: {processed_file}")
    
    # Evaluate at each threshold
    for threshold in ACTIVITY_THRESHOLDS:
        print(f"\n{'='*70}")
        print(f"Evaluating at activity threshold {threshold}...")
        print(f"{'='*70}")
        results, df_passing = evaluate_filters(df_processed, activity_col, threshold)
        
        print(f"\n  Final: {results.get('Dedup', 0)} molecules pass all filters")
        
        # Save passing molecules
        output_file = output_dir / f"{output_name}_passing_threshold_{threshold}.csv"
        df_passing.to_csv(output_file, index=False)
        print(f"  ✓ Saved {len(df_passing)} passing molecules: {output_file}")
    
    # Save summary
    summary_file = output_dir / f"{output_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write(f"Type 2 Evaluation Summary (ALL MOLECULES, ALL REAL PROPERTIES)\n")
        f.write(f"==============================================================\n\n")
        f.write(f"Input: {input_csv}\n")
        f.write(f"Activity column: {activity_col}\n")
        f.write(f"Total molecules: {len(df)}\n\n")
        
        # Overall statistics
        f.write(f"Overall Filter Pass Rates:\n")
        f.write(f"-------------------------\n")
        for col, name in [("antibiotics_novelty", "Novelty"), ("toxicity", "Toxicity"),
                         ("antibiotics_motifs_filter", "Motifs"), ("qed", "QED"),
                         ("sa", "SA"), ("mw", "MW"), ("pains", "PAINS"),
                         ("brenk", "BRENK"), ("ring_score", "Ring")]:
            if col in df_processed.columns:
                thresh_val = FILTERS.get(col, 0.5)
                pass_count = (df_processed[col] >= thresh_val).sum()
                total = df_processed[col].notna().sum()
                pct = 100 * pass_count / total if total > 0 else 0
                mean = df_processed[col].mean()
                f.write(f"  {name}: {pass_count}/{total} pass ({pct:.1f}%), mean={mean:.3f}\n")
        
        for threshold in ACTIVITY_THRESHOLDS:
            f.write(f"\n\nActivity Threshold: {threshold}\n")
            f.write("-" * 40 + "\n")
            df_processed_thresh = df_processed[df_processed[activity_col] >= threshold]
            f.write(f"  Activity pass: {len(df_processed_thresh)}\n")
            
            # Count passes for each filter
            for col, name in [("antibiotics_novelty", "Novelty"), ("toxicity", "Toxicity"),
                             ("antibiotics_motifs_filter", "Motifs"), ("qed", "QED"),
                             ("sa", "SA"), ("mw", "MW"), ("pains", "PAINS"),
                             ("brenk", "BRENK"), ("ring_score", "Ring")]:
                if col in df_processed.columns:
                    thresh_val = FILTERS.get(col, 0.5)
                    pass_count = (df_processed_thresh[col] >= thresh_val).sum()
                    total = len(df_processed_thresh)
                    pct = 100 * pass_count / total if total > 0 else 0
                    f.write(f"  {name} pass: {pass_count}/{total} ({pct:.1f}%)\n")
    
    print(f"\n✓ Saved summary: {summary_file}")
    print("\n" + "=" * 70)
    print("✅ Evaluation Complete!")
    print(f"Results saved to: {output_dir.absolute()}")
    print("=" * 70)


if __name__ == "__main__":
    main()



