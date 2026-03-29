#!/usr/bin/env python3
"""
Type 2 Evaluation: Count molecules passing held-out filters for LLM mutation runs.

Uses the computed properties for top 100 DIVERSE molecules.
Diversity is already applied during top100 file creation (greedy selection with Tanimoto < 0.6).
Ranking is done by GEOMETRIC MEAN of novelty, activity, and toxicity.

Tests activity thresholds: 0.05, 0.1
Held-out filters: QED, SA, DeepDL, MW, PAINS, BRENK, ring_score
Also checks: novelty, toxicity, antibiotics_motifs_filter

Generates count tables and plots showing how many molecules pass at each threshold.

CRITICAL CONSISTENCY REQUIREMENT:
- Both plot_filter_cascade() and save_passing_molecules() MUST use the same filter logic
- They share apply_filter_cascade() to ensure consistency
- plot_filter_cascade() validates its results against apply_filter_cascade()
- Old CSV files are deleted before regeneration to prevent stale molecules
- Filter order matters: activity FIRST, then antibiotics_motifs_filter, then other filters, 
  then dedup, then diversity
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import FilterCatalog

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable


# Setup
plt.style.use("default")
sns.set_palette("husl")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11

DATA_DIR = Path("scileo_ablations")
TOP100_DIR = DATA_DIR / "top100_files"
OUTPUT_DIR = Path("type2_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# Directory for saving top 100 diverse molecules with properties
TOP100_SAVE_DIR = Path("top100_diverse_molecules")
TOP100_SAVE_DIR.mkdir(exist_ok=True)

# Activity thresholds to test
ACTIVITY_THRESHOLDS = [0.05, 0.1]

# Top N molecules selected (already done in compute script)
TOP_N = 100

# Held-out filter thresholds
FILTERS = {
    "qed": 0.5,
    "sa": 0.5,
    "deepdl": 0.3,
    "mw": 0.5,
    "pains": 1.0,
    "brenk": 1.0,
    "antibiotics_novelty": 0.6,
    "antibiotics_motifs_filter": 1.0,  # No known antibiotic motifs
    "toxicity": 0.5,
    "ring_score": 1.0,
}

# Global catalogs for PAINS/BRENK (initialized once)
_pains_catalog = None
_brenk_catalog = None
_deepdl_model = None
_deepdl_error = None


def _init_filters():
    """Initialize PAINS and BRENK catalogs once."""
    global _pains_catalog, _brenk_catalog
    if _pains_catalog is None:
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A
        )
        pains_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B
        )
        pains_params.AddCatalog(
            FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C
        )
        _pains_catalog = FilterCatalog.FilterCatalog(pains_params)

    if _brenk_catalog is None:
        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _brenk_catalog = FilterCatalog.FilterCatalog(brenk_params)


def _compute_pains(smiles_list):
    """Compute PAINS filter scores directly (1.0 = no PAINS, 0.0 = has PAINS)."""
    _init_filters()
    results = []
    for smi in tqdm(
        smiles_list, desc="  Computing PAINS", total=len(smiles_list), leave=False
    ):
        if not smi or smi.strip() == "":
            results.append(None)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(None)
            continue
        matches = _pains_catalog.GetMatches(mol)
        results.append(1.0 if len(matches) == 0 else 0.0)
    return results


def _compute_brenk(smiles_list):
    """Compute BRENK filter scores directly (1.0 = no alerts, 0.5 = 1 alert, 0.0 = 2+ alerts)."""
    _init_filters()
    results = []
    for smi in tqdm(
        smiles_list, desc="  Computing BRENK", total=len(smiles_list), leave=False
    ):
        if not smi or smi.strip() == "":
            results.append(None)
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            results.append(None)
            continue
        matches = _brenk_catalog.GetMatches(mol)
        n = len(matches)
        results.append(1.0 if n == 0 else (0.5 if n == 1 else 0.0))
    return results


def _compute_deepdl(smiles_list):
    """Compute DeepDL scores directly (0-1 range, normalized from 0-100)."""
    global _deepdl_model, _deepdl_error

    if _deepdl_model is None and _deepdl_error is None:
        try:
            from druglikeness.deepdl import DeepDL

            _deepdl_model = DeepDL.from_pretrained("extended", device="cpu")
        except Exception as e:
            _deepdl_error = str(e)
            print(f"  Warning: Could not initialize DeepDL model: {e}")
            return [None] * len(smiles_list)

    if _deepdl_model is None:
        return [None] * len(smiles_list)

    # Filter valid SMILES
    valid_indices = []
    valid_smiles = []
    for i, smi in enumerate(smiles_list):
        if smi and smi.strip() and Chem.MolFromSmiles(smi) is not None:
            valid_indices.append(i)
            valid_smiles.append(smi)

    if not valid_smiles:
        return [None] * len(smiles_list)

    try:
        raw_scores = _deepdl_model.screening(
            smiles_list=valid_smiles, naive=True, batch_size=64
        )
        # Normalize 0-100 to 0-1
        normalized = [
            max(0.0, min(1.0, float(s) / 100.0)) if s is not None else None
            for s in raw_scores
        ]
    except Exception as e:
        print(f"  Warning: DeepDL scoring failed: {e}")
        return [None] * len(smiles_list)

    # Reconstruct full list
    results = [None] * len(smiles_list)
    for idx, score in zip(valid_indices, normalized):
        results[idx] = score

    return results


def _compute_qed(smiles_list):
    """Compute QED (drug-likeness) scores."""
    from rdkit import Chem
    from rdkit.Chem import QED

    results = []
    for smi in tqdm(smiles_list, desc="  Computing QED", leave=False):
        if not smi or not smi.strip():
            results.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
            else:
                qed_score = QED.qed(mol)
                results.append(qed_score)
        except Exception:
            results.append(None)
    return results


def _compute_sa(smiles_list):
    """Compute Synthetic Accessibility scores (normalized to 0-1, higher is better)."""
    from rdkit import Chem
    from rdkit.Chem import RDConfig
    import sys
    import os

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    import sascorer

    results = []
    for smi in tqdm(smiles_list, desc="  Computing SA", leave=False):
        if not smi or not smi.strip():
            results.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
            else:
                sa_score = sascorer.calculateScore(mol)
                # SA score ranges from 1 (easy) to 10 (hard), normalize to 0-1 (higher is better)
                normalized = 1.0 - (sa_score - 1.0) / 9.0
                results.append(max(0.0, min(1.0, normalized)))
        except Exception:
            results.append(None)
    return results


def _compute_mw(smiles_list):
    """Compute molecular weight pass/fail (1.0 if 150-500, 0.0 otherwise)."""
    from rdkit import Chem
    from rdkit.Chem import Descriptors

    results = []
    for smi in tqdm(smiles_list, desc="  Computing MW", leave=False):
        if not smi or not smi.strip():
            results.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
            else:
                mw = Descriptors.MolWt(mol)
                # Pass if 150 <= MW <= 500
                results.append(1.0 if 150 <= mw <= 500 else 0.0)
        except Exception:
            results.append(None)
    return results


def _compute_ring_score(smiles_list):
    """Compute ring score (0.0-1.0, normalized from ring system frequencies)."""
    from rdkit import Chem
    import sys
    import math
    from pathlib import Path

    # Add project root to path
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        # Import directly from the file to avoid triggering scorer package __init__.py
        import importlib.util
        ring_systems_path = (
            project_root
            / "modules"
            / "small_molecule_drug_design"
            / "scorer"
            / "ring_systems.py"
        )
        if not ring_systems_path.exists():
            print(f"  Warning: Ring systems module not found: {ring_systems_path}")
            return [None] * len(smiles_list)
        
        spec = importlib.util.spec_from_file_location("ring_systems", ring_systems_path)
        ring_systems_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ring_systems_module)
        RingSystemLookup = ring_systems_module.RingSystemLookup
        ring_systems_min_score = ring_systems_module.ring_systems_min_score
    except Exception as e:
        print(f"  Warning: Could not import ring_systems: {e}")
        return [None] * len(smiles_list)

    # Initialize ring lookup once
    ring_system_csv = (
        project_root
        / "modules"
        / "small_molecule_drug_design"
        / "scorer_mcp"
        / "ring_score_scorer_mcp"
        / "scorer_data"
        / "chembl_ring_systems.csv"
    )
    if not ring_system_csv.exists():
        print(f"  Warning: Ring system CSV not found: {ring_system_csv}")
        return [None] * len(smiles_list)

    ring_lookup = RingSystemLookup(ring_system_csv=str(ring_system_csv))

    def normalize_ring_score(min_freq: int) -> float:
        """Normalize ring frequency to 0-1 scale where 1 = normal, 0 = weird."""
        if min_freq == -1:
            # No rings found - no penalty, score is best
            return 1.0
        if min_freq == 0:
            # Not in database = weird
            return 0.0
        if min_freq >= 1000:
            # Reasonably common = normal (no penalty)
            return 1.0
        if min_freq <= 10:
            # Very rare = weird (penalize)
            return 0.0

        # Linear interpolation on log scale between 10-1000
        log_freq = math.log(min_freq + 1)
        log_max = math.log(1000 + 1)
        log_min = math.log(10 + 1)
        return (log_freq - log_min) / (log_max - log_min)

    results = []
    for smi in tqdm(smiles_list, desc="  Computing ring_score", leave=False):
        if not smi or not smi.strip():
            results.append(None)
            continue
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                results.append(None)
            else:
                # Get ring systems and their frequencies
                freq_list = ring_lookup.process_mol(mol)
                # Get minimum frequency (most unusual ring)
                min_freq = ring_systems_min_score(freq_list)
                # Normalize to 0-1 scale
                score = normalize_ring_score(min_freq)
                results.append(score)
        except Exception:
            results.append(None)
    return results


def _compute_antibiotics_motifs_filter(smiles_list):
    """Compute antibiotics motifs filter (1.0 = no known motifs, 0.0 = has motifs)."""
    import sys
    from pathlib import Path

    # Add project root to path
    script_dir = Path(__file__).resolve().parent
    # Go from ablasions -> small_molecule_drug_design -> modules -> SciLeoAgent (project root)
    project_root = script_dir.parent.parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from modules.small_molecule_drug_design.utils.rdkit_utils import (
            filter_smiles_preserves_existing_hits,
        )
    except ImportError as e:
        print(
            f"    Warning: Could not import filter_smiles_preserves_existing_hits: {e}"
        )
        # Fallback: return None if import fails
        return [None] * len(smiles_list)

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
    try:
        kept_list, reasons_dict = filter_smiles_preserves_existing_hits(valid_smiles)
        kept_set = set(kept_list)
        for idx, smi in zip(valid_indices, valid_smiles):
            results[idx] = 1.0 if smi in kept_set else 0.0
    except Exception as e:
        print(
            f"    Warning: Batch antibiotics motifs filter failed, falling back to individual: {e}"
        )
        # Fallback to individual processing with progress bar
        for idx in tqdm(
            valid_indices,
            desc="  Computing antibiotics_motifs_filter",
            total=len(valid_indices),
            leave=False,
        ):
            smi = smiles_list[idx]
            try:
                kept_list, _ = filter_smiles_preserves_existing_hits([smi])
                # If kept (length 1), no motifs = 1.0; if not kept (length 0), has motifs = 0.0
                results[idx] = 1.0 if len(kept_list) > 0 else 0.0
            except Exception:
                results[idx] = None

    return results


METHOD_COLORS = {
    "SciLeo_old": "#2E86AB",
    "SciLeo_butina_5_objective": "#4A90A4",
    "SciLeo_diverse_5_objective": "#6A9AB4",
    "SciLeo_round_2_0_pct_enamine": "#8B4789",
    "SciLeo_round_2_50_pct_enamine": "#A05195",
    "SciLeo_round_2_80_pct_enamine": "#B45BA1",
    "SciLeo_round_2_100_pct_enamine": "#C865AD",
    "SciLeo_level1_iter1": "#2D9CDB",
    "SciLeo_level1_iter2": "#4AB3E3",
    "SciLeo_level1_iter3": "#5BC5F2",
    "SciLeo_level2_iter1": "#7DD3FC",
    "SciLeo_level2_iter2": "#9FE5FF",
    "REINVENT4": "#A23B72",
    "NatureLM": "#F18F01",
    "MolT5": "#C73E1D",
    "TextGrad": "#6A994E",
}


def apply_diversity_and_dedup(df, threshold=0.6):
    """Apply deduplication and diversity filter to dataframe."""
    if len(df) == 0:
        return df

    # First deduplicate
    df = df.drop_duplicates(subset=["smiles"])

    if len(df) <= 1:
        return df

    # Then apply diversity filter
    from rdkit import Chem
    from rdkit.Chem import AllChem, DataStructs

    smiles_list = df["smiles"].tolist()
    fps = []
    valid_indices = []

    # Compute fingerprints with progress bar
    for idx, smi in enumerate(
        tqdm(smiles_list, desc="  Computing fingerprints", leave=False)
    ):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(fp)
            valid_indices.append(idx)

    if len(fps) == 0:
        return df.iloc[:0]

    selected_indices = [0]
    selected_fps = [fps[0]]

    # Greedy diversity selection with progress bar
    for i in tqdm(range(1, len(fps)), desc="  Diversity selection", leave=False):
        max_sim = max(
            DataStructs.TanimotoSimilarity(fps[i], sel_fp) for sel_fp in selected_fps
        )
        if max_sim < threshold:
            selected_indices.append(i)
            selected_fps.append(fps[i])

    original_indices = [valid_indices[i] for i in selected_indices]
    return df.iloc[original_indices].copy()


def load_llm_mutation(filename, target):
    """Load LLM mutation file with computed properties."""
    # Try computed version from top100_files folder
    computed_file = TOP100_DIR / filename.replace(".csv", "_top100_computed.csv")
    if computed_file.exists():
        df = pd.read_csv(computed_file)
    else:
        print(f"  Warning: Computed file not found: {computed_file}")
        return None, None

    seed = filename.split("_")[-3]

    if target == "ecoli":
        target_col = "escherichia_coli_minimol"
    else:
        target_col = "klebsiella_pneumoniae_minimol"

    df["target_activity"] = df[target_col]

    if "primary_1_minus_cell_toxicity_chemprop" in df.columns:
        df["toxicity"] = df["primary_1_minus_cell_toxicity_chemprop"]

    return df, seed


def load_scileo_seed(filename, target):
    """Load single SciLeo seed - uses pre-computed diverse top 100 if available."""
    # Try loading pre-computed diverse top 100 from top100_files folder
    computed_file = TOP100_DIR / filename.replace(".csv", "_top100_computed.csv")
    needs_computation = False
    needs_reselection = False
    if computed_file.exists():
        df_computed = pd.read_csv(computed_file)
        seed = filename.split("_")[-3]
        
        # CRITICAL: ALWAYS drop ALL computed filter columns from cached file to force recomputation
        # Cached files may have stale values from old SMARTS patterns or outdated logic
        # This ensures we ALWAYS recompute with the latest filter logic
        computed_filter_props = [
            "qed",
            "sa",
            "mw",
            "pains",
            "brenk",
            "deepdl",
            "antibiotics_motifs_filter",
            "ring_score",
        ]
        dropped_cols = []
        for prop in computed_filter_props:
            if prop in df_computed.columns:
                df_computed = df_computed.drop(columns=[prop])
                dropped_cols.append(prop)
        if dropped_cols:
            print(f"  Dropped stale computed filters from cached file (will recompute with latest logic): {dropped_cols}")

        # If computed file has fewer than expected molecules, we need to re-select from full dataset
        if len(df_computed) < TOP_N:
            needs_reselection = True
            needs_computation = True
            print(
                f"  Computed file has only {len(df_computed)} molecules, need to re-select top {TOP_N} from full dataset"
            )
            df = None
        else:
            df = df_computed
            # Check if any required properties are missing or all NaN
            required_props = ["pains", "brenk", "deepdl", "antibiotics_motifs_filter", "ring_score"]
            missing_props = []
            for prop in required_props:
                if prop not in df.columns or df[prop].isna().all():
                    missing_props.append(prop)

            # ALWAYS recompute antibiotics_motifs_filter to avoid stale data when patterns change
            # (Already dropped above, but ensure it's in missing_props)
            if "antibiotics_motifs_filter" not in missing_props:
                missing_props.append("antibiotics_motifs_filter")

            if missing_props:
                needs_computation = True
                print(
                    f"  Computed file exists but missing properties: {missing_props}, computing on existing {len(df)} molecules..."
                )
    else:
        needs_computation = True
        needs_reselection = True
        df = None

    if needs_computation:
        # If we need to re-select or don't have a valid computed file, load full dataset
        if needs_reselection or df is None:
            df = pd.read_csv(DATA_DIR / filename)
            seed = filename.split("_")[-3]
            
            # CRITICAL: ALWAYS drop ALL computed filter columns from full dataset to force recomputation
            # The full dataset may have stale values from old SMARTS patterns or outdated logic
            # This ensures we ALWAYS recompute with the latest filter logic
            computed_filter_props = [
                "qed",
                "sa", 
                "mw",
                "pains",
                "brenk",
                "deepdl",
                "antibiotics_motifs_filter",
                "ring_score",
            ]
            for prop in computed_filter_props:
                if prop in df.columns:
                    df = df.drop(columns=[prop])
                    print(f"  Dropped stale {prop} from full dataset (will recompute with latest logic)")

        # REQUIRED: Compute ALL missing properties - fail if we cannot
        all_required_props = [
            "qed",
            "sa",
            "mw",
            "pains",
            "brenk",
            "deepdl",
            "antibiotics_motifs_filter",
            "ring_score",
        ]
        missing_props = []
        for prop in all_required_props:
            if prop not in df.columns or df[prop].isna().all():
                missing_props.append(prop)
        
        # CRITICAL: ALWAYS ensure all computed filters are recomputed (already dropped above, but ensure they're in missing_props)
        for prop in all_required_props:
            if prop not in missing_props:
                missing_props.append(prop)

        # FIRST: Select top 100 diverse BEFORE computing expensive properties
        # Normalize toxicity column first
        if (
            "toxicity_safety_chemprop" in df.columns
            and "primary_1_minus_cell_toxicity_chemprop" not in df.columns
        ):
            df["primary_1_minus_cell_toxicity_chemprop"] = df[
                "toxicity_safety_chemprop"
            ]

        if target == "ecoli":
            target_col = "escherichia_coli_minimol"
        else:
            target_col = "klebsiella_pneumoniae_minimol"

        df["target_activity"] = df[target_col]

        if "primary_1_minus_cell_toxicity_chemprop" in df.columns:
            df["toxicity"] = df["primary_1_minus_cell_toxicity_chemprop"]
        elif "toxicity_safety_chemprop" in df.columns:
            df["toxicity"] = df["toxicity_safety_chemprop"]
        else:
            raise ValueError(f"No toxicity column found in {filename}")

        # Calculate aggregated score as GEOMETRIC MEAN
        epsilon = 1e-10
        df["aggregated_score"] = (
            (df["target_activity"].fillna(0) + epsilon)
            * (df["antibiotics_novelty"].fillna(0) + epsilon)
            * (df["toxicity"].fillna(0) + epsilon)
        ) ** (1 / 3)

        # Select top candidates by ORIGINAL aggregate score from optimizer, or aggregated_score if aggregate doesn't exist
        sort_column = "aggregate" if "aggregate" in df.columns else "aggregated_score"
        df_sorted = df.sort_values(sort_column, ascending=False)
        # Take MANY more candidates before diversity filtering to ensure we get 100 diverse
        # Start with TOP_N * 20, but expand if we don't get enough diverse molecules
        max_candidates = min(len(df_sorted), TOP_N * 100)  # Increased from TOP_N * 20
        candidates = df_sorted.head(max_candidates)
        df_top100 = apply_diversity_and_dedup(candidates, threshold=0.6)
        df_top100 = df_top100.head(TOP_N).copy()

        print(f"  Selected top {len(df_top100)} diverse molecules from {len(df)} total")

        # NOW compute ALL missing properties on the top 100 diverse molecules
        if missing_props:
            print(
                f"  Computing missing properties for {filename} (on {len(df_top100)} molecules): {missing_props}"
            )
            # Compute valid SMILES and indices ONCE before the loop to ensure consistency
            valid_smiles = df_top100["smiles"].dropna()
            valid_indices = valid_smiles.index.tolist()
            smiles_list = valid_smiles.tolist()
            if not smiles_list:
                raise ValueError(
                    f"No valid SMILES found in top 100 diverse for {filename}"
                )

            # Compute properties directly
            prop_computers = {
                "qed": _compute_qed,
                "sa": _compute_sa,
                "mw": _compute_mw,
                "pains": _compute_pains,
                "brenk": _compute_brenk,
                "deepdl": _compute_deepdl,
                "antibiotics_motifs_filter": _compute_antibiotics_motifs_filter,
                "ring_score": _compute_ring_score,
            }

            for prop in missing_props:
                if prop in prop_computers:
                    # Use the pre-computed smiles_list and valid_indices to ensure alignment
                    scores = prop_computers[prop](smiles_list)
                    # CRITICAL: Validate scores length matches SMILES list length
                    if len(scores) != len(smiles_list):
                        raise ValueError(
                            f"Property {prop} returned {len(scores)} scores but expected {len(smiles_list)} for {filename}"
                        )
                    # Assign scores to the correct rows using valid_indices
                    df_top100.loc[valid_indices, prop] = scores
                    
                    # Ensure antibiotics_motifs_filter is numeric (not boolean) to avoid CSV save/load issues
                    if prop == "antibiotics_motifs_filter":
                        df_top100[prop] = pd.to_numeric(df_top100[prop], errors="coerce").astype(float)
                    
                    # CRITICAL: Verify assignment worked correctly by checking assigned values match computed scores
                    assigned_values = df_top100.loc[valid_indices, prop].tolist()
                    if len(assigned_values) != len(scores):
                        raise ValueError(
                            f"Assignment failed for {prop}: {len(assigned_values)} values assigned but {len(scores)} scores computed for {filename}"
                        )
                    # For boolean filters, verify the values match (allow for small floating point differences)
                    if prop == "antibiotics_motifs_filter":
                        assigned_pass = sum(1 for v in assigned_values if v >= 1.0)
                        computed_pass = sum(1 for s in scores if s == 1.0)
                        if assigned_pass != computed_pass:
                            raise ValueError(
                                f"Assignment mismatch for {prop}: {assigned_pass} passes assigned but {computed_pass} passes computed for {filename}"
                            )

                    # Verify computation
                    non_null_count = df_top100[prop].notna().sum()
                    if non_null_count == 0:
                        # Allow DeepDL to fail gracefully if model is not available
                        if prop == "deepdl":
                            print(f"    Warning: {prop} could not be computed (model unavailable), all values set to NaN")
                        else:
                            raise ValueError(
                                f"Property {prop} computed but all values are NaN for {filename}"
                            )
                    else:
                        print(f"    {prop}: {non_null_count}/{len(smiles_list)} computed (verified)")
                else:
                    raise ValueError(
                        f"No computation function available for property {prop}"
                    )

        df = df_top100

        # FINAL ASSERTION: All required properties MUST be present and have non-null values
        for prop in all_required_props:
            assert (
                prop in df.columns
            ), f"REQUIRED property {prop} is missing in {filename}"
            non_null = df[prop].notna().sum()
            # Allow deepdl to be all NaN if model is unavailable
            if prop != "deepdl":
                assert (
                    non_null > 0
                ), f"REQUIRED property {prop} has no non-null values in {filename} (found {non_null}/{len(df)} non-null)"

        print(f"  ✓ All required properties verified for {len(df)} molecules")

        # Save the top 100 diverse molecules with all computed properties
        # Ensure antibiotics_motifs_filter is numeric before saving to avoid CSV save/load issues
        if "antibiotics_motifs_filter" in df.columns:
            df["antibiotics_motifs_filter"] = pd.to_numeric(df["antibiotics_motifs_filter"], errors="coerce").astype(float)
        
        save_filename = filename.replace("_all_molecules.csv", "_top100_diverse.csv")
        save_path = TOP100_SAVE_DIR / save_filename
        df.to_csv(save_path, index=False)
        print(f"  ✓ Saved top 100 diverse to: {save_path.name}")

    # Ensure target_activity column exists
    if target == "ecoli":
        target_col = "escherichia_coli_minimol"
    else:
        target_col = "klebsiella_pneumoniae_minimol"

    if "target_activity" not in df.columns:
        df["target_activity"] = df[target_col]

    # Normalize toxicity column if needed
    if (
        "toxicity_safety_chemprop" in df.columns
        and "primary_1_minus_cell_toxicity_chemprop" not in df.columns
    ):
        df["primary_1_minus_cell_toxicity_chemprop"] = df["toxicity_safety_chemprop"]

    if "toxicity" not in df.columns:
        if "primary_1_minus_cell_toxicity_chemprop" in df.columns:
            df["toxicity"] = df["primary_1_minus_cell_toxicity_chemprop"]
        elif "toxicity_safety_chemprop" in df.columns:
            df["toxicity"] = df["toxicity_safety_chemprop"]

    # Final assertion: Check essential properties (computed + required)
    computed_props = ["pains", "brenk", "deepdl", "antibiotics_motifs_filter", "ring_score"]
    for prop in computed_props:
        assert prop in df.columns, f"Computed property {prop} missing in {filename}"
        non_null = df[prop].notna().sum()
        # Allow deepdl to be all NaN if model is unavailable
        if prop != "deepdl":
            assert (
                non_null > 0
            ), f"Computed property {prop} has no non-null values in {filename} (found {non_null}/{len(df)} non-null)"

    # Essential properties that should exist (warn if missing but don't fail)
    essential_props = ["toxicity", "antibiotics_novelty", "target_activity"]
    for prop in essential_props:
        if prop not in df.columns:
            print(f"  Warning: Essential property {prop} missing in {filename}")
        elif df[prop].notna().sum() == 0:
            print(
                f"  Warning: Essential property {prop} has no non-null values in {filename}"
            )

    return df, seed


def load_baseline_data(method, target, limit_reinvent=True):
    """Load baseline method data - uses pre-computed diverse top 100 if available."""
    files = {
        ("REINVENT4", "ecoli"): "stage1_1_reinvent4_ecoli_scored.csv",
        ("REINVENT4", "kpneumoniae"): "stage1_1_reinvent4_kp_scored.csv",
        ("NatureLM", "ecoli"): "naturelm_ecoli_antibiot_final_scored.csv",
        ("NatureLM", "kpneumoniae"): "naturelm_KP_antibiot_final_scored_results.csv",
        ("MolT5", "ecoli"): "molt5_ecoli_antibiot_final_scored.csv",
        ("MolT5", "kpneumoniae"): "molt5_KP_antibiot_final_scored_results.csv",
        ("TextGrad", "ecoli"): "textgrad_ecoli_output_new_epoch_merged_scored.csv",
        ("TextGrad", "kpneumoniae"): "textgrad_kp_output_new_epoch_merged_scored.csv",
    }

    # Try top100_computed diverse version first, fall back to original
    filename = files[(method, target)]
    computed_file = TOP100_DIR / filename.replace(".csv", "_top100_computed.csv")
    if computed_file.exists():
        df = pd.read_csv(computed_file)
        
        # CRITICAL: ALWAYS drop ALL computed filter columns from cached file to force recomputation
        # Cached files may have stale values from old SMARTS patterns or outdated logic
        # This ensures we ALWAYS recompute with the latest filter logic
        computed_filter_props = [
            "qed",
            "sa",
            "mw",
            "pains",
            "brenk",
            "deepdl",
            "antibiotics_motifs_filter",
            "ring_score",
        ]
        dropped_cols = []
        needs_recompute = False
        for prop in computed_filter_props:
            if prop in df.columns:
                df = df.drop(columns=[prop])
                dropped_cols.append(prop)
                needs_recompute = True
        if dropped_cols:
            print(f"  Dropped stale computed filters from {method} {target} cached file (will recompute with latest logic): {dropped_cols}")
        # Check if ring_score is missing or all NaN - if so, recompute it
        if "ring_score" not in df.columns or df["ring_score"].isna().all():
            print(f"  Warning: {method} {target} pre-computed file missing ring_score, will recompute")
            df = None  # Force recomputation
        elif needs_recompute:
            # Force recomputation of all dropped filters
            print(f"  Recomputing dropped filters for {method} {target} (patterns/logic may have changed)")
            # Compute all dropped filters on existing df
            smiles_list = df["smiles"].dropna().tolist()
            if smiles_list:
                # Compute all dropped filters
                prop_computers = {
                    "qed": _compute_qed,
                    "sa": _compute_sa,
                    "mw": _compute_mw,
                    "pains": _compute_pains,
                    "brenk": _compute_brenk,
                    "deepdl": _compute_deepdl,
                    "antibiotics_motifs_filter": _compute_antibiotics_motifs_filter,
                    "ring_score": _compute_ring_score,
                }
                for prop in dropped_cols:
                    if prop in prop_computers:
                        scores = prop_computers[prop](smiles_list)
                        df[prop] = pd.to_numeric(scores, errors="coerce").astype(float) if prop in ["antibiotics_motifs_filter"] else pd.to_numeric(scores, errors="coerce")
                        print(f"    {prop}: {df[prop].notna().sum()}/{len(smiles_list)} computed")
            # Ensure target_activity column exists
            if target == "ecoli":
                target_col = "escherichia_coli"
            else:
                target_col = "klebsiella_pneumoniae"
            if "target_activity" not in df.columns:
                df["target_activity"] = df[target_col]
            # Final assertion: Check all required properties exist
            all_required_props_final = [
                "pains", "brenk", "deepdl", "antibiotics_motifs_filter", "ring_score",
                "qed", "sa", "mw", "toxicity", "antibiotics_novelty", "target_activity",
            ]
            for prop in all_required_props_final:
                assert prop in df.columns, f"Required property {prop} missing in {method} {target}"
                non_null = df[prop].notna().sum()
                assert non_null > 0, f"Required property {prop} has no non-null values in {method} {target} (found {non_null}/{len(df)} non-null)"
            return df
        else:
            # Pre-computed file is good, return early after validation
            # Ensure target_activity column exists
            if target == "ecoli":
                target_col = "escherichia_coli"
            else:
                target_col = "klebsiella_pneumoniae"
            if "target_activity" not in df.columns:
                df["target_activity"] = df[target_col]
            # Final assertion: Check all required properties exist
            all_required_props_final = [
                "pains", "brenk", "deepdl", "antibiotics_motifs_filter", "ring_score",
                "qed", "sa", "mw", "toxicity", "antibiotics_novelty", "target_activity",
            ]
            for prop in all_required_props_final:
                assert prop in df.columns, f"Required property {prop} missing in {method} {target}"
                non_null = df[prop].notna().sum()
                assert non_null > 0, f"Required property {prop} has no non-null values in {method} {target} (found {non_null}/{len(df)} non-null)"
            return df
    
    # Fallback: load full file and compute on the fly (old behavior)
    df = None  # Initialize df to None if it wasn't set above
    if df is None:
        df = pd.read_csv(DATA_DIR / filename)

        # Limit REINVENT4 to first 10k before selecting top N
        if method == "REINVENT4" and limit_reinvent and len(df) > 10000:
            df = df.head(10000).copy()

        # CRITICAL: ALWAYS drop ALL computed filter columns from full dataset to force recomputation
        # The full dataset may have stale values from old SMARTS patterns or outdated logic
        # This ensures we ALWAYS recompute with the latest filter logic
        computed_filter_props = [
            "qed",
            "sa",
            "mw",
            "pains",
            "brenk",
            "deepdl",
            "antibiotics_motifs_filter",
            "ring_score",
        ]
        for prop in computed_filter_props:
            if prop in df.columns:
                df = df.drop(columns=[prop])
                print(f"  Dropped stale {prop} from {method} {target} full dataset (will recompute with latest logic)")

        if target == "ecoli":
            target_col = "escherichia_coli"
        else:
            target_col = "klebsiella_pneumoniae"

        df["target_activity"] = df[target_col]

        # REQUIRED: Compute ALL missing properties
        all_required_props = [
            "qed",
            "sa",
            "mw",
            "pains",
            "brenk",
            "deepdl",
            "antibiotics_motifs_filter",
            "ring_score",
        ]
        missing_props = []
        for prop in all_required_props:
            if prop not in df.columns or df[prop].isna().all():
                missing_props.append(prop)
        
        # CRITICAL: ALWAYS ensure all computed filters are recomputed (already dropped above, but ensure they're in missing_props)
        for prop in all_required_props:
            if prop not in missing_props:
                missing_props.append(prop)

        # FIRST: Select top 100 diverse BEFORE computing expensive properties
        if target == "ecoli":
            target_col = "escherichia_coli"
        else:
            target_col = "klebsiella_pneumoniae"

        df["target_activity"] = df[target_col]

        # Calculate aggregated score as GEOMETRIC MEAN
        epsilon = 1e-10
        df["aggregated_score"] = (
            (df["target_activity"].fillna(0) + epsilon)
            * (df["antibiotics_novelty"].fillna(0) + epsilon)
            * (df["toxicity"].fillna(0) + epsilon)
        ) ** (1 / 3)

        # Select top candidates by ORIGINAL aggregate score from optimizer, or aggregated_score if aggregate doesn't exist
        sort_column = "aggregate" if "aggregate" in df.columns else "aggregated_score"
        df_sorted = df.sort_values(sort_column, ascending=False)
        # Take MANY more candidates before diversity filtering to ensure we get 100 diverse
        # Start with TOP_N * 20, but expand if we don't get enough diverse molecules
        max_candidates = min(len(df_sorted), TOP_N * 100)  # Increased from TOP_N * 20
        candidates = df_sorted.head(max_candidates)
        df_top100 = apply_diversity_and_dedup(candidates, threshold=0.6)
        df_top100 = df_top100.head(TOP_N).copy()

        print(f"  Selected top {len(df_top100)} diverse molecules from {len(df)} total")

        # NOW compute ALL missing properties on the top 100 diverse molecules
        if missing_props:
            print(
                f"  Computing missing properties for {method} {target} (on {len(df_top100)} molecules): {missing_props}"
            )
            # Compute valid SMILES and indices ONCE before the loop to ensure consistency
            valid_smiles = df_top100["smiles"].dropna()
            valid_indices = valid_smiles.index.tolist()
            smiles_list = valid_smiles.tolist()
            if not smiles_list:
                raise ValueError(
                    f"No valid SMILES found in top 100 diverse for {filename}"
                )

            # Compute properties directly
            prop_computers = {
                "qed": _compute_qed,
                "sa": _compute_sa,
                "mw": _compute_mw,
                "pains": _compute_pains,
                "brenk": _compute_brenk,
                "deepdl": _compute_deepdl,
                "antibiotics_motifs_filter": _compute_antibiotics_motifs_filter,
                "ring_score": _compute_ring_score,
            }

            for prop in missing_props:
                if prop in prop_computers:
                    # Use the pre-computed smiles_list and valid_indices to ensure alignment
                    scores = prop_computers[prop](smiles_list)
                    # CRITICAL: Validate scores length matches SMILES list length
                    if len(scores) != len(smiles_list):
                        raise ValueError(
                            f"Property {prop} returned {len(scores)} scores but expected {len(smiles_list)} for {method} {target}"
                        )
                    # Assign scores to the correct rows using valid_indices
                    df_top100.loc[valid_indices, prop] = scores
                    
                    # Ensure antibiotics_motifs_filter is numeric (not boolean) to avoid CSV save/load issues
                    if prop == "antibiotics_motifs_filter":
                        df_top100[prop] = pd.to_numeric(df_top100[prop], errors="coerce").astype(float)
                    
                    # CRITICAL: Verify assignment worked correctly by checking assigned values match computed scores
                    assigned_values = df_top100.loc[valid_indices, prop].tolist()
                    if len(assigned_values) != len(scores):
                        raise ValueError(
                            f"Assignment failed for {prop}: {len(assigned_values)} values assigned but {len(scores)} scores computed for {method} {target}"
                        )
                    # For boolean filters, verify the values match (allow for small floating point differences)
                    if prop == "antibiotics_motifs_filter":
                        assigned_pass = sum(1 for v in assigned_values if v >= 1.0)
                        computed_pass = sum(1 for s in scores if s == 1.0)
                        if assigned_pass != computed_pass:
                            raise ValueError(
                                f"Assignment mismatch for {prop}: {assigned_pass} passes assigned but {computed_pass} passes computed for {method} {target}"
                            )

                    # Verify computation
                    non_null_count = df_top100[prop].notna().sum()
                    if non_null_count == 0:
                        # Allow DeepDL to fail gracefully if model is not available
                        if prop == "deepdl":
                            print(f"    Warning: {prop} could not be computed (model unavailable), all values set to NaN")
                        else:
                            raise ValueError(
                                f"Property {prop} computed but all values are NaN for {method} {target}"
                            )
                    else:
                        print(f"    {prop}: {non_null_count}/{len(smiles_list)} computed (verified)")
                else:
                    raise ValueError(
                        f"No computation function available for property {prop}"
                    )

        df = df_top100

        # FINAL ASSERTION: All required properties MUST be present and have non-null values
        for prop in all_required_props:
            assert (
                prop in df.columns
            ), f"REQUIRED property {prop} is missing for {method} {target}"
            non_null = df[prop].notna().sum()
            # Allow deepdl to be all NaN if model is unavailable
            if prop != "deepdl":
                assert (
                    non_null > 0
                ), f"REQUIRED property {prop} has no non-null values for {method} {target} (found {non_null}/{len(df)} non-null)"

        print(f"  ✓ All required properties verified for {len(df)} molecules")

        # Save the top 100 diverse molecules with all computed properties
        # Ensure antibiotics_motifs_filter is numeric before saving to avoid CSV save/load issues
        if "antibiotics_motifs_filter" in df.columns:
            df["antibiotics_motifs_filter"] = pd.to_numeric(df["antibiotics_motifs_filter"], errors="coerce").astype(float)
        
        save_filename = f"{method}_{target}_top100_diverse.csv"
        save_path = TOP100_SAVE_DIR / save_filename
        df.to_csv(save_path, index=False)
        print(f"  ✓ Saved top 100 diverse to: {save_path.name}")

    # Ensure target_activity column exists
    if target == "ecoli":
        target_col = "escherichia_coli"
    else:
        target_col = "klebsiella_pneumoniae"

    if "target_activity" not in df.columns:
        df["target_activity"] = df[target_col]

    # Final assertion: All required properties must exist and have non-null values
    all_required_props = [
        "pains",
        "brenk",
        "deepdl",
        "antibiotics_motifs_filter",
        "ring_score",
        "qed",
        "sa",
        "mw",
        "toxicity",
        "antibiotics_novelty",
        "target_activity",
    ]
    for prop in all_required_props:
        assert (
            prop in df.columns
        ), f"Required property {prop} missing in {method} {target}"
        non_null = df[prop].notna().sum()
        # Allow deepdl to be all NaN if model is unavailable
        if prop != "deepdl":
            assert (
                non_null > 0
            ), f"Required property {prop} has no non-null values in {method} {target} (found {non_null}/{len(df)} non-null)"

    return df


def apply_filter_cascade(df, activity_threshold, apply_diversity=True):
    """
    Apply filter cascade in the EXACT same order as plot_filter_cascade.
    
    CRITICAL: This function MUST be used by both save_passing_molecules and 
    plot_filter_cascade to ensure consistency. Filter order matters!
    
    Order:
    1. Activity filter FIRST
    2. antibiotics_motifs_filter SECOND  
    3. Remaining filters
    4. Deduplication
    5. Diversity filter (if apply_diversity=True)
    
    Args:
        df: DataFrame with molecules
        activity_threshold: Activity threshold to apply
        apply_diversity: Whether to apply diversity filter (Tanimoto < 0.6)
    
    Returns:
        filtered DataFrame
    """
    filtered = df.copy()
    
    # Apply activity FIRST
    if "target_activity" in filtered.columns:
        filtered["target_activity"] = pd.to_numeric(
            filtered["target_activity"], errors="coerce"
        )
        filtered = filtered[
            filtered["target_activity"].notna()
            & (filtered["target_activity"] >= activity_threshold)
        ]
    
    # Apply antibiotics_motifs_filter SECOND
    if (
        "antibiotics_motifs_filter" in filtered.columns
        and "antibiotics_motifs_filter" in FILTERS
    ):
        col = filtered["antibiotics_motifs_filter"].copy()
        col = col.replace(
            {
                True: 1.0,
                False: 0.0,
                "True": 1.0,
                "False": 0.0,
                "1.0": 1.0,
                "0.0": 0.0,
                "1": 1.0,
                "0": 0.0,
            }
        )
        filtered["antibiotics_motifs_filter"] = pd.to_numeric(
            col, errors="coerce"
        )
        filtered = filtered[
            filtered["antibiotics_motifs_filter"].notna()
            & (
                filtered["antibiotics_motifs_filter"]
                >= FILTERS["antibiotics_motifs_filter"]
            )
        ]
    
    # Apply remaining filters (excluding antibiotics_motifs_filter)
    remaining_filters = {
        k: v
        for k, v in FILTERS.items()
        if k not in ["antibiotics_motifs_filter"]
    }
    
    for col, threshold in remaining_filters.items():
        if col in filtered.columns:
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
            filtered = filtered[
                filtered[col].notna() & (filtered[col] >= threshold)
            ]
    
    # Remove duplicates
    if len(filtered) > 0:
        filtered = filtered.drop_duplicates(subset=["smiles"])
    
    # Apply diversity filter (Tanimoto < 0.6) - same as plot_filter_cascade
    if apply_diversity and len(filtered) > 0:
        from rdkit import Chem
        from rdkit.Chem import AllChem, DataStructs
        
        smiles_list = filtered["smiles"].tolist()
        fps = []
        valid_indices = []
        
        for idx_inner, smi in enumerate(smiles_list):
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                fp = AllChem.GetMorganFingerprintAsBitVect(
                    mol, radius=2, nBits=2048
                )
                fps.append(fp)
                valid_indices.append(idx_inner)
        
        if len(fps) > 0:
            selected_indices = [0]
            selected_fps = [fps[0]]
            
            for i in range(1, len(fps)):
                max_sim = max(
                    DataStructs.TanimotoSimilarity(fps[i], sel_fp)
                    for sel_fp in selected_fps
                )
                if max_sim < 0.6:
                    selected_indices.append(i)
                    selected_fps.append(fps[i])
            
            original_indices = [valid_indices[i] for i in selected_indices]
            filtered = filtered.iloc[original_indices].copy()
    
    return filtered


def count_passing(df, activity_threshold, return_df=False):
    """Count molecules passing all filters at given activity threshold.
    
    NOTE: For saving molecules, use apply_filter_cascade() instead to ensure
    consistency with plot_filter_cascade.
    """
    filtered = df.copy()
    initial_count = len(filtered)

    # Apply all filters
    for col, threshold in FILTERS.items():
        if col in filtered.columns:
            # Convert True/False strings to numeric before converting
            # Only replace if column is not already numeric
            if filtered[col].dtype == 'object':
                filtered[col] = filtered[col].replace(
                    {True: 1.0, False: 0.0, "True": 1.0, "False": 0.0}
                )
            filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
            # Filter out NaN values and apply threshold
            # NEVER skip filters - all NaN means all molecules fail this filter
            filtered = filtered[filtered[col].notna() & (filtered[col] >= threshold)]

    # Apply activity threshold
    if "target_activity" in filtered.columns:
        filtered["target_activity"] = pd.to_numeric(
            filtered["target_activity"], errors="coerce"
        )
        filtered = filtered[
            filtered["target_activity"].notna()
            & (filtered["target_activity"] >= activity_threshold)
        ]

    if return_df:
        return filtered, len(filtered), initial_count
    return len(filtered), initial_count


def count_per_filter(df, activity_threshold):
    """Count pass rate for each individual filter. All filters checked on full dataset, not pre-filtered by activity."""
    results = {}
    initial_count = len(df)

    # Check activity filter on full dataset
    if "target_activity" in df.columns:
        activity_col = pd.to_numeric(df["target_activity"], errors="coerce")
        activity_passing = (
            activity_col.notna() & (activity_col >= activity_threshold)
        ).sum()
        results["activity"] = {
            "passing": activity_passing,
            "total": initial_count,
            "pass_rate": (activity_passing / initial_count * 100)
            if initial_count > 0
            else 0,
        }
    else:
        results["activity"] = {
            "passing": 0,
            "total": initial_count,
            "pass_rate": 0,
        }

    # Check each other filter on full dataset (not pre-filtered by activity)
    for filter_name, threshold in FILTERS.items():
        if filter_name in df.columns:
            # Convert True/False strings to numeric before converting
            col = df[filter_name].copy()
            # Only replace if column is not already numeric
            if not pd.api.types.is_numeric_dtype(col):
                col = col.replace(
                    {
                        True: 1.0,
                        False: 0.0,
                        "True": 1.0,
                        "False": 0.0,
                        "1.0": 1.0,
                        "0.0": 0.0,
                        "1": 1.0,
                        "0": 0.0,
                    }
                )
            col = pd.to_numeric(col, errors="coerce")
            passing = (col.notna() & (col >= threshold)).sum()
            results[filter_name] = {
                "passing": passing,
                "total": initial_count,
                "pass_rate": (passing / initial_count * 100)
                if initial_count > 0
                else 0,
            }
        else:
            results[filter_name] = {
                "passing": 0,
                "total": initial_count,
                "pass_rate": 0,
            }

    return results


def evaluate_method(method_name, data):
    """Evaluate a method at all activity thresholds."""
    results = []
    for act_thresh in ACTIVITY_THRESHOLDS:
        passing, total = count_passing(data, act_thresh)
        results.append(
            {
                "method": method_name,
                "activity_threshold": act_thresh,
                "passing": passing,
                "total": total,
                "pass_rate": (passing / total * 100) if total > 0 else 0,
            }
        )
    return results


def generate_count_table(target):
    """Generate table of passing molecule counts."""
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    print(f"\n{'='*70}")
    print(f"Counting Passing Molecules - {target_name} (Top {TOP_N} Diverse)")
    print("=" * 70)

    all_results = []

    # SciLeo methods (level1 and level2 for E. coli, level1 for KP)
    if target == "ecoli":
        scileo_files = [
            # Level 1 iterations 1, 2, and 3
            ("ecoli_level1_iter1_20251101150814_all_molecules.csv", "level1_iter1"),
            ("ecoli_level1_iter2_20251102002019_all_molecules.csv", "level1_iter2"),
            ("ecoli_level1_iter3_20251102201621_all_molecules.csv", "level1_iter3"),
            # Level 2 iterations 1 and 2
            ("ecoli_level2_iter1_20251101150813_all_molecules.csv", "level2_iter1"),
            ("ecoli_level2_iter2_20251102002529_all_molecules.csv", "level2_iter2"),
        ]
    elif target == "kpneumoniae":
        scileo_files = [
            # Level 1 iterations 1 and 2
            ("kp_level1_iter1_20251104162127_all_molecules.csv", "level1_iter1"),
            ("kp_level1_iter2_20251105131705_all_molecules.csv", "level1_iter2"),
        ]
    else:
        scileo_files = []

    # Process SciLeo methods
    print(f"\nProcessing {len(scileo_files)} SciLeo methods (level 1 and level 2)...")
    for filename, method_suffix in scileo_files:
        data, _ = load_scileo_seed(filename, target)
        method_name = f"SciLeo_{method_suffix}"
        print(f"\nEvaluating {method_name}...")
        results = evaluate_method(method_name, data)
        all_results.extend(results)

    # Baselines
    for method in ["REINVENT4", "NatureLM", "MolT5", "TextGrad"]:
        print(f"\nEvaluating {method}...")
        try:
            data = load_baseline_data(method, target, limit_reinvent=True)
            results = evaluate_method(method, data)
            all_results.extend(results)
        except Exception as e:
            raise RuntimeError(f"Failed to process {method}: {e}") from e

    df_results = pd.DataFrame(all_results)

    # Save to CSV
    output_file = OUTPUT_DIR / f"passing_counts_{target}.csv"
    # CRITICAL: Delete old CSV file to ensure fresh data
    if output_file.exists():
        output_file.unlink()
    
    if len(df_results) > 0:
        df_results.to_csv(output_file, index=False)
        print(f"\n✓ Saved counts to: {output_file.name}")
    else:
        # Create empty DataFrame with correct columns
        empty_df = pd.DataFrame(columns=["method", "activity_threshold", "passing", "total", "pass_rate"])
        empty_df.to_csv(output_file, index=False)
        print(f"\n✓ Saved empty counts to: {output_file.name}")

    # Print table
    print(f"\n{'='*70}")
    print(f"Passing Molecule Counts - {target_name} (Top {TOP_N} Diverse)")
    print("=" * 70)
    print(f"\n{'Method':<20s} ", end="")
    for thresh in ACTIVITY_THRESHOLDS:
        print(f"Act≥{thresh:<5.2f} ", end="")
    print()
    print("-" * 70)

    # Get all methods
    if target == "ecoli":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
            "SciLeo_level1_iter3",
            "SciLeo_level2_iter1",
            "SciLeo_level2_iter2",
        ]
    elif target == "kpneumoniae":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
        ]
    else:
        scileo_methods = []

    all_methods = scileo_methods + ["REINVENT4", "NatureLM", "MolT5", "TextGrad"]

    for method in all_methods:
        if len(df_results) == 0:
            print(f"{method:<20s} ", end="")
            for _ in ACTIVITY_THRESHOLDS:
                print(f"{'N/A':<11s} ", end="")
            print()
            continue
        method_data = df_results[df_results["method"] == method]
        print(f"{method:<20s} ", end="")
        for thresh in ACTIVITY_THRESHOLDS:
            row = method_data[method_data["activity_threshold"] == thresh]
            if len(row) > 0:
                passing = row["passing"].values[0]
                print(f"{passing:<11d} ", end="")
            else:
                print(f"{'N/A':<11s} ", end="")
        print()

    return df_results


def plot_passing_curves(target):
    """Plot curves showing passing molecules vs activity threshold.
    
    CRITICAL: This function always uses fresh data from passing_counts_{target}.csv
    (generated by generate_count_table). Old plot files are deleted before regeneration
    to prevent stale data from persisting.
    """
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    
    # CRITICAL: Delete old plot file to ensure fresh regeneration
    plot_file = OUTPUT_DIR / f"passing_curves_{target}.png"
    if plot_file.exists():
        plot_file.unlink()

    # Load data
    df = pd.read_csv(OUTPUT_DIR / f"passing_counts_{target}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(
        f"Molecules Passing Held-Out Filters - {target_name} (Top {TOP_N} Diverse)",
        fontsize=16,
        fontweight="bold",
    )

    # Get all methods
    if target == "ecoli":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
            "SciLeo_level1_iter3",
            "SciLeo_level2_iter1",
            "SciLeo_level2_iter2",
        ]
    elif target == "kpneumoniae":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
        ]
    else:
        scileo_methods = []

    methods = scileo_methods + ["REINVENT4", "NatureLM", "MolT5", "TextGrad"]

    # Plot 1: Absolute counts
    ax = axes[0]
    for method in methods:
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            ax.plot(
                method_data["activity_threshold"],
                method_data["passing"],
                marker="o",
                linewidth=2,
                label=method,
                color=METHOD_COLORS.get(method, "gray"),
            )

    ax.set_xlabel("Activity Threshold", fontweight="bold")
    ax.set_ylabel("Number of Passing Molecules", fontweight="bold")
    ax.set_title("Absolute Counts")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    # Plot 2: Pass rates
    ax = axes[1]
    for method in methods:
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            ax.plot(
                method_data["activity_threshold"],
                method_data["pass_rate"],
                marker="o",
                linewidth=2,
                label=method,
                color=METHOD_COLORS.get(method, "gray"),
            )

    ax.set_xlabel("Activity Threshold", fontweight="bold")
    ax.set_ylabel("Pass Rate (%)", fontweight="bold")
    ax.set_title("Percentage Passing")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"passing_curves_{target}.png", bbox_inches="tight")
    print(f"\n✓ Saved: passing_curves_{target}.png")
    plt.close()


def save_passing_molecules(target):
    """Save passing molecules for each method at each activity threshold.
    
    CRITICAL: Uses apply_filter_cascade() to ensure consistency with plot_filter_cascade.
    Old CSV files are deleted before regeneration to prevent stale molecules.
    """
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    print(f"\n{'='*70}")
    print(f"Saving Passing Molecules - {target_name} (Top {TOP_N} Diverse)")
    print(
        f"Note: Diversity already applied during top100 file creation (Tanimoto < 0.6)"
    )
    print("=" * 70)

    passing_dir = OUTPUT_DIR / "passing_molecules"
    passing_dir.mkdir(exist_ok=True)
    
    # CRITICAL: Delete old CSV files to prevent stale molecules
    for csv_file in passing_dir.glob(f"*_{target}_activity*.csv"):
        csv_file.unlink()
        print(f"  Deleted stale file: {csv_file.name}")

    # SciLeo methods (level1 for E. coli and KP)
    if target == "ecoli":
        scileo_files = [
            ("ecoli_level1_iter1_20251101150814_all_molecules.csv", "level1_iter1"),
            ("ecoli_level1_iter2_20251102002019_all_molecules.csv", "level1_iter2"),
            ("ecoli_level1_iter3_20251102201621_all_molecules.csv", "level1_iter3"),
        ]
    elif target == "kpneumoniae":
        scileo_files = [
            ("kp_level1_iter1_20251104162127_all_molecules.csv", "level1_iter1"),
            ("kp_level1_iter2_20251105131705_all_molecules.csv", "level1_iter2"),
        ]
    else:
        scileo_files = []

    # Process SciLeo methods
    for filename, method_suffix in scileo_files:
        data, _ = load_scileo_seed(filename, target)
        method_name = f"SciLeo_{method_suffix}"
        print(f"\n{method_name}:")

        for act_thresh in ACTIVITY_THRESHOLDS:
            # CRITICAL: Use shared function to ensure consistency with plot_filter_cascade
            filtered = apply_filter_cascade(data, act_thresh, apply_diversity=True)
            final_count = len(filtered)
            
            if final_count > 0:
                thresh_str = str(act_thresh).replace(".", "p")
                output_file = (
                    passing_dir
                    / f"scileo_{method_suffix}_{target}_activity{thresh_str}.csv"
                )
                filtered.to_csv(output_file, index=False)
                print(
                    f"  Activity ≥ {act_thresh}: {final_count} molecules → {output_file.name}"
                )
            else:
                print(f"  Activity ≥ {act_thresh}: 0 molecules")

    # Baselines
    for method in ["REINVENT4", "NatureLM", "MolT5", "TextGrad"]:
        print(f"\n{method}:")
        try:
            data = load_baseline_data(method, target, limit_reinvent=True)

            for act_thresh in ACTIVITY_THRESHOLDS:
                # CRITICAL: Use shared function to ensure consistency with plot_filter_cascade
                filtered = apply_filter_cascade(data, act_thresh, apply_diversity=True)
                final_count = len(filtered)
                
                if final_count > 0:
                    thresh_str = str(act_thresh).replace(".", "p")
                    output_file = (
                        passing_dir
                        / f"{method.lower()}_{target}_activity{thresh_str}.csv"
                    )
                    filtered.to_csv(output_file, index=False)
                    print(
                        f"  Activity ≥ {act_thresh}: {final_count} molecules → {output_file.name}"
                    )
                else:
                    print(f"  Activity ≥ {act_thresh}: 0 molecules")

        except Exception as e:
            raise RuntimeError(f"Failed to process {method}: {e}") from e


def plot_filter_cascade(target):
    """Plot cascade showing which filter removes most molecules for top 100.
    
    CRITICAL: This function always uses fresh data from load_scileo_seed/load_baseline_data,
    which ensures all filters are recomputed with the latest logic. Old plot files are
    deleted before regeneration to prevent stale data from persisting.
    """
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    
    # CRITICAL: Delete old plot file to ensure fresh regeneration
    plot_file = OUTPUT_DIR / f"filter_cascade_{target}.png"
    if plot_file.exists():
        plot_file.unlink()

    # Test at activity threshold = 0.05
    test_activity = 0.05

    # Get all methods
    if target == "ecoli":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
            "SciLeo_level1_iter3",
            "SciLeo_level2_iter1",
            "SciLeo_level2_iter2",
        ]
    elif target == "kpneumoniae":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
        ]
    else:
        scileo_methods = []

    methods = scileo_methods + ["REINVENT4", "NatureLM", "MolT5", "TextGrad"]

    n_methods = len(methods)
    n_cols = 3
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.suptitle(
        f"Filter Cascade (Activity ≥ {test_activity}) - {target_name} (Top {TOP_N} Diverse)",
        fontsize=16,
        fontweight="bold",
    )

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, method in enumerate(methods):
        ax = axes[idx]

        try:
            # Load data
            if method.startswith("SciLeo_"):
                method_suffix = method.replace("SciLeo_", "")
                if target == "ecoli":
                    # Map method suffix to filename
                    filename_map = {
                        "level1_iter1": "ecoli_level1_iter1_20251101150814_all_molecules.csv",
                        "level1_iter2": "ecoli_level1_iter2_20251102002019_all_molecules.csv",
                        "level1_iter3": "ecoli_level1_iter3_20251102201621_all_molecules.csv",
                        "level2_iter1": "ecoli_level2_iter1_20251101150813_all_molecules.csv",
                        "level2_iter2": "ecoli_level2_iter2_20251102002529_all_molecules.csv",
                    }
                    filename = filename_map.get(method_suffix)
                    if not filename:
                        raise ValueError(f"Unknown method suffix: {method_suffix}")
                elif target == "kpneumoniae":
                    # Map method suffix to filename for KP
                    filename_map = {
                        "level1_iter1": "kp_level1_iter1_20251104162127_all_molecules.csv",
                        "level1_iter2": "kp_level1_iter2_20251105131705_all_molecules.csv",
                    }
                    filename = filename_map.get(method_suffix)
                    if not filename:
                        raise ValueError(f"Unknown method suffix: {method_suffix}")
                else:
                    raise ValueError(f"No data for {method} with target {target}")
                data, _ = load_scileo_seed(filename, target)
            else:
                data = load_baseline_data(method, target, limit_reinvent=True)

            # Track counts through filter cascade
            filter_names = [f"Top {TOP_N}\nDiverse"]
            counts = [len(data)]

            filtered = data.copy()

            # Apply activity FIRST
            if "target_activity" in filtered.columns:
                filtered["target_activity"] = pd.to_numeric(
                    filtered["target_activity"], errors="coerce"
                )
                filtered = filtered[
                    filtered["target_activity"].notna()
                    & (filtered["target_activity"] >= test_activity)
                ]
                filter_names.append(f"Activity≥{test_activity}")
                counts.append(len(filtered))

            # Apply antibiotics_motifs_filter SECOND
            if (
                "antibiotics_motifs_filter" in filtered.columns
                and "antibiotics_motifs_filter" in FILTERS
            ):
                # Convert True/False strings to numeric (same as per_filter_pass_rates)
                col = filtered["antibiotics_motifs_filter"].copy()
                col = col.replace(
                    {
                        True: 1.0,
                        False: 0.0,
                        "True": 1.0,
                        "False": 0.0,
                        "1.0": 1.0,
                        "0.0": 0.0,
                        "1": 1.0,
                        "0": 0.0,
                    }
                )
                filtered["antibiotics_motifs_filter"] = pd.to_numeric(
                    col, errors="coerce"
                )
                filtered = filtered[
                    filtered["antibiotics_motifs_filter"].notna()
                    & (
                        filtered["antibiotics_motifs_filter"]
                        >= FILTERS["antibiotics_motifs_filter"]
                    )
                ]
                filter_names.append("antibiotics_motifs_filter")
                counts.append(len(filtered))

            # Apply remaining filters (excluding activity and antibiotics_motifs_filter)
            remaining_filters = {
                k: v
                for k, v in FILTERS.items()
                if k not in ["antibiotics_motifs_filter"]
            }

            for col, threshold in remaining_filters.items():
                if col in filtered.columns:
                    # Convert to numeric before comparison
                    filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
                    filtered = filtered[
                        filtered[col].notna() & (filtered[col] >= threshold)
                    ]
                    filter_names.append(col)
                    counts.append(len(filtered))

            # Remove duplicates
            if len(filtered) > 0:
                filtered = filtered.drop_duplicates(subset=["smiles"])
                filter_names.append("Dedup")
                counts.append(len(filtered))

            # Apply diversity filter (Tanimoto < 0.6)
            if len(filtered) > 0:
                from rdkit import Chem
                from rdkit.Chem import AllChem, DataStructs

                smiles_list = filtered["smiles"].tolist()
                fps = []
                valid_indices = []

                for idx_inner, smi in enumerate(smiles_list):
                    mol = Chem.MolFromSmiles(smi)
                    if mol is not None:
                        fp = AllChem.GetMorganFingerprintAsBitVect(
                            mol, radius=2, nBits=2048
                        )
                        fps.append(fp)
                        valid_indices.append(idx_inner)

                if len(fps) > 0:
                    selected_indices = [0]
                    selected_fps = [fps[0]]

                    for i in range(1, len(fps)):
                        max_sim = max(
                            DataStructs.TanimotoSimilarity(fps[i], sel_fp)
                            for sel_fp in selected_fps
                        )
                        if max_sim < 0.6:
                            selected_indices.append(i)
                            selected_fps.append(fps[i])

                    original_indices = [valid_indices[i] for i in selected_indices]
                    filtered = filtered.iloc[original_indices].copy()
                    filter_names.append("Diverse\n(Tan<0.6)")
                    counts.append(len(filtered))
                else:
                    filter_names.append("Diverse\n(Tan<0.6)")
                    counts.append(0)
            else:
                filter_names.append("Diverse\n(Tan<0.6)")
                counts.append(0)

            # CRITICAL: Validate consistency with shared function
            # This ensures plot_filter_cascade and save_passing_molecules always match
            validated_filtered = apply_filter_cascade(data, test_activity, apply_diversity=True)
            if len(filtered) != len(validated_filtered):
                raise RuntimeError(
                    f"INCONSISTENCY DETECTED for {method}: "
                    f"plot_filter_cascade has {len(filtered)} molecules, "
                    f"but apply_filter_cascade has {len(validated_filtered)} molecules. "
                    f"This should never happen - filter logic must match!"
                )

            # Plot
            x = np.arange(len(filter_names))
            bars = ax.bar(x, counts, color=METHOD_COLORS.get(method, "gray"), alpha=0.7)

            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2.0,
                        height,
                        f"{int(height)}",
                        ha="center",
                        va="bottom",
                        fontsize=8,
                    )

            ax.set_xlabel("Filter Stage", fontweight="bold")
            ax.set_ylabel("Molecules Remaining", fontweight="bold")
            ax.set_title(f"{method} (Final: {counts[-1]})")
            ax.set_xticks(x)
            ax.set_xticklabels(filter_names, rotation=45, ha="right", fontsize=9)
            ax.grid(axis="y", alpha=0.3)

        except Exception as e:
            raise RuntimeError(f"Failed to plot cascade for {method}: {e}") from e

    # Hide unused subplots
    for idx in range(n_methods, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"filter_cascade_{target}.png", bbox_inches="tight")
    print(f"✓ Saved: filter_cascade_{target}.png")
    plt.close()


def plot_round2_only(target):
    """Plot curves showing only Round 2 SciLeo methods.
    
    CRITICAL: This function always uses fresh data from passing_counts_{target}.csv
    (generated by generate_count_table). Old plot files are deleted before regeneration
    to prevent stale data from persisting.
    """
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    
    # CRITICAL: Delete old plot file to ensure fresh regeneration
    plot_file = OUTPUT_DIR / f"round2_only_passing_curves_{target}.png"
    if plot_file.exists():
        plot_file.unlink()

    # Load data
    df = pd.read_csv(OUTPUT_DIR / f"passing_counts_{target}.csv")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        f"Round 2 SciLeo - Enamine Library Incorporation - {target_name} (Top {TOP_N} Diverse)",
        fontsize=16,
        fontweight="bold",
    )

    # Only Round 2 methods
    if target == "ecoli":
        methods = [
            "SciLeo_round_2_0_pct_enamine",
            "SciLeo_round_2_50_pct_enamine",
            "SciLeo_round_2_80_pct_enamine",
            "SciLeo_round_2_100_pct_enamine",
        ]
    else:
        methods = []

    if not methods:
        plt.close()
        return

    # Plot 1: Absolute counts
    ax = axes[0]
    for method in methods:
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            # Create display name
            pct = method.split("_")[-2]
            display_name = f"{pct}% Enamine"
            ax.plot(
                method_data["activity_threshold"],
                method_data["passing"],
                marker="o",
                linewidth=2,
                label=display_name,
                color=METHOD_COLORS.get(method, "gray"),
            )

    ax.set_xlabel("Activity Threshold", fontweight="bold")
    ax.set_ylabel("Number of Passing Molecules", fontweight="bold")
    ax.set_title("Absolute Counts")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    # Plot 2: Pass rates
    ax = axes[1]
    for method in methods:
        method_data = df[df["method"] == method]
        if len(method_data) > 0:
            pct = method.split("_")[-2]
            display_name = f"{pct}% Enamine"
            ax.plot(
                method_data["activity_threshold"],
                method_data["pass_rate"],
                marker="o",
                linewidth=2,
                label=display_name,
                color=METHOD_COLORS.get(method, "gray"),
            )

    ax.set_xlabel("Activity Threshold", fontweight="bold")
    ax.set_ylabel("Pass Rate (%)", fontweight="bold")
    ax.set_title("Percentage Passing")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"round2_only_passing_curves_{target}.png", bbox_inches="tight"
    )
    print(f"\n✓ Saved: round2_only_passing_curves_{target}.png")
    plt.close()


def plot_per_filter_pass_rates(target):
    """Plot pass rates for each individual filter in a single file.
    
    CRITICAL: This function always uses fresh data from load_scileo_seed/load_baseline_data,
    which ensures all filters are recomputed with the latest logic. Old plot files are
    deleted before regeneration to prevent stale data from persisting.
    """
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    
    # CRITICAL: Delete old plot and CSV files to ensure fresh regeneration
    plot_file = OUTPUT_DIR / f"per_filter_pass_rates_{target}.png"
    csv_file = OUTPUT_DIR / f"per_filter_pass_rates_{target}.csv"
    if plot_file.exists():
        plot_file.unlink()
    if csv_file.exists():
        csv_file.unlink()

    # Get all methods
    if target == "ecoli":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
            "SciLeo_level1_iter3",
            "SciLeo_level2_iter1",
            "SciLeo_level2_iter2",
        ]
    elif target == "kpneumoniae":
        scileo_methods = [
            "SciLeo_level1_iter1",
            "SciLeo_level1_iter2",
        ]
    else:
        scileo_methods = []

    methods = scileo_methods + ["REINVENT4", "NatureLM", "MolT5", "TextGrad"]
    activity_threshold = 0.05

    # Collect data
    filter_names = ["activity"] + list(FILTERS.keys())
    all_results = []

    for method in methods:
        try:
            if method.startswith("SciLeo_"):
                method_suffix = method.replace("SciLeo_", "")
                if target == "ecoli":
                    filename_map = {
                        "level1_iter1": "ecoli_level1_iter1_20251101150814_all_molecules.csv",
                        "level1_iter2": "ecoli_level1_iter2_20251102002019_all_molecules.csv",
                        "level1_iter3": "ecoli_level1_iter3_20251102201621_all_molecules.csv",
                        "level2_iter1": "ecoli_level2_iter1_20251101150813_all_molecules.csv",
                        "level2_iter2": "ecoli_level2_iter2_20251102002529_all_molecules.csv",
                    }
                    filename = filename_map.get(method_suffix)
                    if not filename:
                        continue
                elif target == "kpneumoniae":
                    filename_map = {
                        "level1_iter1": "kp_level1_iter1_20251104162127_all_molecules.csv",
                        "level1_iter2": "kp_level1_iter2_20251105131705_all_molecules.csv",
                    }
                    filename = filename_map.get(method_suffix)
                    if not filename:
                        continue
                else:
                    continue
                data, _ = load_scileo_seed(filename, target)
            else:
                data = load_baseline_data(method, target, limit_reinvent=True)

            filter_results = count_per_filter(data, activity_threshold)
            for filter_name in filter_names:
                if filter_name in filter_results:
                    all_results.append(
                        {
                            "method": method,
                            "filter": filter_name,
                            "pass_rate": filter_results[filter_name]["pass_rate"],
                            "passing": filter_results[filter_name]["passing"],
                            "total": filter_results[filter_name]["total"],
                        }
                    )
        except Exception as e:
            raise RuntimeError(f"Failed to process {method} for {target}: {e}") from e

    if not all_results:
        return

    df_plot = pd.DataFrame(all_results)

    # Save CSV with per-filter pass rates
    csv_file = OUTPUT_DIR / f"per_filter_pass_rates_{target}.csv"
    df_plot.to_csv(csv_file, index=False)
    print(f"✓ Saved: per_filter_pass_rates_{target}.csv")

    # Create subplots
    n_filters = len(filter_names)
    n_cols = 3
    n_rows = (n_filters + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, filter_name in enumerate(filter_names):
        ax = axes[idx]
        filter_data = df_plot[df_plot["filter"] == filter_name]

        if len(filter_data) == 0:
            ax.axis("off")
            continue

        pass_rates = []
        method_labels = []
        method_colors = []

        for method in methods:
            method_row = filter_data[filter_data["method"] == method]
            if len(method_row) > 0:
                pass_rates.append(method_row["pass_rate"].values[0])
                method_labels.append(method)
                method_colors.append(METHOD_COLORS.get(method, "gray"))

        if pass_rates:
            x = np.arange(len(method_labels))
            bars = ax.bar(x, pass_rates, color=method_colors, alpha=0.8)

            for bar, rate in zip(bars, pass_rates):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{rate:.1f}%",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

            filter_display = filter_name.replace("_", " ").title()
            ax.set_title(filter_display, fontweight="bold")
            ax.set_xticks(x)
            ax.set_xticklabels(method_labels, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Pass Rate (%)", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
            ax.set_ylim(0, 105)
        else:
            ax.axis("off")

    # Hide unused subplots
    for idx in range(n_filters, len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"per_filter_pass_rates_{target}.png", bbox_inches="tight", dpi=300
    )
    print(f"✓ Saved: per_filter_pass_rates_{target}.png")
    plt.close()
    
    # CRITICAL: Always regenerate dependent plots to ensure they're in sync with fresh data
    # Delete old cascade plot to force regeneration with fresh data
    cascade_plot = OUTPUT_DIR / f"filter_cascade_{target}.png"
    if cascade_plot.exists():
        cascade_plot.unlink()
        print(f"  Deleted old {cascade_plot.name} to force regeneration with fresh data")
    
    # Regenerate cascade plot with fresh data
    plot_filter_cascade(target)


def plot_round2_filter_cascade(target):
    """Plot filter cascade for Round 2 SciLeo methods only.
    
    CRITICAL: This function always uses fresh data from load_scileo_seed,
    which ensures all filters are recomputed with the latest logic. Old plot files are
    deleted before regeneration to prevent stale data from persisting.
    """
    target_name = "E. coli" if target == "ecoli" else "K. pneumoniae"
    
    # CRITICAL: Delete old plot file to ensure fresh regeneration
    plot_file = OUTPUT_DIR / f"round2_only_filter_cascade_{target}.png"
    if plot_file.exists():
        plot_file.unlink()

    # Test at activity threshold = 0.05
    test_activity = 0.05

    # Only Round 2 methods
    if target == "ecoli":
        methods = [
            "SciLeo_round_2_0_pct_enamine",
            "SciLeo_round_2_50_pct_enamine",
            "SciLeo_round_2_80_pct_enamine",
            "SciLeo_round_2_100_pct_enamine",
        ]
    else:
        methods = []

    if not methods:
        return

    n_methods = len(methods)
    n_cols = 2
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 5 * n_rows))
    fig.suptitle(
        f"Round 2 Filter Cascade - Enamine Library (Activity ≥ {test_activity}) - {target_name}",
        fontsize=16,
        fontweight="bold",
    )

    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]

    for idx, method in enumerate(methods):
        ax = axes[idx]

        try:
            # Load data
            method_suffix = method.replace("SciLeo_", "")
            if target == "ecoli":
                filename_map = {
                    "round_2_0_pct_enamine": "escherichia_coli_level1_0_pct_enamine_20251023_8921_all_molecules.csv",
                    "round_2_50_pct_enamine": "escherichia_coli_level1_50_pct_enamine_20251023_d5ec_all_molecules.csv",
                    "round_2_80_pct_enamine": "escherichia_coli_level1_80_pct_enamine_20251023_8f0d_all_molecules.csv",
                    "round_2_100_pct_enamine": "escherichia_coli_level1_100_pct_enamine_20251023_ed3d_all_molecules.csv",
                }
                filename = filename_map.get(method_suffix)
            data, _ = load_scileo_seed(filename, target)

            # Track counts through filter cascade
            filter_names = [f"Top {TOP_N}\nDiverse"]
            counts = [len(data)]

            filtered = data.copy()

            # Apply activity FIRST
            if "target_activity" in filtered.columns:
                filtered["target_activity"] = pd.to_numeric(
                    filtered["target_activity"], errors="coerce"
                )
                filtered = filtered[
                    filtered["target_activity"].notna()
                    & (filtered["target_activity"] >= test_activity)
                ]
                filter_names.append(f"Activity≥{test_activity}")
                counts.append(len(filtered))

            # Apply antibiotics_motifs_filter SECOND
            if (
                "antibiotics_motifs_filter" in filtered.columns
                and "antibiotics_motifs_filter" in FILTERS
            ):
                # Convert True/False strings to numeric (same as per_filter_pass_rates)
                col = filtered["antibiotics_motifs_filter"].copy()
                col = col.replace(
                    {
                        True: 1.0,
                        False: 0.0,
                        "True": 1.0,
                        "False": 0.0,
                        "1.0": 1.0,
                        "0.0": 0.0,
                        "1": 1.0,
                        "0": 0.0,
                    }
                )
                filtered["antibiotics_motifs_filter"] = pd.to_numeric(
                    col, errors="coerce"
                )
                filtered = filtered[
                    filtered["antibiotics_motifs_filter"].notna()
                    & (
                        filtered["antibiotics_motifs_filter"]
                        >= FILTERS["antibiotics_motifs_filter"]
                    )
                ]
                filter_names.append("antibiotics_motifs_filter")
                counts.append(len(filtered))

            # Apply remaining filters
            remaining_filters = {
                k: v
                for k, v in FILTERS.items()
                if k not in ["antibiotics_motifs_filter"]
            }

            for col, threshold in remaining_filters.items():
                if col in filtered.columns:
                    # Convert to numeric before comparison
                    filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
                    filtered = filtered[
                        filtered[col].notna() & (filtered[col] >= threshold)
                    ]
                    filter_names.append(col)
                    counts.append(len(filtered))

            # Remove duplicates
            if len(filtered) > 0:
                filtered = filtered.drop_duplicates(subset=["smiles"])
                filter_names.append("Dedup")
                counts.append(len(filtered))

            # Apply diversity filter
            if len(filtered) > 0:
                filtered = apply_diversity_and_dedup(filtered, threshold=0.6)
                filter_names.append("Diversity\n(Tanimoto<0.6)")
                counts.append(len(filtered))

            # Plot waterfall
            colors = [
                "#2E86AB"
                if i == 0
                else "#E63946"
                if counts[i] < counts[i - 1]
                else "#52B788"
                for i in range(len(counts))
            ]
            colors[0] = "#2E86AB"
            if len(counts) > 0:
                colors[-1] = "#06D6A0"

            bars = ax.bar(
                range(len(counts)), counts, color=colors, edgecolor="black", linewidth=1
            )

            # Add count labels
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height,
                    f"{int(count)}",
                    ha="center",
                    va="bottom",
                    fontweight="bold",
                    fontsize=9,
                )

            # Display name
            pct = method_suffix.split("_")[-2]
            ax.set_title(f"{pct}% Enamine Library", fontweight="bold", fontsize=12)
            ax.set_xticks(range(len(filter_names)))
            ax.set_xticklabels(filter_names, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel("Number of Molecules", fontweight="bold")
            ax.set_ylim(0, max(counts) * 1.15)
            ax.grid(axis="y", alpha=0.3)

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error loading data:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            pct = method.split("_")[-2] if "_" in method else method
            ax.set_title(f"{pct}% Enamine", fontweight="bold")

    # Hide empty subplots
    for idx in range(len(methods), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(
        OUTPUT_DIR / f"round2_only_filter_cascade_{target}.png", bbox_inches="tight"
    )
    print(f"✓ Saved: round2_only_filter_cascade_{target}.png")
    plt.close()


def main():
    """Main evaluation function."""
    print("=" * 70)
    print(
        f"Type 2 Evaluation: Held-Out Filter Analysis (with LLM Mutation, Top {TOP_N} Diverse)"
    )
    print("=" * 70)
    print(f"\nUsing top {TOP_N} DIVERSE molecules (Tanimoto < 0.6)")
    print(
        f"Selection: greedy by aggregated score (GEOMETRIC MEAN of novelty, activity, toxicity)"
    )
    print(f"\nActivity thresholds: {ACTIVITY_THRESHOLDS}")
    print(f"Held-out filters:")
    for k, v in FILTERS.items():
        print(f"  {k}: >= {v}")
    print(
        f"\nREINVENT4 limited to first 10k molecules before top-{TOP_N} diverse selection"
    )
    print(f"SciLeo seeds and LLM mutation runs shown separately")
    print(f"\nInput directory: {DATA_DIR.absolute()}")
    print(f"Output directory: {OUTPUT_DIR.absolute()}")

    # Process both targets
    for target in ["ecoli", "kpneumoniae"]:
        # CRITICAL: Generate fresh data first - this ensures all filters are recomputed
        # with the latest logic before any plots are generated
        df_results = generate_count_table(target)
        save_passing_molecules(target)
        
        # CRITICAL: Regenerate ALL plots with fresh data - order matters:
        # 1. per_filter_pass_rates regenerates filter_cascade (dependency)
        # 2. Other plots are independent
        plot_per_filter_pass_rates(target)  # This also regenerates filter_cascade
        plot_passing_curves(target)  # Uses passing_counts_{target}.csv from generate_count_table
        plot_round2_only(target)  # Uses passing_counts_{target}.csv from generate_count_table
        plot_round2_filter_cascade(target)  # Uses fresh data from load_scileo_seed

    # Update combined passing molecules file for E. coli
    print("\n" + "=" * 70)
    print("Updating Combined SciLeo Passing Molecules")
    print("=" * 70)
    try:
        from update_combined_passing import main as update_combined
        update_combined()
    except Exception as e:
        raise RuntimeError(f"Failed to update combined passing molecules: {e}") from e

    print("\n" + "=" * 70)
    print("✅ Type 2 Evaluation Complete!")
    print(f"Results saved to: {OUTPUT_DIR.absolute()}")
    print(f"Passing molecules saved to: {OUTPUT_DIR / 'passing_molecules'}")
    print("=" * 70)


if __name__ == "__main__":
    main()
