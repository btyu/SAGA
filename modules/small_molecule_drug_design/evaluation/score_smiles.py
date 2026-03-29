#!/usr/bin/env python3
"""
Score a list of SMILES with specified objectives and output results to CSV.

This script takes a list of SMILES strings and objective names, scores each molecule
using the framework's scoring functions, and outputs a CSV with all scores and filtering
information (PAINS, motif filters, etc.).

Usage:
    python -m modules.small_molecule_drug_design.evaluation.score_smiles \
        --smiles "CCO" "c1ccccc1" \
        --objectives qed logp mw \
        --output results.csv

    python -m modules.small_molecule_drug_design.evaluation.score_smiles \
        --smiles-file input.txt \
        --objectives klebsiella_pneumoniae toxicity pains brenk antibiotics_novelty deepdl mw \
        --output scored_results.csv

Example:
    python -m modules.small_molecule_drug_design.evaluation.score_smiles \
        --smiles-file test_molecules.txt \
        --objectives escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
        --output scored_molecules.csv
"""

import argparse
import logging
import sys
import os
import warnings
from typing import List, Dict, Optional
import pandas as pd

# Add parent directories to path for imports
module_dir = os.path.dirname(os.path.abspath(__file__))
smdd_dir = os.path.dirname(module_dir)  # small_molecule_drug_design/
modules_dir = os.path.dirname(smdd_dir)  # modules/
base_dir = os.path.dirname(modules_dir)  # SciLeoAgent/
sys.path.insert(0, base_dir)
sys.path.insert(0, os.path.join(base_dir, "scileo_agent"))

from scileo_agent.core.data_models import Candidate
from scileo_agent.core.registry import get_scorer

# Import modules to trigger scorer registration
import modules.small_molecule_drug_design.scorer.antibiotics_scorer  # noqa: F401

# import modules.small_molecule_drug_design.scorer.minimol_scorer  # noqa: F401

# # Skip unidock_scorer to avoid openbabel issues when not using docking targets
# # import modules.small_molecule_drug_design.scorer.unidock_scorer  # noqa: F401
import modules.small_molecule_drug_design.scorer.druglikeness_scorer  # noqa: F401

import modules.small_molecule_drug_design.scorer.chemprop_scorer  # noqa: F401

from modules.small_molecule_drug_design.utils.rdkit_utils import (
    filter_smiles_preserves_existing_hits,
)


def configure_logging():
    """Configure logging to reduce noisy INFO messages and suppress warnings."""
    logging.getLogger().setLevel(logging.WARNING)

    noisy_loggers = [
        "LiteLLM",
        "litellm",
        "openai",
        "httpx",
        "httpcore",
        "MDAnalysis",
        "MDAnalysis.coordinates",
        "MDAnalysis.topology",
        "MDAnalysis.universe",
        "MDAnalysis.topology.base",
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Suppress RDKit warnings
    from rdkit import RDLogger

    RDLogger.DisableLog("rdApp.*")

    # Suppress specific MDAnalysis warnings
    warnings.filterwarnings("ignore", message="Unit cell dimensions not found.*")
    warnings.filterwarnings("ignore", message="Found no information for attr.*")
    warnings.filterwarnings("ignore", message=".*CRYST1 record.*")


# Available scoring functions (from run_optimization.py)
AVAILABLE_OBJECTIVES = {
    "ampcclean": "ampcclean_unidock",
    "muopioidclean": "muopioidclean_unidock",
    "mpro": "mpro_unidock",
    "mars1": "mars1_unidock",
    "qed": "qed",
    "sa": "sa_score",
    "logp": "logp_score",
    "mw": "mw_score",
    "deepdl": "deepdl_druglikeness",
    "toxicity": "toxicity_safety_chemprop",
    "staph_aureus": "staph_aureus_chemprop",
    "pains": "pains_filter",
    "brenk": "brenk_filter",
    "ra": "ra_score_xgb",
    "antibiotics_novelty": "antibiotics_novelty",
    "antibiotics_motifs_filter": "antibiotics_motifs_filter",
    "mpro_his161_a": "mpro_his161_a",
    "mpro_glu164_a": "mpro_glu164_a",
    "mpro_his39_a": "mpro_his39_a",
    "acinetobacter_baumanii": "acinetobacter_baumanii_minimol",
    "escherichia_coli": "escherichia_coli_minimol",
    "klebsiella_pneumoniae": "klebsiella_pneumoniae_minimol",
    "pseudomonas_aeruginosa": "pseudomonas_aeruginosa_minimol",
    "neisseria_gonorrhoeae": "neisseria_gonorrhoeae_minimol",
}


def load_smiles_from_file(file_path: str) -> List[str]:
    """Load SMILES from a text file (one per line) or CSV file.

    Args:
        file_path: Path to file containing SMILES

    Returns:
        List of SMILES strings
    """
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        # Try to find SMILES column
        smiles_col = None
        for col in ["smiles", "SMILES", "Smiles", "smile"]:
            if col in df.columns:
                smiles_col = col
                break
        if smiles_col is None:
            raise ValueError(
                f"Could not find SMILES column in CSV. Available columns: {df.columns.tolist()}"
            )
        return df[smiles_col].dropna().astype(str).tolist()
    else:
        # Text file, one SMILES per line
        with open(file_path, "r") as f:
            return [line.strip() for line in f if line.strip()]


def score_smiles(
    smiles_list: List[str], objective_names: List[str], batch_size: int = 100
) -> pd.DataFrame:
    """Score a list of SMILES with specified objectives in batches.

    Args:
        smiles_list: List of SMILES strings to score
        objective_names: List of objective names (keys from AVAILABLE_OBJECTIVES)
        batch_size: Number of SMILES to process per batch (default: 100)

    Returns:
        DataFrame with SMILES and scores for each objective
    """
    print(
        f"[INFO] Scoring {len(smiles_list)} SMILES with {len(objective_names)} objectives (batch_size={batch_size})"
    )

    # Initialize results dictionary
    results = {
        "smiles": smiles_list,
    }

    # Calculate number of batches
    num_batches = (len(smiles_list) + batch_size - 1) // batch_size

    # Score each objective
    for obj_name in objective_names:
        if obj_name not in AVAILABLE_OBJECTIVES:
            print(
                f"[WARN] Unknown objective '{obj_name}', skipping. Available: {list(AVAILABLE_OBJECTIVES.keys())}"
            )
            continue

        scorer_name = AVAILABLE_OBJECTIVES[obj_name]
        print(f"[INFO] Scoring objective '{obj_name}' using scorer '{scorer_name}'")

        try:

            scorer = get_scorer(scorer_name)

            # Check if population-wise scorer
            is_population_wise = scorer._scorer_metadata.get("population_wise", False)

            if is_population_wise:
                # Population-wise scorers must use all candidates at once
                candidates = [Candidate(representation=smi) for smi in smiles_list]
                pop_score = scorer(candidates)
                results[obj_name] = [pop_score] * len(candidates)
                print(f"[INFO] Population-wise score for '{obj_name}': {pop_score}")
            else:
                # Regular scorer - process in batches
                all_scores = []

                for batch_idx in range(num_batches):
                    start_idx = batch_idx * batch_size
                    end_idx = min(start_idx + batch_size, len(smiles_list))
                    batch_smiles = smiles_list[start_idx:end_idx]

                    # Create candidates for this batch
                    batch_candidates = [
                        Candidate(representation=smi) for smi in batch_smiles
                    ]

                    # Score batch
                    batch_scores = scorer(batch_candidates)
                    all_scores.extend(batch_scores)

                    print(
                        f"[INFO] Processed batch {batch_idx + 1}/{num_batches} for '{obj_name}' "
                        f"({start_idx}-{end_idx}, {end_idx - start_idx} molecules)"
                    )

                results[obj_name] = all_scores
                valid_scores = [s for s in all_scores if s is not None]
                if valid_scores:
                    print(
                        f"[INFO] Scored '{obj_name}': {len(valid_scores)}/{len(all_scores)} valid, "
                        f"mean={sum(valid_scores)/len(valid_scores):.3f}, "
                        f"min={min(valid_scores):.3f}, max={max(valid_scores):.3f}"
                    )
                else:
                    print(f"[WARN] No valid scores for '{obj_name}'")

        except Exception as e:
            print(f"[ERROR] Failed to score '{obj_name}': {e}")
            import traceback

            traceback.print_exc()
            results[obj_name] = [None] * len(smiles_list)

    return pd.DataFrame(results)


def add_filtering_info(df: pd.DataFrame) -> pd.DataFrame:
    """Add filtering information (PAINS, motif filters) to the dataframe.

    Args:
        df: DataFrame with 'smiles' column

    Returns:
        DataFrame with added columns: 'motif_filtered' and 'filter_reason'
    """
    print("[INFO] Adding filtering information (PAINS, motif filters)")

    smiles_list = df["smiles"].tolist()

    try:
        kept_smiles, dropped_reasons = filter_smiles_preserves_existing_hits(
            smiles_list
        )
        kept_set = set(kept_smiles)

        # Add columns
        df["motif_filtered"] = ~df["smiles"].isin(kept_set)
        df["filter_reason"] = df["smiles"].map(dropped_reasons).fillna("")

        num_filtered = df["motif_filtered"].sum()
        print(
            f"[INFO] {num_filtered}/{len(df)} molecules would be filtered by motif/PAINS filters"
        )

        # Print summary of filter reasons
        if num_filtered > 0:
            filtered_df = df[df["motif_filtered"]]
            reason_counts = filtered_df["filter_reason"].value_counts()
            print(f"[INFO] Filter reason distribution:")
            for reason, count in reason_counts.head(10).items():
                print(f"  - {reason}: {count}")

    except Exception as e:
        print(f"[ERROR] Failed to add filtering info: {e}")
        import traceback

        traceback.print_exc()
        df["motif_filtered"] = False
        df["filter_reason"] = ""

    return df


def main():
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Score SMILES with specified objectives and output to CSV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available objectives:
  {', '.join(sorted(AVAILABLE_OBJECTIVES.keys()))}

Examples:
  # Score from command-line SMILES
  python -m modules.small_molecule_drug_design.evaluation.score_smiles \\
      --smiles "CCO" "c1ccccc1" "CC(C)O" \\
      --objectives qed logp mw \\
      --output results.csv

  # Score from file
  python -m modules.small_molecule_drug_design.evaluation.score_smiles \\
      --smiles-file molecules.txt \\
      --objectives klebsiella_pneumoniae toxicity pains brenk antibiotics_novelty deepdl mw \\
      --output scored_molecules.csv

  # Score antibiotics optimization targets
  python -m modules.small_molecule_drug_design.evaluation.score_smiles \\
      --smiles-file candidates.csv \\
      --objectives escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \\
      --output escherichia_coli_scored.csv

  # Score large file with smaller batch size to save memory
  python -m modules.small_molecule_drug_design.evaluation.score_smiles \\
      --smiles-file large_library.csv \\
      --objectives klebsiella_pneumoniae toxicity \\
      --batch-size 50 \\
      --output scored_large.csv
        """,
    )

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--smiles", nargs="+", help="SMILES strings to score")
    input_group.add_argument(
        "--smiles-file",
        type=str,
        help="File containing SMILES (one per line, or CSV with smiles column)",
    )

    parser.add_argument(
        "--objectives",
        nargs="+",
        required=True,
        help="Objective names to score (space-separated)",
    )

    parser.add_argument(
        "--output", type=str, required=True, help="Output CSV file path"
    )

    parser.add_argument(
        "--no-filter-info",
        action="store_true",
        help="Skip adding filtering information (motif_filtered, filter_reason columns)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Number of SMILES to process per batch (default: 100). Smaller batches use less memory.",
    )

    args = parser.parse_args()

    # Load SMILES
    if args.smiles:
        smiles_list = args.smiles
    else:
        smiles_list = load_smiles_from_file(args.smiles_file)

    if not smiles_list:
        print("[ERROR] No SMILES provided")
        sys.exit(1)

    print(f"[INFO] Loaded {len(smiles_list)} SMILES")

    # Score SMILES
    df = score_smiles(smiles_list, args.objectives, batch_size=args.batch_size)

    # Add filtering information unless disabled
    if not args.no_filter_info:
        df = add_filtering_info(df)

    # Save to CSV
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"[INFO] Results saved to {args.output}")

    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Total molecules: {len(df)}")
    if not args.no_filter_info:
        print(
            f"Filtered by motif/PAINS: {df['motif_filtered'].sum()} ({df['motif_filtered'].sum()/len(df)*100:.1f}%)"
        )

    for obj_name in args.objectives:
        if obj_name in df.columns:
            scores = df[obj_name].dropna()
            if len(scores) > 0:
                print(f"\n{obj_name}:")
                print(
                    f"  Valid scores: {len(scores)}/{len(df)} ({len(scores)/len(df)*100:.1f}%)"
                )
                print(f"  Mean: {scores.mean():.4f}")
                print(f"  Std: {scores.std():.4f}")
                print(f"  Min: {scores.min():.4f}")
                print(f"  Max: {scores.max():.4f}")
                print(f"  Median: {scores.median():.4f}")

    print("=" * 80)


if __name__ == "__main__":
    main()
