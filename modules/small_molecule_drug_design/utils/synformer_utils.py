"""
Synformer utilities for generating molecular analogs.

This module provides utilities for using Synformer to generate synthesizable
analogs of input molecules.
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Union

# Add synformer directory to path if not already there
synformer_dir = Path(__file__).parent.parent / "synformer"
if str(synformer_dir) not in sys.path:
    sys.path.insert(0, str(synformer_dir))

from synformer.chem.mol import Molecule
from synformer.sampler.analog.parallel import run_parallel_sampling_return_smiles


def generate_analogs(
    smiles_list: List[str],
    model_path: Optional[str] = None,
    search_width: int = 24,
    exhaustiveness: int = 64,
    num_gpus: int = -1,
    num_workers_per_gpu: int = 2,
    time_limit: int = 180,
    max_results_per_molecule: Optional[int] = None,
    sort_by_scores: bool = True,
    return_mapping: bool = False,
) -> Union[List[str], Dict[str, str]]:
    """
    Generate synthesizable analogs for a list of input SMILES.

    Args:
        smiles_list: List of input SMILES strings to generate analogs for
        model_path: Path to Synformer model checkpoint. If None, uses default
            location: synformer/data/trained_weights/sf_ed_default.ckpt
        search_width: Search width parameter (default: 24)
        exhaustiveness: Exhaustiveness parameter (default: 64)
        num_gpus: Number of GPUs to use (-1 for auto-detect, default: -1)
        num_workers_per_gpu: Number of workers per GPU (default: 1)
        time_limit: Time limit per molecule in seconds (default: 180)
        max_results_per_molecule: Maximum number of analogs to return per input
            molecule. If None, returns all generated analogs (default: None)
        sort_by_scores: Whether to sort results by similarity score (default: True)
        return_mapping: If True, returns a dict mapping original SMILES to analog SMILES.
            If False, returns a list of analog SMILES (default: False)

    Returns:
        If return_mapping is False: List of analog SMILES strings. Results are deduplicated
        and optionally limited per input molecule.
        If return_mapping is True: Dict mapping original SMILES to analog SMILES.

    Example:
        >>> smiles = ["CC(C)CC1=CC=C(C=C1)C(C)C(=O)O", "CC1=CC=C(C=C1)O"]
        >>> analogs = generate_analogs(smiles)
        >>> print(f"Generated {len(analogs)} analogs")
    """
    if not smiles_list:
        return []

    # Convert SMILES to Molecule objects
    input_molecules = []
    for smiles in smiles_list:
        try:
            mol = Molecule(smiles)
            input_molecules.append(mol)
        except Exception as e:
            print(f"Warning: Failed to parse SMILES '{smiles}': {e}")
            continue

    if not input_molecules:
        return []

    # Set default model path if not provided
    if model_path is None:
        # Try to find synformer directory relative to this file
        synformer_dir = Path(__file__).parent.parent / "synformer"
        model_path = synformer_dir / "data" / "trained_weights" / "sf_ed_default.ckpt"
        if not model_path.exists():
            # Fallback: try relative to current working directory
            model_path = Path("synformer/data/trained_weights/sf_ed_default.ckpt")
            if not model_path.exists():
                raise FileNotFoundError(
                    f"Model not found at default location. Please specify model_path. "
                    f"Tried: {synformer_dir / 'data' / 'trained_weights' / 'sf_ed_default.ckpt'} "
                    f"and {Path('synformer/data/trained_weights/sf_ed_default.ckpt')}"
                )

    model_path = Path(model_path)

    # Run parallel sampling
    result_df = run_parallel_sampling_return_smiles(
        input=input_molecules,
        model_path=model_path,
        search_width=search_width,
        exhaustiveness=exhaustiveness,
        num_gpus=num_gpus,
        num_workers_per_gpu=num_workers_per_gpu,
        task_qsize=0,
        result_qsize=0,
        time_limit=time_limit,
        sort_by_scores=sort_by_scores,
    )

    if result_df.empty:
        return {} if return_mapping else []

    # Deduplicate by target (input molecule) and keep best result per target
    result_df = result_df.drop_duplicates(subset="target", keep="first")

    # If max_results_per_molecule is specified, limit results per input molecule
    if max_results_per_molecule is not None:
        # Group by target and take top N per target
        result_df = (
            result_df.groupby("target")
            .head(max_results_per_molecule)
            .reset_index(drop=True)
        )

    # Extract and return SMILES list or mapping
    if return_mapping:
        # Return mapping: original SMILES -> analog SMILES
        mapping = dict(zip(result_df["target"], result_df["smiles"]))
        return mapping
    else:
        analog_smiles = result_df["smiles"].tolist()
        return analog_smiles
