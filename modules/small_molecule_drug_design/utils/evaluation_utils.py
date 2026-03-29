"""
Utilities for evaluating SMILES lists across registered objectives/scorers.

This module exposes a single entrypoint `evaluate_smiles_across_objectives` that:
- optionally ensures scorer modules are imported (to register scorers)
- builds `Candidate` objects from input SMILES (optionally keeping the longest fragment)
- evaluates any number of objectives (scorer names) for the given SMILES list
- returns a `(DataFrame, summary_dict)` for downstream analysis

Notes:
- This function logs exceptions with full tracebacks and continues, filling None
  where evaluation failed, to avoid silent failures.
"""

from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import traceback
import pandas as pd

from scileo_agent.core.registry import get_scorer
from scileo_agent.core.data_models import Candidate
from scileo_agent.utils.logging import get_logger


def _keep_longest_fragment(smiles: str) -> str:
    """Keep the longest dot-delimited fragment if present; otherwise return input."""
    if not isinstance(smiles, str):
        return smiles
    if "." not in smiles:
        return smiles
    parts = smiles.split(".")
    lengths = [len(p) for p in parts]
    return parts[lengths.index(max(lengths))]


def _coerce_objectives(
    objectives: Union[Sequence[str], Mapping[str, str]],
    available_targets: Optional[Mapping[str, str]] = None,
) -> Dict[str, str]:
    """
    Normalize objectives to a mapping of {label -> scorer_name}.

    - If a mapping is provided, assume values are scorer registry names
    - If a list is provided, try to map each item via `available_targets` if present,
      otherwise treat the item itself as the scorer registry name. The label is the
      input string.
    """
    if isinstance(objectives, Mapping):
        return dict(objectives)

    resolved: Dict[str, str] = {}
    for item in objectives:
        if available_targets and item in available_targets:
            resolved[item] = available_targets[item]
        else:
            resolved[item] = item
    return resolved


def evaluate_smiles_across_objectives(
    smiles_list: Sequence[str],
    objectives: Union[Sequence[str], Mapping[str, str]],
    *,
    available_targets: Optional[Mapping[str, str]] = None,
    required_modules: Optional[Iterable[str]] = None,
    optional_modules: Optional[Iterable[str]] = None,
    exclude_if_contains: Optional[str] = None,
    keep_longest_fragment: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, Optional[float]]]]:
    """
    Evaluate a list of SMILES across a set of objectives (scorer registry names).

    Args:
        smiles_list: Input SMILES strings.
        objectives: Either a list of objective identifiers or a mapping of
            {label -> scorer_registry_name}. If a list is provided and
            `available_targets` is given, items are resolved via that mapping; else
            items are treated as scorer registry names directly.
        available_targets: Optional mapping for resolving objective keys to scorer
            registry names (e.g., `run_optimization.AVAILABLE_TARGETS`).
        required_modules: Import strings that must be imported (errors raise).
        optional_modules: Import strings that are best-effort (errors logged).
        exclude_if_contains: If provided, skip any scorer whose name contains this
            substring (e.g., "dock").
        keep_longest_fragment: If True, when a SMILES has dot-separated fragments,
            keep only the longest fragment for evaluation.

    Returns:
        (df, summary) where:
          - df: pandas DataFrame with columns: "smiles" plus one column per scorer
          - summary: {scorer_name -> {n, mean, min, max, n_none}}
    """
    logger = get_logger()

    # Ensure scorer modules are imported so their scorers are registered
    if required_modules:
        for mod in required_modules:
            try:
                __import__(mod)
            except Exception:
                logger.error(f"Error importing required scorer module: {mod}")
                traceback.print_exc()
                raise

    if optional_modules:
        for mod in optional_modules:
            try:
                __import__(mod)
            except Exception as e:
                logger.warning(f"Optional scorer module not available: {mod}: {e}")
                traceback.print_exc()

    # Prepare candidates
    processed_smiles: List[str] = []
    if keep_longest_fragment:
        processed_smiles = [_keep_longest_fragment(s) for s in smiles_list]
    else:
        processed_smiles = list(smiles_list)

    candidates = [Candidate(representation=s) for s in processed_smiles]

    # Resolve objectives -> scorer names
    target_map = _coerce_objectives(objectives, available_targets)

    # Evaluate
    results: Dict[str, List[Optional[float]]] = {"smiles": list(smiles_list)}

    for label, scorer_name in target_map.items():
        if exclude_if_contains and exclude_if_contains in scorer_name:
            continue

        try:
            scorer = get_scorer(scorer_name)
            if scorer is None:
                logger.warning(f"Scorer not found/registered: {scorer_name}")
                results[scorer_name] = [None] * len(candidates)
                continue

            scores = scorer(candidates)

            # Handle population-wise scorers that may return a single Optional[float]
            population_wise = False
            try:
                population_wise = bool(getattr(scorer, "_scorer_metadata", {}).get("population_wise", False))
            except Exception:
                population_wise = False

            if population_wise and not isinstance(scores, list):
                # replicate a single population score across all candidates
                results[scorer_name] = [scores] * len(candidates)
            else:
                if not isinstance(scores, list):
                    raise ValueError(
                        f"Scorer '{scorer_name}' returned non-list scores for per-candidate evaluation"
                    )
                if len(scores) != len(candidates):
                    raise ValueError(
                        f"Scorer '{scorer_name}' returned {len(scores)} scores for {len(candidates)} candidates"
                    )
                # Ensure Optional[float]
                results[scorer_name] = [None if s is None else float(s) for s in scores]

        except Exception as exc:
            logger.error(f"Error while scoring with {scorer_name}: {exc}")
            traceback.print_exc()
            results[scorer_name] = [None] * len(candidates)

    # Build DataFrame
    df = pd.DataFrame(results)

    # Summary statistics per scorer column
    summary: Dict[str, Dict[str, Optional[float]]] = {}
    for col in df.columns:
        if col == "smiles":
            continue
        vals = pd.to_numeric(df[col], errors="coerce")
        if vals.notna().any():
            summary[col] = {
                "n": int(vals.notna().sum()),
                "mean": float(vals.mean(skipna=True)),
                "min": float(vals.min(skipna=True)),
                "max": float(vals.max(skipna=True)),
                "n_none": int(vals.isna().sum()),
            }
        else:
            summary[col] = {
                "n": 0,
                "mean": None,
                "min": None,
                "max": None,
                "n_none": int(len(vals)),
            }

    return df, summary


__all__ = [
    "evaluate_smiles_across_objectives",
]


