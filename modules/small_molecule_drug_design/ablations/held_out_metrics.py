#!/usr/bin/env python3
"""Reusable helpers for computing and summarizing held-out molecular metrics."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, QED

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for minimal installs
    def tqdm(iterable, desc=None, total=None):
        if desc:
            print(desc)
        return iterable


# Ensure project root is importable so scorer modules resolve correctly.
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


HELD_OUT_FILTERS: Dict[str, float] = {
    "qed": 0.5,
    "sa": 0.5,
    "mw": 0.5,
    "pains": 1.0,
    "brenk": 1.0,
    "deepdl": 0.3,
    "antibiotics_novelty": 0.6,
    "antibiotics_motifs_filter": 1.0,
    "toxicity": 0.5,
    "ring_score": 1.0,
}


_PAINS_CATALOG: Optional[FilterCatalog.FilterCatalog] = None
_BRENK_CATALOG: Optional[FilterCatalog.FilterCatalog] = None
_DEEPDL_MODEL = None
_DEEPDL_ERROR: Optional[str] = None


def _init_filters() -> None:
    """Initialize PAINS and BRENK catalogs."""
    global _PAINS_CATALOG, _BRENK_CATALOG
    if _PAINS_CATALOG is None:
        pains_params = FilterCatalog.FilterCatalogParams()
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_A)
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_B)
        pains_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS_C)
        _PAINS_CATALOG = FilterCatalog.FilterCatalog(pains_params)

    if _BRENK_CATALOG is None:
        brenk_params = FilterCatalog.FilterCatalogParams()
        brenk_params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
        _BRENK_CATALOG = FilterCatalog.FilterCatalog(brenk_params)


def _compute_qed(smiles_list: Sequence[str]) -> List[float]:
    scores: List[float] = []
    for smi in tqdm(smiles_list, desc="Computing QED"):
        mol = Chem.MolFromSmiles(smi)
        scores.append(QED.qed(mol) if mol is not None else np.nan)
    return scores


def _compute_sa(smiles_list: Sequence[str]) -> List[float]:
    from rdkit.Chem import RDConfig  # type: ignore[attr-defined]

    sys.path.append(os.path.join(RDConfig.RDContribDir, "SA_Score"))
    try:
        import sascorer

        scores: List[float] = []
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


def _compute_mw(smiles_list: Sequence[str]) -> List[float]:
    scores: List[float] = []
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


def _compute_pains(smiles_list: Sequence[str]) -> List[float]:
    _init_filters()
    scores: List[float] = []
    for smi in tqdm(smiles_list, desc="Computing PAINS"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            has_pains = _PAINS_CATALOG.HasMatch(mol)  # type: ignore[union-attr]
            scores.append(0.0 if has_pains else 1.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_brenk(smiles_list: Sequence[str]) -> List[float]:
    _init_filters()
    scores: List[float] = []
    for smi in tqdm(smiles_list, desc="Computing BRENK"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            has_brenk = _BRENK_CATALOG.HasMatch(mol)  # type: ignore[union-attr]
            scores.append(0.0 if has_brenk else 1.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_deepdl(smiles_list: Sequence[str]) -> List[float]:
    """Compute DeepDL scores directly (0-1 range, normalized from 0-100)."""
    global _DEEPDL_MODEL, _DEEPDL_ERROR

    if _DEEPDL_MODEL is None and _DEEPDL_ERROR is None:
        try:
            from druglikeness.deepdl import DeepDL

            _DEEPDL_MODEL = DeepDL.from_pretrained("extended", device="cpu")
        except ImportError as e:
            _DEEPDL_ERROR = str(e)
            print(f"  ERROR: Could not import druglikeness.deepdl: {e}")
            print(f"  ERROR: DeepDL computation will fail. Install the druglikeness package.")
            print("  ERROR: Run: pip install druglikeness")
            raise
        except Exception as e:
            _DEEPDL_ERROR = str(e)
            print(f"  ERROR: Could not initialize DeepDL model: {e}")
            raise

    if _DEEPDL_MODEL is None:
        error_msg = f"DeepDL model not available: {_DEEPDL_ERROR}"
        print(f"  ERROR: {error_msg}")
        raise RuntimeError(error_msg)

    # Filter valid SMILES
    valid_indices: List[int] = []
    valid_smiles: List[str] = []
    for i, smi in enumerate(smiles_list):
        if smi and smi.strip() and Chem.MolFromSmiles(smi) is not None:
            valid_indices.append(i)
            valid_smiles.append(smi)

    if not valid_smiles:
        return [np.nan] * len(smiles_list)

    try:
        raw_scores = _DEEPDL_MODEL.screening(
            smiles_list=valid_smiles, naive=True, batch_size=64
        )
        # Normalize 0-100 to 0-1
        normalized = [
            max(0.0, min(1.0, float(s) / 100.0)) if s is not None else np.nan
            for s in raw_scores
        ]
    except Exception as e:
        print(f"  Warning: DeepDL scoring failed: {e}")
        return [np.nan] * len(smiles_list)

    # Reconstruct full list
    results: List[float] = [np.nan] * len(smiles_list)
    for idx, score in zip(valid_indices, normalized):
        results[idx] = score

    return results


def _compute_ring_score(smiles_list: Sequence[str]) -> List[float]:
    scores: List[float] = []
    for smi in tqdm(smiles_list, desc="Computing Ring Score"):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings == 0:
                scores.append(0.0)
            else:
                max_ring_size = max((len(ring) for ring in ring_info.AtomRings()), default=0)
                scores.append(0.0 if max_ring_size > 7 else 1.0)
        else:
            scores.append(np.nan)
    return scores


def _compute_toxicity(smiles_list: Sequence[str]) -> List[float]:
    print("  Loading chemprop toxicity model...")
    try:
        from modules.small_molecule_drug_design.scorer_mcp.chemprop_scorers_mcp.base import (  # noqa: PLC0415
            Scorer as ChempropScorer,
        )

        scorer = ChempropScorer()
        return scorer.score_primary_cell_toxicity(smiles_list)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"    Warning: Toxicity computation failed: {exc}")
        print("    Returning NaN for all molecules")
        return [np.nan] * len(smiles_list)


def _compute_antibiotics_novelty(smiles_list: Sequence[str]) -> List[float]:
    print("  Computing novelty against known antibiotics...")
    try:
        from modules.small_molecule_drug_design.scorer_mcp.antibiotics_scorer_mcp.base import (  # noqa: PLC0415
            Scorer as AntibioticsScorer,
        )

        scorer = AntibioticsScorer()
        return scorer.score_antibiotics_novelty(smiles_list)
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"    Warning: Novelty computation failed: {exc}")
        print("    Returning NaN for all molecules")
        return [np.nan] * len(smiles_list)


def _compute_antibiotics_motifs_filter(smiles_list: Sequence[str]) -> List[float]:
    print("  Computing antibiotics motifs filter...")
    try:
        from modules.small_molecule_drug_design.utils.rdkit_utils import (  # noqa: PLC0415
            filter_smiles_preserves_existing_hits,
        )

        results: List[Optional[float]] = [np.nan] * len(smiles_list)
        valid_smiles: List[str] = []
        valid_indices: List[int] = []

        for idx, smi in enumerate(smiles_list):
            if smi and smi.strip():
                valid_smiles.append(smi)
                valid_indices.append(idx)

        if not valid_smiles:
            return results

        kept_list, _ = filter_smiles_preserves_existing_hits(valid_smiles)
        kept_set = set(kept_list)
        for idx, smi in zip(valid_indices, valid_smiles):
            results[idx] = 1.0 if smi in kept_set else 0.0

        return results
    except Exception as exc:  # pragma: no cover - depends on optional deps
        print(f"    Warning: Motifs filter computation failed: {exc}")
        print("    Returning NaN for all molecules")
        return [np.nan] * len(smiles_list)


METRIC_FUNCTIONS = {
    "qed": _compute_qed,
    "sa": _compute_sa,
    "mw": _compute_mw,
    "pains": _compute_pains,
    "brenk": _compute_brenk,
    "deepdl": _compute_deepdl,
    "ring_score": _compute_ring_score,
    "toxicity": _compute_toxicity,
    "antibiotics_novelty": _compute_antibiotics_novelty,
    "antibiotics_motifs_filter": _compute_antibiotics_motifs_filter,
}


def ensure_held_out_metrics(
    df: pd.DataFrame,
    smiles_col: str = "smiles",
    metrics: Optional[Iterable[str]] = None,
    *,
    prefix: str = "",
    recompute: bool = False,
) -> pd.DataFrame:
    """Ensure the requested metrics exist on the dataframe."""
    df_copy = df.copy()
    metrics_to_compute = list(metrics or METRIC_FUNCTIONS.keys())

    if smiles_col not in df_copy.columns:
        raise ValueError(f"Missing SMILES column '{smiles_col}' in dataframe")

    smiles_series = df_copy[smiles_col]
    valid_mask = smiles_series.notna()
    valid_indices = smiles_series[valid_mask].index.tolist()
    smiles_list = smiles_series.loc[valid_indices].astype(str).tolist()

    if not smiles_list:
        return df_copy

    for metric_name in metrics_to_compute:
        column_name = f"{prefix}{metric_name}"
        needs_metric = recompute or column_name not in df_copy.columns or df_copy[column_name].isna().all()
        if not needs_metric:
            continue

        print(f"\n{'=' * 60}")
        print(f"Computing held-out metric: {metric_name}")
        print(f"{'=' * 60}")

        scores = METRIC_FUNCTIONS[metric_name](smiles_list)
        df_copy.loc[valid_indices, column_name] = scores

        non_null = df_copy[column_name].notna().sum()
        if non_null > 0:
            mean_val = df_copy[column_name].mean()
            threshold = HELD_OUT_FILTERS.get(metric_name, 0.5)
            pass_count = (df_copy[column_name] >= threshold).sum()
            pct = 100 * pass_count / non_null
            print(f"  ✓ Computed: {non_null}/{len(smiles_list)}")
            print(f"  Mean: {mean_val:.3f}")
            print(f"  Passing (>= {threshold}): {pass_count} ({pct:.1f}%)")
        else:
            print("  ⚠ Warning: All values are NaN")

    return df_copy


def summarize_pass_rates(
    df: pd.DataFrame,
    metrics: Optional[Iterable[str]] = None,
    *,
    prefix: str = "",
    thresholds: Optional[Dict[str, float]] = None,
) -> List[Dict[str, float]]:
    """Summarize pass counts/rates for the selected metrics."""
    summary: List[Dict[str, float]] = []
    metrics_to_summarize = list(metrics or METRIC_FUNCTIONS.keys())
    threshold_map = {**HELD_OUT_FILTERS, **(thresholds or {})}

    for metric_name in metrics_to_summarize:
        column_name = f"{prefix}{metric_name}"
        if column_name not in df.columns:
            continue

        series = df[column_name].dropna()
        total = int(len(series))
        if total == 0:
            continue

        threshold = threshold_map.get(metric_name, 0.5)
        pass_count = int((series >= threshold).sum())
        entry = {
            "metric_name": metric_name,
            "column_name": column_name,
            "threshold": float(threshold),
            "pass_count": pass_count,
            "total": total,
            "pass_rate": pass_count / total if total else 0.0,
        }
        summary.append(entry)

    return summary


__all__ = [
    "HELD_OUT_FILTERS",
    "ensure_held_out_metrics",
    "summarize_pass_rates",
]







