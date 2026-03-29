"""
Helper to register a conventional Arthor similarity scorer when MCP version is disabled.
"""

from typing import List, Optional, Dict, Any

import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scileo_agent.core.data_models import Candidate
from scileo_agent.core.registry import ScorerManager

from .arthor_utils import search_similar_compounds_parallel

logger = logging.getLogger(__name__)

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _smiles_to_fingerprint(smiles: str):
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def _tanimoto_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
    fp_a = _smiles_to_fingerprint(smiles_a)
    if fp_a is None:
        return None
    fp_b = _smiles_to_fingerprint(smiles_b)
    if fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _map_similarity_to_score(similarity: Optional[float]) -> Optional[float]:
    if similarity is None:
        return None
    if similarity <= 0.5:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    return (similarity - 0.5) / 0.5


def _build_metadata(serializer_name: Optional[str] = None) -> Dict[str, Any]:
    metadata = {
        "description": (
            "Similarity to Enamine REAL via Arthor API (0-1). Calculates RDKit "
            "Tanimoto vs the closest REAL hit; scores below 0.5 similarity map "
            "to 0, 1.0 similarity maps to 1.0."
        ),
        "population_wise": False,
        "type": "candidate-wise",
    }
    if serializer_name:
        metadata["serializer"] = serializer_name
    return metadata


def register_manual_arthor_scorer(
    *,
    serializer_name: Optional[str] = None,
    db_name: str = "REAL-Database-22Q1",
    max_results: int = 5,
    start: int = 0,
    base_url: str = "http://arthor.docking.org",
    max_workers: int = 1,
    timeout: int = 30,
):
    """
    Register the Arthor similarity scorer directly with ScorerManager if not already present.
    """

    manager = ScorerManager()
    if manager.get_scorer("arthor_similarity") is not None:
        return

    def _score_arthor(
        candidates: List[Candidate], force_evaluation: bool = False, **kwargs
    ) -> List[float]:
        results: List[float] = [0.0] * len(candidates)
        query_indices: List[int] = []
        query_smiles: List[str] = []

        for idx, candidate in enumerate(candidates):
            smiles = getattr(candidate, "representation", None)
            if not isinstance(smiles, str):
                continue
            smiles = smiles.strip()
            if not smiles:
                continue
            if Chem.MolFromSmiles(smiles) is None:
                continue
            query_indices.append(idx)
            query_smiles.append(smiles)

        if not query_smiles:
            return results

        # Deduplicate to reduce API calls
        unique_smiles = list(dict.fromkeys(query_smiles))
        logger.info(
            "Arthor similarity scorer: querying %d candidates (%d unique SMILES)...",
            len(query_smiles),
            len(unique_smiles),
        )
        arthor_hits = search_similar_compounds_parallel(
            smiles_list=unique_smiles,
            db_name=db_name,
            max_results=max_results,
            start=start,
            base_url=base_url,
            max_workers=max_workers,
            timeout=timeout,
        )
        logger.info(
            "Arthor similarity scorer: completed search for %d unique SMILES.",
            len(arthor_hits),
        )

        for idx, smiles in zip(query_indices, query_smiles):
            hits = arthor_hits.get(smiles, [])
            if not hits:
                continue
            best_similarity: Optional[float] = None
            for hit_smiles in hits:
                similarity = _tanimoto_similarity(smiles, hit_smiles)
                if similarity is None:
                    continue
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity
            mapped_score = _map_similarity_to_score(best_similarity)
            if mapped_score is not None:
                results[idx] = mapped_score

        return results

    metadata = _build_metadata(serializer_name)
    manager.register_scorer(
        scorer_func=_score_arthor,
        name="arthor_similarity",
        metadata=metadata,
        type="candidate-wise",
        is_mcp_scorer=False,
    )
