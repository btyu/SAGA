# from __future__ import annotations

from typing import Dict, List, Optional
import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scileo_agent.core.registry import register_scorer, register_scorer_class
from scileo_agent.core.data_models import Candidate

from modules.small_molecule_drug_design.utils.small_world_utils import (
    search_similar_compounds,
    DEFAULT_BASE_URL as SW_BASE_URL,
    DEFAULT_DB_NAME as SW_DB_NAME,
    DEFAULT_SCORES as SW_DEFAULT_SCORES,
)

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


def _map_similarity_to_score(similarity: Optional[float]) -> float:
    if similarity is None or similarity <= 0.5:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    return (similarity - 0.5) / 0.5


@register_scorer_class
class SmallWorldSimilarityScorer:
    """Score molecules via similarity to Enamine REAL compounds using SmallWorld."""

    def __init__(
        self,
        db_name: str = SW_DB_NAME,
        top: int = 5,
        base_url: str = SW_BASE_URL,
        timeout: int = 60,
        scores: Optional[str] = SW_DEFAULT_SCORES,
    ):
        self._db_name = db_name
        self._top = top
        self._base_url = base_url
        self._timeout = timeout
        self._scores = scores

    def _query_smallworld(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        if not smiles_list:
            return {}
        try:
            # Query SmallWorld sequentially (service doesn't support parallel requests)
            results = {}
            for smiles in unique_smiles:
                result = search_similar_compounds(
                    smiles, db_name=self._db_name, top=self._top
                )
                results.update(result)
            return results
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("SmallWorld similarity query failed: %s", exc)
            return {smiles: [] for smiles in smiles_list}

    @register_scorer(
        name="small_world_similarity",
        population_wise=False,
        description=(
            "Similarity to Enamine REAL via SmallWorld API (0-1). "
            "Computes RDKit Tanimoto similarity vs closest hits returned by SmallWorld."
        ),
    )
    def score_small_world_similarity(self, candidates: List[Candidate]) -> List[float]:
        """Score candidates by similarity to their closest SmallWorld hits."""
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

        deduped: Dict[str, None] = {}
        unique_smiles = []
        for smiles in query_smiles:
            if smiles not in deduped:
                deduped[smiles] = None
                unique_smiles.append(smiles)

        # Query SmallWorld sequentially (service doesn't support parallel requests)
        hits_by_smiles = {}
        for smiles in unique_smiles:
            result = search_similar_compounds(
                smiles, db_name=self._db_name, top=self._top
            )
            hits_by_smiles.update(result)

        for idx, smiles in zip(query_indices, query_smiles):
            hits = hits_by_smiles.get(smiles, [])
            if not hits:
                results[idx] = 0.0
                continue

            best_similarity: Optional[float] = None
            for hit_smiles in hits:
                similarity = _tanimoto_similarity(smiles, hit_smiles)
                if similarity is None:
                    continue
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity

            results[idx] = _map_similarity_to_score(best_similarity)

        return results
