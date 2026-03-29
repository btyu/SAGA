from typing import Dict, List, Optional
import logging

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate

from modules.small_molecule_drug_design.utils.arthor_utils import (
    search_similar_compounds_parallel,
)

logger = logging.getLogger(__name__)


_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _smiles_to_fingerprint(smiles: str):
    """Convert a SMILES string to an RDKit fingerprint."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def _tanimoto_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
    """Compute Tanimoto similarity between two SMILES strings."""
    fp_a = _smiles_to_fingerprint(smiles_a)
    if fp_a is None:
        return None

    fp_b = _smiles_to_fingerprint(smiles_b)
    if fp_b is None:
        return None

    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _map_similarity_to_score(similarity: Optional[float]) -> Optional[float]:
    """Convert similarity to 0-1 score with linear scaling between 0.5 and 1.0."""
    if similarity is None:
        return None
    if similarity <= 0.5:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    # Linear interpolation: 0.5 -> 0.0, 1.0 -> 1.0
    return (similarity - 0.5) / 0.5


@register_scorer_class
class ArthorSimilarityScorer:
    """Score molecules by similarity to Enamine REAL compounds via Arthor API."""

    def __init__(
        self,
        db_name: str = "REAL-Database-22Q1",
        max_results: int = 5,
        start: int = 0,
        base_url: str = "https://arthor.docking.org",
        max_workers: int = 1,
    ):
        self._db_name = db_name
        self._max_results = max_results
        self._start = start
        self._base_url = base_url
        self._max_workers = max_workers

    def _query_arthor(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        """Query Arthor for similar compounds."""
        if not smiles_list:
            return {}
        try:
            return search_similar_compounds_parallel(
                smiles_list=smiles_list,
                db_name=self._db_name,
                max_results=self._max_results,
                start=self._start,
                base_url=self._base_url,
                max_workers=self._max_workers,
            )
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning(f"Arthor similarity query failed: {exc}")
            return {smiles: [] for smiles in smiles_list}

    @register_scorer(
        name="arthor_similarity",
        population_wise=False,
        description=(
            "Similarity to Enamine REAL via Arthor API (0-1). "
            "Calculates RDKit Tanimoto vs the closest REAL hit; "
            "scores below 0.5 similarity map to 0, 1.0 similarity maps to 1.0."
        ),
    )
    def score_arthor_similarity(
        self, candidates: List[Candidate]
    ) -> List[float]:
        """Score candidates by similarity to their closest Enamine REAL analog."""
        results: List[float] = [0.0] * len(candidates)

        # Prepare query list for valid candidates
        query_indices: List[int] = []
        query_smiles: List[str] = []
        for idx, candidate in enumerate(candidates):
            smiles = getattr(candidate, "representation", None)
            if not isinstance(smiles, str):
                results[idx] = 0.0
                continue

            smiles = smiles.strip()
            if not smiles:
                results[idx] = 0.0
                continue

            if Chem.MolFromSmiles(smiles) is None:
                results[idx] = 0.0
                continue

            query_indices.append(idx)
            query_smiles.append(smiles)

        if not query_smiles:
            return results

        # Deduplicate SMILES to avoid redundant Arthor calls
        unique_smiles: Dict[str, None] = {}
        for smiles in query_smiles:
            if smiles not in unique_smiles:
                unique_smiles[smiles] = None

        # Query Arthor API for similar compounds
        arthor_hits = self._query_arthor(list(unique_smiles.keys()))

        # Evaluate similarity for each candidate
        for idx, smiles in zip(query_indices, query_smiles):
            hits = arthor_hits.get(smiles, [])
            if not hits:
                results[idx] = 0.0
                continue

            # Evaluate highest similarity among returned hits
            best_similarity: Optional[float] = None
            for hit_smiles in hits:
                similarity = _tanimoto_similarity(smiles, hit_smiles)
                if similarity is None:
                    continue
                if best_similarity is None or similarity > best_similarity:
                    best_similarity = similarity

            mapped_score = _map_similarity_to_score(best_similarity)
            results[idx] = mapped_score if mapped_score is not None else 0.0

        return results

