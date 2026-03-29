import os
import sys
import time
from typing import Dict, List, Optional
from loguru import logger

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

import requests

try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm is not available
    def tqdm(iterable=None, desc=None, total=None, leave=False, **kwargs):
        if iterable is None:
            return iterable
        return iterable

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

# Retry defaults (configurable via env vars)
DEFAULT_MAX_RETRIES = 5
DEFAULT_RETRY_DELAY = 2.0
DEFAULT_MAX_BACKOFF = 30.0
DEFAULT_TOTAL_TIMEOUT = 180.0  # seconds

_MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _smiles_to_fingerprint(smiles: str):
    """Convert SMILES to a Morgan fingerprint."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MORGAN_GENERATOR.GetFingerprint(mol)


def _tanimoto_similarity(smiles_a: str, smiles_b: str) -> Optional[float]:
    """Compute RDKit Tanimoto similarity between two SMILES."""
    fp_a = _smiles_to_fingerprint(smiles_a)
    if fp_a is None:
        return None
    fp_b = _smiles_to_fingerprint(smiles_b)
    if fp_b is None:
        return None
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _map_similarity_to_score(similarity: Optional[float]) -> Optional[float]:
    """Map similarity values to scores with linear interpolation between 0.5 and 1.0."""
    if similarity is None:
        return None
    if similarity <= 0.5:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    return (similarity - 0.5) / 0.5


class ArthorClient:
    """Thin client for Arthor similarity search."""

    def __init__(
        self,
        base_url: str = "http://arthor.docking.org",
        db_name: str = "REAL-Database-22Q1",
        max_results: int = 5,
        start: int = 0,
        timeout: int = 30,
        max_retries: int = DEFAULT_MAX_RETRIES,
        retry_delay: float = DEFAULT_RETRY_DELAY,
        max_backoff: float = DEFAULT_MAX_BACKOFF,
        total_timeout: float = DEFAULT_TOTAL_TIMEOUT,
    ):
        self.base_url = base_url.rstrip("/")
        self.db_name = db_name
        self.max_results = max_results
        self.start = start
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.max_backoff = max_backoff
        self.total_timeout = total_timeout

    def _search_single_smiles(self, smiles: str) -> List[str]:
        if not smiles or not smiles.strip():
            return []

        url = f"{self.base_url}/dt/{self.db_name}/search"
        params = {
            "query": smiles.strip(),
            "type": "SIM",
            "start": self.start,
            "length": self.max_results,
        }

        deadline = time.monotonic() + self.total_timeout
        for attempt in range(self.max_retries):
            try:
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()

                if isinstance(data, dict) and "data" in data:
                    results = data["data"]
                elif isinstance(data, list):
                    results = data
                else:
                    logger.warning(f"Unexpected Arthor response format: {type(data)}")
                    return []

                similar_smiles: List[str] = []
                for item in results:
                    if isinstance(item, (list, tuple)):
                        if len(item) > 2:
                            similar_smiles.append(str(item[2]))
                        elif len(item) > 1:
                            similar_smiles.append(str(item[1]))
                    elif isinstance(item, dict) and "smiles" in item:
                        similar_smiles.append(str(item["smiles"]))
                    elif isinstance(item, str):
                        similar_smiles.append(item)

                return similar_smiles[: self.max_results]

            except requests.exceptions.HTTPError as e:
                # Don't retry 4xx client errors
                if hasattr(e, "response") and 400 <= e.response.status_code < 500:
                    logger.error(f"Client error for SMILES '{smiles[:30]}...': {e}")
                    return []
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
                Exception,
            ) as e:
                logger.debug(
                    "Arthor attempt %d/%d failed for SMILES '%s': %s",
                    attempt + 1,
                    self.max_retries,
                    f"{smiles[:30]}...",
                    e,
                )

            # Exponential backoff with cap
            if attempt < self.max_retries - 1:
                now = time.monotonic()
                if now >= deadline:
                    logger.error(
                        f"Arthor query exceeded total timeout ({self.total_timeout}s) for SMILES '{smiles[:50]}...'"
                    )
                    break
                delay = min(
                    self.retry_delay * (2**attempt),
                    self.max_backoff,
                    max(0.0, deadline - now),
                )
                if delay > 0:
                    time.sleep(delay)

        logger.error(
            f"FAILED after {self.max_retries} retries for SMILES: {smiles[:50]}..."
        )
        return []

    def search_similar(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        """Search Arthor for each SMILES in the list."""
        total = len(smiles_list)
        print(f"[ARTHOR] Querying Arthor API for {total} SMILES", file=sys.stderr, flush=True)
        sys.stderr.flush()
        results: Dict[str, List[str]] = {}
        for idx, smiles in enumerate(smiles_list):
            print(f"[ARTHOR] Querying SMILES {idx+1}/{total}: {smiles[:50]}...", file=sys.stderr, flush=True)
            sys.stderr.flush()
            results[smiles] = self._search_single_smiles(smiles)
            num_hits = len(results[smiles])
            print(f"[ARTHOR] Got {num_hits} hits for SMILES {idx+1}/{total}", file=sys.stderr, flush=True)
            sys.stderr.flush()
            if (idx + 1) % max(1, total // 10) == 0 or idx == total - 1:
                print(f"[ARTHOR] Progress: Queried {idx + 1}/{total} SMILES", file=sys.stderr, flush=True)
                sys.stderr.flush()
        print(f"[ARTHOR] Completed queries for {total} SMILES", file=sys.stderr, flush=True)
        sys.stderr.flush()
        return results


class Scorer(BaseScorer):
    """Scoring Enamine REAL similarity via Arthor."""

    def __init__(self):
        super().__init__()

        base_url = os.getenv("ARTHOR_BASE_URL", "http://arthor.docking.org")
        db_name = os.getenv("ARTHOR_DB_NAME", "REAL-Database-22Q1")
        max_results = int(os.getenv("ARTHOR_MAX_RESULTS", "5"))
        start = int(os.getenv("ARTHOR_START", "0"))
        timeout = int(os.getenv("ARTHOR_TIMEOUT", "30"))
        max_retries = int(
            os.getenv("ARTHOR_MAX_RETRIES", str(DEFAULT_MAX_RETRIES))
        )
        retry_delay = float(
            os.getenv("ARTHOR_RETRY_DELAY", str(DEFAULT_RETRY_DELAY))
        )
        max_backoff = float(
            os.getenv("ARTHOR_MAX_BACKOFF", str(DEFAULT_MAX_BACKOFF))
        )
        total_timeout = float(
            os.getenv("ARTHOR_MAX_TOTAL_SECONDS", str(DEFAULT_TOTAL_TIMEOUT))
        )

        self._client = ArthorClient(
            base_url=base_url,
            db_name=db_name,
            max_results=max_results,
            start=start,
            timeout=timeout,
            max_retries=max_retries,
            retry_delay=retry_delay,
            max_backoff=max_backoff,
            total_timeout=total_timeout,
        )

    def _query_arthor(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        if not smiles_list:
            print(f"[ARTHOR] _query_arthor: empty list", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return {}
        print(f"[ARTHOR] _query_arthor: calling search_similar for {len(smiles_list)} SMILES", file=sys.stderr, flush=True)
        sys.stderr.flush()
        try:
            result = self._client.search_similar(smiles_list)
            print(f"[ARTHOR] _query_arthor: got results for {len(result)} SMILES", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return result
        except Exception as exc:  # pragma: no cover - defensive logging
            print(f"[ARTHOR] _query_arthor: ERROR - {exc}", file=sys.stderr, flush=True)
            sys.stderr.flush()
            logger.warning(f"Arthor similarity query failed: {exc}")
            return {smiles: [] for smiles in smiles_list}

    @scorer(
        name="arthor_similarity",
        population_wise=False,
        description=(
            "Similarity to Enamine REAL via Arthor API (0-1). "
            "Calculates RDKit Tanimoto vs the closest REAL hit; "
            "scores below 0.5 similarity map to 0, 1.0 similarity maps to 1.0."
        ),
    )
    def score_arthor_similarity(self, samples: List[str]) -> List[Optional[float]]:
        """Compute Arthor similarity scores for a list of SMILES."""
        print(f"[ARTHOR] score_arthor_similarity called with {len(samples)} samples", file=sys.stderr, flush=True)
        sys.stderr.flush()
        
        results: List[Optional[float]] = [None] * len(samples)

        valid_indices: List[int] = []
        valid_smiles: List[str] = []
        for idx, sample in enumerate(samples):
            if not isinstance(sample, str):
                results[idx] = None
                continue
            smiles = sample.strip()
            if not smiles:
                results[idx] = None
                continue
            if Chem.MolFromSmiles(smiles) is None:
                results[idx] = None
                continue
            valid_indices.append(idx)
            valid_smiles.append(smiles)

        print(f"[ARTHOR] Found {len(valid_smiles)} valid SMILES out of {len(samples)}", file=sys.stderr, flush=True)
        sys.stderr.flush()

        if not valid_smiles:
            print(f"[ARTHOR] No valid SMILES, returning", file=sys.stderr, flush=True)
            sys.stderr.flush()
            return results

        # Deduplicate for efficiency
        unique_smiles = list(dict.fromkeys(valid_smiles))
        print(f"[ARTHOR] Querying {len(unique_smiles)} unique SMILES", file=sys.stderr, flush=True)
        sys.stderr.flush()
        arthor_hits = self._query_arthor(unique_smiles)

        # Process results with progress bar
        pairs = list(zip(valid_indices, valid_smiles))
        print(f"[ARTHOR] Computing similarity scores for {len(pairs)} pairs", file=sys.stderr, flush=True)
        sys.stderr.flush()
        
        for pair_idx, (idx, smiles) in enumerate(pairs):
            hits = arthor_hits.get(smiles, [])
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
            
            if (pair_idx + 1) % max(1, len(pairs) // 10) == 0 or pair_idx == len(pairs) - 1:
                print(f"[ARTHOR] Computed scores for {pair_idx + 1}/{len(pairs)} pairs", file=sys.stderr, flush=True)
                sys.stderr.flush()

        scored_count = sum(1 for r in results if r is not None)
        print(f"[ARTHOR] Completed scoring: {scored_count}/{len(samples)} samples scored", file=sys.stderr, flush=True)
        sys.stderr.flush()
        return results

