import os
import sys
import threading
from pathlib import Path
from typing import Dict, List, Optional, Any

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
import numpy as np

try:
    import faiss
except ImportError:
    faiss = None
    print(
        "[LOCAL_SIMILARITY] WARNING: faiss not installed. Please install faiss-cpu or faiss-gpu.",
        file=sys.stderr,
        flush=True,
    )

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

# Fingerprinter - MACCS keys (167 dimensions)
_MACCS_FINGERPRINTER = AllChem.GetMACCSKeysFingerprint


def _smiles_to_fingerprint(smiles: str):
    """Convert SMILES to MACCS fingerprint (numpy array for FAISS)."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    bv = _MACCS_FINGERPRINTER(mol)
    fp = np.zeros(len(bv), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(bv, fp)
    return fp


def _smiles_to_bitvector(smiles: str):
    """Convert SMILES to RDKit bit vector for Tanimoto."""
    if not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _MACCS_FINGERPRINTER(mol)


def _map_similarity_to_score(similarity: Optional[float]) -> Optional[float]:
    """Map similarity to score: <0.5 -> 0.0, 1.0 -> 1.0, linear in between."""
    if similarity is None:
        return None
    if similarity <= 0.5:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    return (similarity - 0.5) / 0.5


class Scorer(BaseScorer):
    """Scoring Enamine REAL similarity via local FAISS index."""

    # Class-level cache (shared across instances in same process)
    _lock = threading.Lock()
    _index = None
    _smiles_list = None
    _gpu_resources = None
    _use_gpu = None

    def __init__(self):
        if faiss is None:
            raise RuntimeError(
                "faiss is not installed. Please install faiss-cpu or faiss-gpu."
            )

        module_dir = Path(__file__).parent.absolute()
        index_path = Path(
            os.getenv(
                "LOCAL_SIMILARITY_INDEX_FILE",
                str(module_dir / "data" / "enamine_10m_search.index"),
            )
        )
        smiles_path = Path(
            os.getenv(
                "LOCAL_SIMILARITY_SMILES_FILE",
                str(module_dir / "data" / "enamine_10m_search_smiles.txt"),
            )
        )

        if not index_path.is_absolute():
            index_path = module_dir / index_path
        if not smiles_path.is_absolute():
            smiles_path = module_dir / smiles_path

        if not index_path.exists():
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not smiles_path.exists():
            raise FileNotFoundError(f"SMILES file not found: {smiles_path}")

        self._index_path = index_path
        self._smiles_path = smiles_path
        self._k = int(os.getenv("LOCAL_SIMILARITY_K", "5"))
        super().__init__()

    def _load_resources(self):
        """Load FAISS index and SMILES file once."""
        if self.__class__._index is not None:
            return

        with self.__class__._lock:
            if self.__class__._index is not None:
                return

            print(
                f"[LOCAL_SIMILARITY] Loading FAISS index...",
                file=sys.stderr,
                flush=True,
            )
            cpu_index = faiss.read_index(str(self._index_path))

            if self.__class__._use_gpu is None:
                if faiss.get_num_gpus() == 0:
                    raise RuntimeError("GPU is required but not available")
                self.__class__._gpu_resources = faiss.StandardGpuResources()
                self.__class__._index = faiss.index_cpu_to_gpu(
                    self.__class__._gpu_resources, 0, cpu_index
                )
                self.__class__._use_gpu = True

            print(
                f"[LOCAL_SIMILARITY] Loading SMILES file...",
                file=sys.stderr,
                flush=True,
            )
            with open(self._smiles_path, "r") as f:
                self.__class__._smiles_list = [line.strip() for line in f]

            print(
                f"[LOCAL_SIMILARITY] Ready: {self.__class__._index.ntotal} molecules, k={self._k}",
                file=sys.stderr,
                flush=True,
            )

    @scorer(
        name="local_similarity",
        population_wise=False,
        description=(
            "Similarity to Enamine REAL via local FAISS index (0-1). "
            "Calculates RDKit Tanimoto vs the closest REAL hit; "
            "scores below 0.5 similarity map to 0, 1.0 similarity maps to 1.0."
        ),
    )
    def score_local_similarity(self, samples: List[str]) -> List[Optional[float]]:
        """Compute local similarity scores for a list of SMILES."""
        print(
            f"[LOCAL_SIMILARITY] score_local_similarity called with {len(samples)} samples",
            file=sys.stderr,
            flush=True,
        )
        self._load_resources()

        results: List[Optional[float]] = [None] * len(samples)

        # Filter valid SMILES
        valid_data = []
        for idx, sample in enumerate(samples):
            if not isinstance(sample, str):
                continue
            smiles = sample.strip()
            if not smiles or Chem.MolFromSmiles(smiles) is None:
                continue
            valid_data.append((idx, smiles))

        print(
            f"[LOCAL_SIMILARITY] Found {len(valid_data)} valid SMILES out of {len(samples)}",
            file=sys.stderr,
            flush=True,
        )

        if not valid_data:
            return results

        # Deduplicate
        unique_smiles = {}
        for idx, smiles in valid_data:
            if smiles not in unique_smiles:
                unique_smiles[smiles] = []
            unique_smiles[smiles].append(idx)

        print(
            f"[LOCAL_SIMILARITY] Querying {len(unique_smiles)} unique SMILES",
            file=sys.stderr,
            flush=True,
        )

        # Batch compute fingerprints and search FAISS
        print(
            f"[LOCAL_SIMILARITY] Computing fingerprints...", file=sys.stderr, flush=True
        )
        fingerprints = []
        smiles_order = []
        for smiles in unique_smiles.keys():
            fp = _smiles_to_fingerprint(smiles)
            if fp is not None:
                fingerprints.append(fp)
                smiles_order.append(smiles)

        if not fingerprints:
            return results

        print(
            f"[LOCAL_SIMILARITY] Batch FAISS search for {len(fingerprints)} fingerprints...",
            file=sys.stderr,
            flush=True,
        )
        fp_array = np.array(fingerprints, dtype=np.float32)
        distances, indices = self.__class__._index.search(fp_array, self._k)
        print(
            f"[LOCAL_SIMILARITY] FAISS search completed, got {len(indices)} result rows",
            file=sys.stderr,
            flush=True,
        )

        # Get neighbor SMILES
        all_neighbor_indices = []
        for row in indices:
            all_neighbor_indices.extend([int(i) for i in row])

        print(
            f"[LOCAL_SIMILARITY] Fetching {len(all_neighbor_indices)} neighbor SMILES...",
            file=sys.stderr,
            flush=True,
        )
        neighbor_smiles = [
            (
                self.__class__._smiles_list[i]
                if 0 <= i < len(self.__class__._smiles_list)
                else None
            )
            for i in all_neighbor_indices
        ]

        # Compute Tanimoto similarities using BulkTanimotoSimilarity for speed
        print(
            f"[LOCAL_SIMILARITY] Computing Tanimoto similarities...",
            file=sys.stderr,
            flush=True,
        )

        # Pre-compute all query bit vectors
        query_bvs = {}
        for query_smiles in smiles_order:
            query_bv = _smiles_to_bitvector(query_smiles)
            if query_bv is not None:
                query_bvs[query_smiles] = query_bv

        # Pre-compute all neighbor bit vectors (cache)
        neighbor_bv_cache = {}
        for neighbor_smiles_str in neighbor_smiles:
            if (
                neighbor_smiles_str is not None
                and neighbor_smiles_str not in neighbor_bv_cache
            ):
                neighbor_bv = _smiles_to_bitvector(neighbor_smiles_str)
                if neighbor_bv is not None:
                    neighbor_bv_cache[neighbor_smiles_str] = neighbor_bv

        # Batch compute similarities using BulkTanimotoSimilarity
        for i, query_smiles in enumerate(smiles_order):
            query_bv = query_bvs.get(query_smiles)
            if query_bv is None:
                continue

            start_idx = i * self._k
            neighbor_bvs_list = []
            for j in range(self._k):
                neighbor_smiles_str = neighbor_smiles[start_idx + j]
                if neighbor_smiles_str and neighbor_smiles_str in neighbor_bv_cache:
                    neighbor_bvs_list.append(neighbor_bv_cache[neighbor_smiles_str])

            if neighbor_bvs_list:
                # Use BulkTanimotoSimilarity for batch computation
                sims = DataStructs.BulkTanimotoSimilarity(query_bv, neighbor_bvs_list)
                best_sim = float(max(sims)) if sims else None
            else:
                best_sim = None

            # Map similarity to score and assign to all indices with this SMILES
            score = _map_similarity_to_score(best_sim) if best_sim is not None else 0.0
            for idx in unique_smiles[query_smiles]:
                results[idx] = score

            # Progress logging
            if (i + 1) % max(1, len(smiles_order) // 10) == 0 or i == len(
                smiles_order
            ) - 1:
                print(
                    f"[LOCAL_SIMILARITY] Processed {i + 1}/{len(smiles_order)} unique SMILES",
                    file=sys.stderr,
                    flush=True,
                )

        scored_count = sum(1 for r in results if r is not None)
        print(
            f"[LOCAL_SIMILARITY] Completed scoring: {scored_count}/{len(samples)} samples scored",
            file=sys.stderr,
            flush=True,
        )
        return results
