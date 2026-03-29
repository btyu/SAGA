import json
import os
import time
from typing import Dict, List, Optional

import requests
from loguru import logger
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

MAX_RETRIES = int(os.getenv("SMALLWORLD_MAX_RETRIES", "6"))
RETRY_DELAY = float(os.getenv("SMALLWORLD_RETRY_DELAY", "2.0"))
MAX_BACKOFF = float(os.getenv("SMALLWORLD_MAX_BACKOFF", "60.0"))

DEFAULT_BASE_URL = os.getenv("SMALLWORLD_BASE_URL", "https://sw.docking.org")
DEFAULT_DB_NAME = os.getenv("SMALLWORLD_DB_NAME", "REALDB-2025-07.smi.anon")
DEFAULT_TOP = int(os.getenv("SMALLWORLD_TOP", "5"))
DEFAULT_TIMEOUT = int(os.getenv("SMALLWORLD_TIMEOUT", "60"))
DEFAULT_SCORES = os.getenv("SMALLWORLD_SCORES", "Atom Alignment,ECFP4,Daylight")
DEFAULT_DRAW = int(os.getenv("SMALLWORLD_DRAW", "10"))
DEFAULT_MIN_INTERVAL = float(os.getenv("SMALLWORLD_MIN_INTERVAL", "5.0"))

DEFAULT_BOUNDS = {
    "dist": int(os.getenv("SMALLWORLD_DIST", "8")),
    "sdist": int(os.getenv("SMALLWORLD_SDIST", "12")),
    "tdn": int(os.getenv("SMALLWORLD_TDN", "6")),
    "tup": int(os.getenv("SMALLWORLD_TUP", "6")),
    "rdn": int(os.getenv("SMALLWORLD_RDN", "6")),
    "rup": int(os.getenv("SMALLWORLD_RUP", "2")),
    "ldn": int(os.getenv("SMALLWORLD_LDN", "2")),
    "lup": int(os.getenv("SMALLWORLD_LUP", "2")),
    "maj": int(os.getenv("SMALLWORLD_MAJ", "6")),
    "min": int(os.getenv("SMALLWORLD_MIN", "6")),
    "sub": int(os.getenv("SMALLWORLD_SUB", "6")),
}

DEFAULT_SUBMIT_PARAMS = {
    "dist": DEFAULT_BOUNDS["dist"],
    "tdn": DEFAULT_BOUNDS["tdn"],
    "rdn": DEFAULT_BOUNDS["rdn"],
    "rup": DEFAULT_BOUNDS["rup"],
    "ldn": DEFAULT_BOUNDS["ldn"],
    "lup": DEFAULT_BOUNDS["lup"],
    "maj": DEFAULT_BOUNDS["maj"],
    "min": DEFAULT_BOUNDS["min"],
    "sub": DEFAULT_BOUNDS["sub"],
    "sdist": DEFAULT_BOUNDS["sdist"],
    "tup": DEFAULT_BOUNDS["tup"],
    "scores": DEFAULT_SCORES,
}

VALID_EXPORT_COLUMNS = {
    "columns[0][data]": "0",
    "columns[0][name]": "alignment",
    "columns[0][searchable]": "true",
    "columns[0][orderable]": "false",
    "columns[0][search][value]": "",
    "columns[0][search][regex]": "false",
    "columns[1][data]": "1",
    "columns[1][name]": "dist",
    "columns[1][searchable]": "true",
    "columns[1][orderable]": "true",
    "columns[1][search][value]": "0-12",
    "columns[1][search][regex]": "false",
    "columns[2][data]": "2",
    "columns[2][name]": "ecfp4",
    "columns[2][searchable]": "true",
    "columns[2][orderable]": "true",
    "columns[2][search][value]": "",
    "columns[2][search][regex]": "false",
    "columns[3][data]": "3",
    "columns[3][name]": "daylight",
    "columns[3][searchable]": "true",
    "columns[3][orderable]": "true",
    "columns[3][search][value]": "",
    "columns[3][search][regex]": "false",
    "columns[4][data]": "4",
    "columns[4][name]": "topodist",
    "columns[4][searchable]": "true",
    "columns[4][orderable]": "true",
    "columns[4][search][value]": "0-8",
    "columns[4][search][regex]": "false",
    "columns[5][data]": "5",
    "columns[5][name]": "mces",
    "columns[5][searchable]": "true",
    "columns[5][orderable]": "true",
    "columns[5][search][value]": "",
    "columns[5][search][regex]": "false",
    "columns[6][data]": "6",
    "columns[6][name]": "tdn",
    "columns[6][searchable]": "true",
    "columns[6][orderable]": "true",
    "columns[6][search][value]": "0-6",
    "columns[6][search][regex]": "false",
    "columns[7][data]": "7",
    "columns[7][name]": "tup",
    "columns[7][searchable]": "true",
    "columns[7][orderable]": "true",
    "columns[7][search][value]": "0-6",
    "columns[7][search][regex]": "false",
    "columns[8][data]": "8",
    "columns[8][name]": "rdn",
    "columns[8][searchable]": "true",
    "columns[8][orderable]": "true",
    "columns[8][search][value]": "0-6",
    "columns[8][search][regex]": "false",
    "columns[9][data]": "9",
    "columns[9][name]": "rup",
    "columns[9][searchable]": "true",
    "columns[9][orderable]": "true",
    "columns[9][search][value]": "0-2",
    "columns[9][search][regex]": "false",
    "columns[10][data]": "10",
    "columns[10][name]": "ldn",
    "columns[10][searchable]": "true",
    "columns[10][orderable]": "true",
    "columns[10][search][value]": "0-2",
    "columns[10][search][regex]": "false",
    "columns[11][data]": "11",
    "columns[11][name]": "lup",
    "columns[11][searchable]": "true",
    "columns[11][orderable]": "true",
    "columns[11][search][value]": "0-2",
    "columns[11][search][regex]": "false",
    "columns[12][data]": "12",
    "columns[12][name]": "mut",
    "columns[12][searchable]": "true",
    "columns[12][orderable]": "true",
    "columns[12][search][value]": "",
    "columns[12][search][regex]": "false",
    "columns[13][data]": "13",
    "columns[13][name]": "maj",
    "columns[13][searchable]": "true",
    "columns[13][orderable]": "true",
    "columns[13][search][value]": "0-6",
    "columns[13][search][regex]": "false",
    "columns[14][data]": "14",
    "columns[14][name]": "min",
    "columns[14][searchable]": "true",
    "columns[14][orderable]": "true",
    "columns[14][search][value]": "0-6",
    "columns[14][search][regex]": "false",
    "columns[15][data]": "15",
    "columns[15][name]": "hyb",
    "columns[15][searchable]": "true",
    "columns[15][orderable]": "true",
    "columns[15][search][value]": "0-6",
    "columns[15][search][regex]": "false",
    "columns[16][data]": "16",
    "columns[16][name]": "sub",
    "columns[16][searchable]": "true",
    "columns[16][orderable]": "true",
    "columns[16][search][value]": "0-6",
    "columns[16][search][regex]": "false",
    "order[0][column]": "0",
    "order[0][dir]": "asc",
    "search[value]": "",
    "search[regex]": "false",
}

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
    if similarity is None or similarity <= 0.5:
        return 0.0
    if similarity >= 1.0:
        return 1.0
    return (similarity - 0.5) / 0.5


class SmallWorldClient:
    """Thin client for querying the SmallWorld search API."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        db_name: str = DEFAULT_DB_NAME,
        top: int = DEFAULT_TOP,
        timeout: int = DEFAULT_TIMEOUT,
        scores: Optional[str] = DEFAULT_SCORES,
        bounds: Optional[Dict[str, Optional[int]]] = None,
        draw: int = DEFAULT_DRAW,
        min_request_interval: float = DEFAULT_MIN_INTERVAL,
    ):
        self.base_url = base_url.rstrip("/")
        self.db_name = db_name
        self.top = max(1, int(top))
        self.timeout = timeout
        self.scores = scores or DEFAULT_SCORES
        self.draw = draw
        self._min_request_interval = max(0.0, min_request_interval)
        self._last_submit_ts: Optional[float] = None

        merged_bounds = dict(DEFAULT_BOUNDS)
        if bounds:
            merged_bounds.update({k: v for k, v in bounds.items() if v is not None})
        self.bounds = merged_bounds

        self._submit_params = dict(DEFAULT_SUBMIT_PARAMS)
        self._submit_params.update(
            {k: v for k, v in self.bounds.items() if v is not None}
        )
        if self.scores:
            self._submit_params["scores"] = self.scores

        self._view_params = dict(VALID_EXPORT_COLUMNS)
        self._search_url = f"{self.base_url}/search/submit"
        self._view_url = f"{self.base_url}/search/view"

    @staticmethod
    def _clean_smiles(smiles: str) -> str:
        if not smiles:
            return ""
        smiles = smiles.strip()
        if " " in smiles:
            return smiles.split()[0].strip()
        return smiles

    def _throttle_if_needed(self) -> None:
        if self._last_submit_ts is None:
            return
        elapsed = time.monotonic() - self._last_submit_ts
        remaining = self._min_request_interval - elapsed
        if remaining > 0:
            time.sleep(remaining)

    def _submit_query(self, smiles: str) -> int:
        params = dict(self._submit_params)
        params["smi"] = smiles
        params["db"] = self.db_name

        self._throttle_if_needed()
        self._last_submit_ts = time.monotonic()

        response = requests.get(
            self._search_url,
            params=params,
            stream=True,
            timeout=self.timeout,
        )
        response.raise_for_status()

        hlid: Optional[int] = None
        for line in response.iter_lines(decode_unicode=True):
            if not line or "data:" not in line:
                continue
            payload = json.loads(line.split("data:", 1)[1].strip())
            if "hlid" in payload:
                hlid = int(payload["hlid"])
            if payload.get("status") == "END":
                break

        response.close()

        if hlid is None:
            raise RuntimeError("SmallWorld submit did not return a hit list id")
        return hlid

    def _fetch_hits(self, hlid: int) -> List[str]:
        params = {
            "hlid": hlid,
            "start": 0,
            "length": self.top,
            "draw": self.draw,
        }
        params.update(self._view_params)

        response = requests.get(self._view_url, params=params, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()

        if not payload.get("recordsTotal"):
            return []

        rows = payload.get("data") or []
        hits: List[str] = []
        for row in rows:
            if not isinstance(row, list) or not row:
                continue
            first = row[0]
            if not isinstance(first, dict):
                continue
            raw = first.get("hitSmiles") or first.get("smiles")
            cleaned = self._clean_smiles(raw or "")
            if cleaned:
                hits.append(cleaned)
        return hits[: self.top]

    def _query_single_smiles(self, smiles: str) -> List[str]:
        for attempt in range(MAX_RETRIES):
            try:
                hlid = self._submit_query(smiles)
                return self._fetch_hits(hlid)
            except requests.exceptions.HTTPError as exc:
                status = exc.response.status_code if exc.response else None
                if status and 400 <= status < 500:
                    logger.warning(
                        "SmallWorld client error (%s) for %s: %s", status, smiles, exc
                    )
                    return []
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as exc:
                logger.warning("SmallWorld connection issue for %s: %s", smiles, exc)
            except Exception as exc:
                logger.warning("Unexpected SmallWorld error for %s: %s", smiles, exc)

            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_DELAY * (2**attempt), MAX_BACKOFF)
                time.sleep(delay)
        logger.error("SmallWorld request exhausted retries for %s", smiles)
        return []

    def search_similar(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for smiles in smiles_list:
            query = smiles.strip() if isinstance(smiles, str) else ""
            if not query:
                results[smiles] = []
                continue
            hits = self._query_single_smiles(query)
            results[smiles] = hits
        return results


class Scorer(BaseScorer):
    """SmallWorld similarity scorer."""

    def __init__(self):
        super().__init__()
        base_url = os.getenv("SMALLWORLD_BASE_URL", DEFAULT_BASE_URL)
        db_name = os.getenv("SMALLWORLD_DB_NAME", DEFAULT_DB_NAME)
        top = int(os.getenv("SMALLWORLD_TOP", str(DEFAULT_TOP)))
        timeout = int(os.getenv("SMALLWORLD_TIMEOUT", str(DEFAULT_TIMEOUT)))
        scores = os.getenv("SMALLWORLD_SCORES", DEFAULT_SCORES)

        bounds_override = {}
        for key in DEFAULT_BOUNDS.keys():
            env_key = f"SMALLWORLD_{key.upper()}"
            if env_key in os.environ:
                try:
                    bounds_override[key] = int(os.environ[env_key])
                except ValueError:
                    bounds_override[key] = DEFAULT_BOUNDS[key]

        self._client = SmallWorldClient(
            base_url=base_url,
            db_name=db_name,
            top=top,
            timeout=timeout,
            scores=scores,
            bounds=bounds_override or None,
        )

    def _query_smallworld(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        if not smiles_list:
            return {}
        try:
            # Query sequentially - SmallWorld doesn't support parallel requests
            results = {}
            for smiles in smiles_list:
                query_result = self._client.search_similar([smiles])
                results.update(query_result)
            return results
        except Exception as exc:  # pragma: no cover - defensive logging
            logger.warning("SmallWorld similarity query failed: %s", exc)
            return {smiles: [] for smiles in smiles_list}

    @scorer(
        name="small_world_similarity",
        population_wise=False,
        description=(
            "Similarity to Enamine REAL via SmallWorld API (0-1). "
            "Scores computed by RDKit Tanimoto similarity against closest hits."
        ),
    )
    def score_small_world_similarity(self, samples: List[str]) -> List[Optional[float]]:
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

        if not valid_smiles:
            return results

        unique_smiles = list(dict.fromkeys(valid_smiles))
        hits_by_smiles = self._query_smallworld(unique_smiles)

        for idx, smiles in zip(valid_indices, valid_smiles):
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
