"""
Arthor API helper copied from the MCP arthor_similarity scorer.
"""

import concurrent.futures
import logging
import time
from typing import Dict, List, Tuple, Union

import requests

logger = logging.getLogger(__name__)

MAX_RETRIES = 999_999
RETRY_DELAY = 2.0
MAX_BACKOFF = 120.0

DEFAULT_PARALLEL_WORKERS = 12
MAX_PARALLEL_WORKERS = 12


class ArthorClient:
    """Thin client for Arthor similarity search."""

    def __init__(
        self,
        base_url: str = "http://arthor.docking.org",
        db_name: str = "REAL-Database-22Q1",
        max_results: int = 5,
        start: int = 0,
        timeout: int = 30,
    ):
        self.base_url = base_url.rstrip("/")
        self.db_name = db_name
        self.max_results = max_results
        self.start = start
        self.timeout = timeout

    def _search_single_smiles(self, smiles: str) -> List[str]:
        if not smiles or not smiles.strip():
            return []

        url = f"{self.base_url}/dt/{self.db_name}/search"
        smiles_clean = smiles.strip()
        truncated_smiles = (
            f"{smiles_clean[:57]}..." if len(smiles_clean) > 60 else smiles_clean
        )
        params = {
            "query": smiles_clean,
            "type": "SIM",
            "start": self.start,
            "length": self.max_results,
        }

        last_error_msg: Union[str, None] = None
        for attempt in range(MAX_RETRIES):
            if attempt == 0:
                logger.info(
                    "Arthor API call started for '%s' (db=%s, start=%d, max_results=%d)",
                    truncated_smiles,
                    self.db_name,
                    self.start,
                    self.max_results,
                )
            elif last_error_msg:
                logger.warning(
                    "Retrying Arthor API for '%s' (attempt %d/%d). Last error: %s",
                    truncated_smiles,
                    attempt + 1,
                    MAX_RETRIES,
                    last_error_msg,
                )

            try:
                request_start = time.perf_counter()
                response = requests.get(url, params=params, timeout=self.timeout)
                response.raise_for_status()
                data = response.json()
                request_duration = time.perf_counter() - request_start

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

                logger.info(
                    "Arthor API success for '%s' (%d hits) in %.2fs",
                    truncated_smiles,
                    len(similar_smiles),
                    request_duration,
                )
                return similar_smiles[: self.max_results]

            except requests.exceptions.HTTPError as e:
                if hasattr(e, "response") and 400 <= e.response.status_code < 500:
                    logger.error(f"Client error for SMILES '{smiles[:30]}...': {e}")
                    return []
                status_code = (
                    e.response.status_code if hasattr(e, "response") and e.response else "?"
                )
                last_error_msg = f"HTTP {status_code}: {e}"
                logger.warning(
                    "Server error for SMILES '%s': %s", truncated_smiles, last_error_msg
                )
            except (
                requests.exceptions.ConnectionError,
                requests.exceptions.Timeout,
            ) as conn_exc:
                last_error_msg = f"{conn_exc.__class__.__name__}: {conn_exc}"
                logger.warning(
                    "Connectivity issue for SMILES '%s': %s",
                    truncated_smiles,
                    last_error_msg,
                )
            except Exception as unexpected_exc:  # pragma: no cover - defensive logging
                last_error_msg = f"{unexpected_exc.__class__.__name__}: {unexpected_exc}"
                logger.exception(
                    "Unexpected error for SMILES '%s': %s",
                    truncated_smiles,
                    last_error_msg,
                )

            if attempt < MAX_RETRIES - 1:
                delay = min(RETRY_DELAY * (2**attempt), MAX_BACKOFF)
                logger.info(
                    "Sleeping %.1fs before retrying Arthor for '%s'",
                    delay,
                    truncated_smiles,
                )
                time.sleep(delay)

        logger.error(
            "FAILED after %d retries for SMILES '%s'. Last error: %s",
            MAX_RETRIES,
            truncated_smiles,
            last_error_msg,
        )
        return []

    def search_similar(self, smiles_list: List[str]) -> Dict[str, List[str]]:
        results: Dict[str, List[str]] = {}
        for smiles in smiles_list:
            results[smiles] = self._search_single_smiles(smiles)
        return results


def search_similar_compounds(
    smiles: Union[str, List[str]],
    db_name: str = "REAL-Database-22Q1",
    max_results: int = 5,
    start: int = 0,
    base_url: str = "http://arthor.docking.org",
    timeout: int = 30,
) -> Dict[str, List[str]]:
    if isinstance(smiles, str):
        smiles_list = [smiles]
    elif isinstance(smiles, list):
        smiles_list = smiles
    else:
        logger.warning(f"Invalid SMILES input type: {type(smiles)}")
        return {}

    if not smiles_list:
        return {}

    client = ArthorClient(
        base_url=base_url,
        db_name=db_name,
        max_results=max_results,
        start=start,
        timeout=timeout,
    )
    return client.search_similar(smiles_list)


def search_similar_compounds_parallel(
    smiles_list: List[str],
    db_name: str = "REAL-Database-22Q1",
    max_results: int = 5,
    start: int = 0,
    base_url: str = "http://arthor.docking.org",
    max_workers: int = DEFAULT_PARALLEL_WORKERS,
    timeout: int = 30,
) -> Dict[str, List[str]]:
    if not smiles_list:
        return {}

    worker_count = min(max_workers or DEFAULT_PARALLEL_WORKERS, MAX_PARALLEL_WORKERS)
    client = ArthorClient(
        base_url=base_url,
        db_name=db_name,
        max_results=max_results,
        start=start,
        timeout=timeout,
    )
    results: Dict[str, List[str]] = {}

    def _run(smiles_str: str) -> Tuple[str, List[str]]:
        return smiles_str, client._search_single_smiles(smiles_str)

    if worker_count <= 1:
        for s in smiles_list:
            key, similar = _run(s)
            results[key] = similar
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_smiles = {executor.submit(_run, s): s for s in smiles_list}
        for future in concurrent.futures.as_completed(future_to_smiles):
            smiles_key = future_to_smiles[future]
            try:
                key, similar = future.result()
                results[key] = similar
            except Exception as exc:
                logger.warning(f"Failed to process SMILES '{smiles_key}': {exc}")
                results[smiles_key] = []

    return results


def _get_available_databases(base_url: str = "http://arthor.docking.org") -> List[Dict]:
    url = f"{base_url.rstrip('/')}/dt/data"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, list):
            return payload
        logger.warning(f"Unexpected Arthor response format: {type(payload)}")
        return []
    except Exception as exc:
        logger.warning(f"Failed to fetch available databases: {exc}")
        return []



