"""
Simplified SmallWorld API client that closely follows the official smallworld_api library pattern.

This implementation:
1. Uses sessions like the official client
2. Handles streaming responses properly
3. Throttles requests to avoid overwhelming the server
4. Processes queries sequentially (SmallWorld doesn't support parallelism)
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Union

import requests

logger = logging.getLogger(__name__)

# Configuration matching official client behavior
DEFAULT_BASE_URL = "https://sw.docking.org"
DEFAULT_DB_NAME = "REALDB-2025-07.smi.anon"
DEFAULT_TOP = 5
DEFAULT_TIMEOUT = 600  # 10 minutes like official client
DEFAULT_SCORES = "Atom Alignment,ECFP4,Daylight"

# Official client defaults
DEFAULT_SUBMIT_PARAMS = {
    "dist": 8,
    "tdn": 6,
    "rdn": 6,
    "rup": 2,
    "ldn": 2,
    "lup": 2,
    "maj": 6,
    "min": 6,
    "sub": 6,
    "sdist": 12,
    "tup": 6,
    "scores": DEFAULT_SCORES,
}

# DataTables columns from official client
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


def _clean_smiles(smiles: str) -> str:
    """Clean SMILES string like official client."""
    if not smiles:
        return ""
    smiles = smiles.strip()
    if " " in smiles:
        return smiles.split()[0].strip()
    return smiles


class SmallWorldClient:
    """Simplified SmallWorld client closely following official library pattern."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        db_name: str = DEFAULT_DB_NAME,
        top: int = DEFAULT_TOP,
        timeout: int = DEFAULT_TIMEOUT,
        scores: Optional[str] = DEFAULT_SCORES,
    ):
        self.base_url = base_url.rstrip("/")
        self.db_name = db_name
        self.top = max(1, int(top))
        self.timeout = timeout
        self.scores = scores or DEFAULT_SCORES

        # Use session like official client
        self.session = requests.Session()

        self._submit_params = dict(DEFAULT_SUBMIT_PARAMS)
        if self.scores:
            self._submit_params["scores"] = self.scores

        self._view_params = dict(VALID_EXPORT_COLUMNS)
        self._search_url = f"{self.base_url}/search/submit"
        self._view_url = f"{self.base_url}/search/view"

        # Track last request time for throttling
        self._last_request_time = 0.0

    def _throttle_request(self):
        """Throttle requests to avoid overwhelming the server (like official client)."""
        min_interval = 5.0  # 5 seconds between requests
        elapsed = time.time() - self._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_request_time = time.time()

    def search_similar(self, smiles_list: Sequence[str]) -> Dict[str, List[str]]:
        """Query SmallWorld for similar compounds (sequential only)."""
        results = {}

        for smiles in smiles_list:
            query = smiles.strip() if isinstance(smiles, str) else ""
            if not query:
                results[smiles] = []
                continue

            try:
                # Submit query (streaming response)
                params = dict(self._submit_params)
                params["smi"] = query
                params["db"] = self.db_name

                self._throttle_request()
                response = self.session.get(
                    self._search_url,
                    params=params,
                    stream=True,
                    timeout=self.timeout,
                )
                response.raise_for_status()

                # Parse streaming response for hlid
                hlid = None
                for line in response.iter_lines(decode_unicode=True):
                    line = line.strip()
                    if line.startswith("data:"):
                        try:
                            data = json.loads(line[5:].strip())  # Remove "data:" prefix
                            if "hlid" in data:
                                hlid = int(data["hlid"])
                            if data.get("status") == "END":
                                break
                        except json.JSONDecodeError:
                            continue
                response.close()

                if hlid is None:
                    logger.warning("No hlid received for SMILES: %s", query[:50])
                    results[smiles] = []
                    continue

                # Fetch results
                view_params = {
                    "hlid": hlid,
                    "start": 0,
                    "length": self.top,
                    "draw": 10,
                }
                view_params.update(self._view_params)

                response = requests.get(self._view_url, params=view_params, timeout=60)
                response.raise_for_status()
                payload = response.json()

                hits = []
                if payload.get("recordsTotal", 0) > 0:
                    for row in payload.get("data", []):
                        if isinstance(row, list) and row:
                            first = row[0]
                            if isinstance(first, dict):
                                hit_smiles = first.get("hitSmiles") or first.get(
                                    "smiles"
                                )
                                if hit_smiles:
                                    cleaned = _clean_smiles(hit_smiles)
                                    if cleaned:
                                        hits.append(cleaned)
                                        if len(hits) >= self.top:
                                            break

                results[smiles] = hits[: self.top]

            except Exception as exc:
                logger.warning("SmallWorld query failed for %s: %s", query[:50], exc)
                results[smiles] = []

        return results


def search_similar_compounds(
    smiles: Union[str, Sequence[str]],
    *,
    db_name: str = DEFAULT_DB_NAME,
    top: int = DEFAULT_TOP,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    scores: Optional[str] = DEFAULT_SCORES,
) -> Dict[str, List[str]]:
    """Convenience wrapper for SmallWorld queries."""
    if isinstance(smiles, str):
        smiles_list = [smiles]
    elif isinstance(smiles, Sequence):
        smiles_list = list(smiles)
    else:
        logger.warning("Invalid SMILES input type: %s", type(smiles))
        return {}

    client = SmallWorldClient(
        base_url=base_url,
        db_name=db_name,
        top=top,
        timeout=timeout,
        scores=scores,
    )
    return client.search_similar(smiles_list)


# Keep parallel function for API compatibility but force sequential
def search_similar_compounds_parallel(
    smiles_list: Sequence[str],
    *,
    db_name: str = DEFAULT_DB_NAME,
    top: int = DEFAULT_TOP,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    scores: Optional[str] = DEFAULT_SCORES,
    max_workers: int = 1,  # Force sequential
) -> Dict[str, List[str]]:
    """Sequential SmallWorld queries (parallel not supported by service)."""
    logger.info("SmallWorld queries are processed sequentially")
    return search_similar_compounds(
        smiles_list,
        db_name=db_name,
        top=top,
        base_url=base_url,
        timeout=timeout,
        scores=scores,
    )


def get_available_databases(base_url: str = DEFAULT_BASE_URL) -> List[Dict[str, Any]]:
    """Return metadata for available databases."""
    url = f"{base_url.rstrip('/')}/search/maps"
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        payload = response.json()
        if isinstance(payload, dict):
            return list(payload.values())
        if isinstance(payload, list):
            return payload
        logger.warning("Unexpected payload format: %s", type(payload))
        return []
    except Exception as exc:
        logger.warning("Failed to fetch databases: %s", exc)
        return []
