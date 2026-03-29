from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Set

from scileo_agent.core.data_models import Candidate, Objective


class ObjectiveCombiner:
    """
    Base class for aggregating multiple objective scores into one scalar.

    Contract:
    - required_objective_names(): objective names that MUST be present for this combiner.
    - validate_against_objectives(objectives): raise if required names are absent.
    - verify_scores_present(candidate, objectives): raise if any required score key is absent in candidate.scores.
      Note: None or 0.0 values are OK; key must exist (scorer was called).
    - combine(candidate, objectives, weights) -> float (higher is better).
    """

    def __init__(self, params: Optional[Dict[str, Any]] = None) -> None:
        self.params = params or {}

    def required_objective_names(self) -> Set[str]:
        """Names of objectives that MUST be present for this combiner."""
        return set()

    def validate_against_objectives(self, objectives: List[Objective]) -> None:
        """Verify that all required objectives exist in the provided list."""
        need = self.required_objective_names()
        if not need:
            return
        names = {o.name for o in objectives}
        missing = sorted([n for n in need if n not in names])
        if missing:
            raise ValueError(
                f"Combiner requires objectives not provided: {missing}. Available: {sorted(names)}"
            )

    def _score_keys_required(self, objectives: List[Objective]) -> Iterable[str]:
        """
        Score keys that MUST exist on candidate.scores for this combiner to run.
        By default equal to required_objective_names(); if empty, require all objectives.
        """
        need = self.required_objective_names()
        return list(need) if need else [o.name for o in objectives]

    def verify_scores_present(self, candidate: Candidate, objectives: List[Objective]) -> bool:
        """
        Raise if any required score key is absent in candidate.scores.
        Value may be None/0.0; the key must exist (scorer ran).
        """
        required = list(self._score_keys_required(objectives))
        missing = [k for k in required if k not in candidate.scores]
        if missing:
            print(
                "Required scores not computed for candidate (missing keys in candidate.scores): "
                f"{missing}. Ensure these scorers were executed before aggregation."
            )
            return False
        return True

    def combine(self, candidate: Candidate, objectives: List[Objective], weights: List[float]) -> float:
        """Return a single scalar score (higher is better). Must be implemented by subclasses."""
        raise NotImplementedError

    def aggregation_equation(self, objectives: List[Objective], weights: List[float]) -> str:
        """
        Human-readable string describing how objective scores are aggregated into a single score.
        Subclasses should override to provide a concise equation description.
        """
        return f"{self.__class__.__name__} aggregation over objectives"



