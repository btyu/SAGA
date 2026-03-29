from __future__ import annotations

from typing import List

from scileo_agent.core.data_models import Candidate, Objective
from .base import ObjectiveCombiner


class SimpleSumCombiner(ObjectiveCombiner):
    """
    Simple unweighted sum over all objectives.
    Strict on score-key presence; handles min/max direction by sign.
    """

    def combine(
        self, candidate: Candidate, objectives: List[Objective], weights: List[float]
    ) -> float:
        # Enforce that each objective's score key exists on the candidate
        if not self.verify_scores_present(candidate, objectives):
            return 0.0

        total = 0.0
        for obj in objectives:
            val = candidate.scores.get(obj.name)
            if val is None:
                # Treat None as extremely poor but remain finite
                val = -1e9 if obj.is_maximization else 1e9
            total += val if obj.is_maximization else -val
        return float(total)

    def aggregation_equation(
        self, objectives: List[Objective], weights: List[float]
    ) -> str:
        names = [getattr(o, 'name', 'obj') for o in objectives]
        # Note: sign flips for minimize are handled internally
        return "Score = " + " + ".join(names)


class SimpleProductCombiner(ObjectiveCombiner):
    """
    Simple unweighted product over all objectives.
    Strict on score-key presence; handles min/max direction by inverting minimize objectives.
    """

    def combine(self, candidate: Candidate, objectives: List[Objective], weights: List[float]) -> float:
        # Enforce that each objective's score key exists on the candidate
        if not self.verify_scores_present(candidate, objectives):
            return 0.0

        product = 1.0
        for obj in objectives:
            val = candidate.scores.get(obj.name)
            if val is None:
                # Treat None as extremely poor (near zero for product)
                val = 1e-9
            
            # Handle filter objectives specially: treat as gates (0 = fail, 1 = pass)
            if obj.type == "filter":
                # Convert boolean to float: True -> 1.0, False -> 0.0
                # For filters, False means failure, so multiply by 0 (or very small value)
                if isinstance(val, bool):
                    val = 1.0 if val else 0.0
                # If filter fails (val == 0), the product becomes 0 (candidate fails filter)
                product *= max(float(val), 1e-9)
            elif obj.is_maximization:
                product *= max(float(val), 1e-9)  # Avoid zero/negative products
            else:
                # Invert minimize objectives: smaller val -> larger contribution
                product *= (1.0 / max(float(val), 1e-9))
        return float(product)

    def aggregation_equation(self, objectives: List[Objective], weights: List[float]) -> str:
        names = [getattr(o, 'name', 'obj') for o in objectives]
        terms = []
        for name, obj in zip(names, objectives):
            if obj.type == "filter":
                terms.append(name)  # Filters are shown as-is (0 = fail, 1 = pass)
            elif obj.is_maximization:
                terms.append(name)
            else:
                terms.append(f"(1/{name})")
        return "Score = " + " × ".join(terms)


class WeightedSumCombiner(ObjectiveCombiner):
    """
    Weighted sum over all objectives using provided weights.
    Strict on score-key presence; handles min/max direction by sign.
    """

    def combine(self, candidate: Candidate, objectives: List[Objective], weights: List[float]) -> float:
        # Enforce that each objective's score key exists on the candidate
        if not self.verify_scores_present(candidate, objectives):
            return 0.0

        assert len(objectives) == len(weights), "Objectives and weights must match length"
        total = 0.0
        for w, obj in zip(weights, objectives):
            val = candidate.scores.get(obj.name)
            if val is None:
                val = -1e9 if obj.is_maximization else 1e9
            total += (w * val) if obj.is_maximization else (-w * val)
        return float(total)

    def aggregation_equation(self, objectives: List[Objective], weights: List[float]) -> str:
        names = [getattr(o, 'name', 'obj') for o in objectives]
        w = list(weights) if weights is not None else [1.0 for _ in names]
        # Display with direction-aware sign: +w_i*name for maximize, -w_i*name for minimize
        pieces = []
        for name, obj, wi in zip(names, objectives, w):
            sign = '+' if getattr(obj, 'is_maximization', True) else '-'
            pieces.append(f"{sign}{wi}*{name}")
        expr = " ".join(pieces).lstrip('+')
        return f"Score = {expr}"

