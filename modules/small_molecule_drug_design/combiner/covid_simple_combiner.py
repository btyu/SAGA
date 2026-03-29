from __future__ import annotations

from typing import List

from scileo_agent.core.data_models import Candidate, Objective
from .base import ObjectiveCombiner


class CovidSimpleCombiner(ObjectiveCombiner):
    """
    Simple weighted-sum for COVID Mpro optimization with binary gates.

    Main objectives (higher is better, all assumed in [0,1]):
      - mpro_unidock (most important)
      - deepdl_druglikeness
      - mpro_his161_a (H)
      - mpro_glu164_a (G)
      - mpro_his39_a (S)

    Gate objectives (optional; if present in objectives, they must pass):
      - pains_filter (1 pass, 0 fail)
      - brenk_filter (0.5 if exactly one alert, else 0)

    Base = (0.6*M + 0.2*D)
    Multiplier from H/G/S (treating them as binary via int()):
        - H == 1: mult += 0.20
        - G == 1: mult += 0.20
        - S == 1: mult += 0.05
    Final score = Base * mult * P_gate * B_gate, where P/B default to 1.0 unless requested.
    """

    M = "mpro_unidock"
    D = "deepdl_druglikeness"
    P = "pains_filter"
    B = "brenk_filter"
    H = "mpro_his161_a"
    G = "mpro_glu164_a"
    S = "mpro_his39_a"

    WEIGHTS = (0.6, 0.2)  # (M, D)

    def required_objective_names(self):
        return {self.M, self.D, self.H, self.G, self.S}

    @staticmethod
    def _clamp01(v, default=0.0):
        try:
            x = float(v)
        except (TypeError, ValueError):
            x = float(default)
        if x != x or x == float("inf") or x == float("-inf"):
            x = float(default)
        return max(0.0, min(1.0, x))

    def combine(self, candidate: Candidate, objectives: List[Objective],
                weights: List[float]) -> float:
        self.validate_against_objectives(objectives)
        if not self.verify_scores_present(candidate, objectives):
            return 0.0

        wM, wD = self.WEIGHTS

        M = self._clamp01(candidate.scores.get(self.M), 0.0)
        D = self._clamp01(candidate.scores.get(self.D), 0.0)
        H = self._clamp01(candidate.scores.get(self.H), 0.0)
        G = self._clamp01(candidate.scores.get(self.G), 0.0)
        S = self._clamp01(candidate.scores.get(self.S), 0.0)

        base = (wM * M) + (wD * D)

        mult = 1.00
        if H == 1:
            mult += 0.20
        if G == 1:
            mult += 0.20
        if S == 1:
            mult += 0.05

        # Optional gates
        names_present = {o.name for o in objectives}
        if self.P in names_present:
            if self.P not in candidate.scores:
                raise ValueError(
                    f"Required score for optional filter '{self.P}' missing despite being requested."
                )
            P = self._clamp01(candidate.scores.get(self.P), 0.0)
        else:
            P = 1.0

        if self.B in names_present:
            if self.B not in candidate.scores:
                raise ValueError(
                    f"Required score for optional filter '{self.B}' missing despite being requested."
                )
            B = self._clamp01(candidate.scores.get(self.B), 0.0)
        else:
            B = 1.0

        gated = base * mult * P * B
        return float(gated)

    def aggregation_equation(self, objectives: List[Objective],
                             weights: List[float]) -> str:
        return (
            "Score = (0.6*M + 0.2*D) * mult(H,G,S) * P * B, where mult is: "
            "H==1->mult+=0.20, G==1->mult+=0.20, S==1->mult+=0.05")
