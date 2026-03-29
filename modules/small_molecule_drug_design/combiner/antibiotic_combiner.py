from __future__ import annotations
from typing import List
from math import isfinite
from scileo_agent.core.data_models import Candidate, Objective
from .base import ObjectiveCombiner


class AntibioticGeoMeanCombiner(ObjectiveCombiner):
    """
    Activity-only base with unified soft gates for all other constraints.

    Score =
        A^{w_A}
        * [g(S; 0.80, α_S)]^{λ_S}
        * [g(D; 0.80, α_D)]^{λ_D}
        * [g(N*;0.50, α_N)]^{λ_N}
        * [g(P; 0.50, α_bin)]^{λ_P}
        * [g(B; 0.50, α_bin)]^{λ_B}

    g(x; thr, α) = 1                      if x >= thr
                 = (x/thr)^α (clamped ≥0) if x <  thr

    Orientations:
      - If toxicity model returns safety, S = T; else S = 1 - T.
      - If novelty metric N is actually similarity (higher = less novel), use N* = 1 - N; otherwise N* = N.
    """

    # Objective names
    A = "staph_aureus_chemprop"
    T = "toxicity_safety_chemprop"  # may already be safety; see TOX_RETURNS_SAFETY
    R = "ra_score_xgb"
    D = "deepdl_druglikeness"
    N = "antibiotics_novelty"
    P = "pains_filter"
    B = "brenk_filter"

    # ---- Base weight (only activity used) -----------------------------------
    # Keep tuple for backward compatibility, but only wA (index 0) is used.
    WEIGHTS = (0.8, 0.0, 0.0, 0.0, 0.0
               )  # (A, S, R, D, N) — S/R/D/N ignored in base

    NOVELTY_IS_SIMILARITY = False  # set True if N is similarity (higher=less novel)
    TOX_RETURNS_SAFETY = True  # set False if T is toxicity prob (then S = 1 - T)

    # ---- Gate thresholds -----------------------------------------------------
    SAFETY_FLOOR = 0.80
    DRUGLIKE_FLOOR = 0.80
    NOVELTY_FLOOR = 0.60  # applied to N* (novelty-or-similarity flipped)
    BIN_THRESHOLD = 1.0  # for PAINS/Brenk, typically 0/1

    # ---- Gate sharpness (higher α => harsher below threshold) ---------------
    ALPHA_SAFETY = 4.0
    ALPHA_RA = 3.0
    ALPHA_DRUGLIKE = 2.0
    ALPHA_NOVELTY = 2.0
    ALPHA_BINARY = 4.0

    # ---- Gate strengths (exponents on each gate term) -----------------------
    LAMBDA_SAFETY = 1.2
    LAMBDA_RA = 0.8
    LAMBDA_DRUGLIKE = 1.0
    LAMBDA_NOVELTY = 0.5
    LAMBDA_PAINS = 0.30
    LAMBDA_BRENK = 0.20

    # -------------------------------------------------------------------------

    def required_objective_names(self):
        return {self.A, self.T, self.D, self.N}

    @staticmethod
    def _clamp01(v, default=0.0):
        try:
            v = float(v)
            if not isfinite(v):
                v = float(default)
        except Exception:
            v = float(default)
        return max(0.0, min(1.0, v))

    @staticmethod
    def _flag01(v, default=1.0):
        try:
            v = float(v)
        except Exception:
            v = float(default)
        return 1.0 if v >= 0.5 else 0.0

    def _g_soft(self, x: float, thr: float, alpha: float) -> float:
        x = max(0.0, min(1.0, float(x)))
        thr = max(1e-8, min(1.0, float(thr)))
        if x >= thr:
            return 1.0
        a = max(1.0, float(alpha))
        return max(0.0, (x / thr)**a)

    def combine(self, candidate: Candidate, objectives: List[Objective],
                weights: List[float]) -> float:
        self.validate_against_objectives(objectives)
        if not self.verify_scores_present(candidate, objectives):
            return 0.0

        # Only activity weight is used from WEIGHTS
        wA = self.WEIGHTS[0]

        # raw scores
        A = self._clamp01(candidate.scores.get(self.A), 0.0)
        T = self._clamp01(candidate.scores.get(self.T), 0.0)
        D = self._clamp01(candidate.scores.get(self.D), 0.0)
        N = self._clamp01(candidate.scores.get(self.N), 0.0)

        # orientation
        S = T if self.TOX_RETURNS_SAFETY else (1.0 - T)

        # novelty (flip if similarity)
        N_star = (1.0 - N) if self.NOVELTY_IS_SIMILARITY else N
        N_star = max(0.0, min(1.0, N_star))

        # optional filters (assume pass=1.0 if not requested)
        names_present = {o.name for o in objectives}
        if self.P in names_present:
            if self.P not in candidate.scores:
                raise ValueError(
                    f"Required score for optional filter '{self.P}' missing despite being requested."
                )
            P = self._flag01(candidate.scores.get(self.P), 1.0)
        else:
            P = 1.0
        if self.B in names_present:
            if self.B not in candidate.scores:
                raise ValueError(
                    f"Required score for optional filter '{self.B}' missing despite being requested."
                )
            B = self._flag01(candidate.scores.get(self.B), 1.0)
        else:
            B = 1.0

        # base: activity only
        base = A**wA

        # gates
        gS = self._g_soft(S, self.SAFETY_FLOOR, self.ALPHA_SAFETY)
        gD = self._g_soft(D, self.DRUGLIKE_FLOOR, self.ALPHA_DRUGLIKE)
        gN = self._g_soft(N_star, self.NOVELTY_FLOOR, self.ALPHA_NOVELTY)
        gP = self._g_soft(P, self.BIN_THRESHOLD, self.ALPHA_BINARY)
        gB = self._g_soft(B, self.BIN_THRESHOLD, self.ALPHA_BINARY)

        gate = ((gS**self.LAMBDA_SAFETY) * (gD**self.LAMBDA_DRUGLIKE) *
                (gN**self.LAMBDA_NOVELTY) * (gP**self.LAMBDA_PAINS) *
                (gB**self.LAMBDA_BRENK))

        return float(base * gate)

    def aggregation_equation(self, objectives: List[Objective],
                             weights: List[float]) -> str:
        return (
            "Score = (A^{0.45}) * "
            "[g(S;0.80,α_S)]^{1.2} * [g(D;0.80,α_D)]^{1.0} * "
            "[g(N*;0.50,α_N)]^{0.5} * [g(P;0.50,α_bin)]^{0.30} * [g(B;0.50,α_bin)]^{0.20}, "
            "where N* flips to (1-N) if N is similarity, and g(x;thr,α)=1 if x≥thr else (x/thr)^α."
        )
