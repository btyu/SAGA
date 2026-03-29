"""
Parent selection strategies for genetic algorithms.

Provides pluggable parent selection methods with different selection pressures.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

import numpy as np

from scileo_agent.core.data_models import Candidate, Objective, Population


class ParentSelector(ABC):
    """Abstract base class for parent selection strategies."""

    @abstractmethod
    def select(
        self, population: Population, objectives: List[Objective], k: int = 2
    ) -> List[Candidate]:
        """
        Select k parents from population.

        Args:
            population: Current population
            objectives: Optimization objectives
            k: Number of parents to select

        Returns:
            List of k selected parent candidates
        """
        pass


class RankBasedSelector(ParentSelector):
    """Rank-based selection with configurable percentile bins."""

    def __init__(
        self,
        compute_score_fn: Callable,
        bins: List[float] = None,
        weights: List[float] = None,
        seed: int = 42,
    ):
        """
        Initialize rank-based selector.

        Args:
            compute_score_fn: Function to compute candidate scores
            bins: Percentile bins (e.g., [0.1, 0.3, 0.6, 1.0])
            weights: Weights for each bin (e.g., [4.0, 2.0, 1.0, 0.5])
            seed: Random seed
        """
        self.compute_score_fn = compute_score_fn
        self.bins = bins or [0.1, 0.3, 0.6, 1.0]
        self.bin_weights = weights or [4.0, 2.0, 1.0, 0.5]
        np.random.seed(seed)

        # Validate configuration
        if len(self.bins) != len(self.bin_weights):
            raise ValueError("bins and weights must have same length")
        if self.bins[-1] < 1.0:
            raise ValueError("Last bin must be >= 1.0")

    def select(
        self, population: Population, objectives: List[Objective], k: int = 2
    ) -> List[Candidate]:
        """Select k unique parents using rank-based probabilities."""
        num_candidates = len(population.candidates)

        if num_candidates == 0 or k <= 0:
            return []

        if k > num_candidates:
            k = num_candidates

        # Compute aggregate scores for ranking
        raw_scores = [
            self.compute_score_fn(candidate, objectives)
            for candidate in population.candidates
        ]

        # Rank candidates (0 = best)
        sorted_indices_desc = np.argsort(-np.array(raw_scores))
        ranks = np.empty(num_candidates, dtype=int)
        ranks[sorted_indices_desc] = np.arange(num_candidates)

        # Convert ranks to percentiles
        percentiles = ranks / float(num_candidates)

        # Assign weights based on percentile bins
        selection_weights = np.zeros(num_candidates, dtype=float)
        for i, p in enumerate(percentiles):
            for j, threshold in enumerate(self.bins):
                if p <= threshold:
                    selection_weights[i] = float(self.bin_weights[j])
                    break

        # Ensure no zero weights
        selection_weights[selection_weights <= 0] = 1e-9

        # Normalize to probabilities
        total_w = float(np.sum(selection_weights))
        probabilities = (
            selection_weights / total_w
            if total_w > 0
            else np.full(num_candidates, 1.0 / num_candidates)
        )

        # Weighted sampling without replacement
        indices = np.random.choice(
            num_candidates, size=k, replace=False, p=probabilities
        )

        return [population.candidates[i] for i in indices]


class TournamentSelector(ParentSelector):
    """Tournament selection."""

    def __init__(
        self, compute_score_fn: Callable, tournament_size: int = 3, seed: int = 42
    ):
        """
        Initialize tournament selector.

        Args:
            compute_score_fn: Function to compute candidate scores
            tournament_size: Number of candidates per tournament
            seed: Random seed
        """
        self.compute_score_fn = compute_score_fn
        self.tournament_size = tournament_size
        np.random.seed(seed)

    def select(
        self, population: Population, objectives: List[Objective], k: int = 2
    ) -> List[Candidate]:
        """Select k parents using tournament selection."""
        selected = []

        for _ in range(k):
            # Randomly select tournament_size candidates
            tournament_indices = np.random.choice(
                len(population.candidates),
                size=min(self.tournament_size, len(population.candidates)),
                replace=False,
            )

            # Find best in tournament
            best_idx = None
            best_score = float("-inf")

            for idx in tournament_indices:
                candidate = population.candidates[idx]
                score = self.compute_score_fn(candidate, objectives)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(population.candidates[best_idx])

        return selected


class RouletteWheelSelector(ParentSelector):
    """Fitness-proportional (roulette wheel) selection."""

    def __init__(self, compute_score_fn: Callable, seed: int = 42):
        """
        Initialize roulette wheel selector.

        Args:
            compute_score_fn: Function to compute candidate scores
            seed: Random seed
        """
        self.compute_score_fn = compute_score_fn
        np.random.seed(seed)

    def select(
        self, population: Population, objectives: List[Objective], k: int = 2
    ) -> List[Candidate]:
        """Select k parents using fitness-proportional selection."""
        # Compute scores
        scores = np.array(
            [
                self.compute_score_fn(candidate, objectives)
                for candidate in population.candidates
            ]
        )

        # Shift to make all positive if needed
        min_score = np.min(scores)
        if min_score < 0:
            scores = scores - min_score + 1e-9

        # Normalize to probabilities
        total_score = np.sum(scores)
        probabilities = scores / total_score if total_score > 0 else np.ones(len(scores)) / len(scores)

        # Sample with replacement (allow same parent twice)
        indices = np.random.choice(
            len(population.candidates), size=k, replace=True, p=probabilities
        )

        return [population.candidates[i] for i in indices]
