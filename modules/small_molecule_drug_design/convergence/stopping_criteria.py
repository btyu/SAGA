"""
Stopping criteria for genetic algorithm convergence.

Provides reusable stopping criteria for optimization loops.
"""

from abc import ABC, abstractmethod
from typing import Callable, List

from scileo_agent.core.data_models import Objective, Population


class StoppingCriterion(ABC):
    """Abstract base class for stopping criteria."""

    @abstractmethod
    def should_stop(self, population: Population, objectives: List[Objective]) -> bool:
        """
        Check if optimization should stop.

        Args:
            population: Current population
            objectives: Optimization objectives

        Returns:
            True if stopping criterion is met
        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the criterion state."""
        pass

    @abstractmethod
    def get_state(self) -> dict:
        """Get current state for logging."""
        pass


class MultiMetricStopping(StoppingCriterion):
    """Stop when multiple metrics plateau (no improvement)."""

    def __init__(
        self,
        compute_score_fn: Callable,
        metrics: List[str] = None,
        threshold: float = 1e-3,
        patience: int = 10,
    ):
        """
        Initialize multi-metric stopping.

        Args:
            compute_score_fn: Function to compute candidate scores
            metrics: List of metric names to track (default: top1, top10_mean, top100_mean)
            threshold: Minimum improvement threshold
            patience: Generations without improvement before stopping
        """
        self.compute_score_fn = compute_score_fn
        self.metrics = metrics or ["top1", "top10_mean", "top100_mean"]
        self.threshold = threshold
        self.patience = patience

        self.best_scores = {metric: float("-inf") for metric in self.metrics}
        self.generations_without_improvement = 0

    def should_stop(self, population: Population, objectives: List[Objective]) -> bool:
        """Check if any metric has improved."""
        current_scores = self._compute_current_scores(population, objectives)

        # Check if any metric improved
        improved = False
        for metric in self.metrics:
            if current_scores[metric] > self.best_scores[metric] + self.threshold:
                improved = True
                self.best_scores[metric] = max(
                    self.best_scores[metric], current_scores[metric]
                )

        if improved:
            self.generations_without_improvement = 0
            return False
        else:
            self.generations_without_improvement += 1
            return self.generations_without_improvement >= self.patience

    def _compute_current_scores(
        self, population: Population, objectives: List[Objective]
    ) -> dict:
        """Compute current metric scores."""
        if not population.candidates:
            return {metric: 0.0 for metric in self.metrics}

        scores = [
            self.compute_score_fn(c, objectives) for c in population.candidates
        ]
        sorted_scores = sorted(scores, reverse=True)

        result = {}

        if "top1" in self.metrics:
            result["top1"] = sorted_scores[0] if sorted_scores else 0.0

        if "top10_mean" in self.metrics:
            top_10 = sorted_scores[: min(10, len(sorted_scores))]
            result["top10_mean"] = sum(top_10) / len(top_10) if top_10 else 0.0

        if "top100_mean" in self.metrics:
            top_100 = sorted_scores[: min(100, len(sorted_scores))]
            result["top100_mean"] = sum(top_100) / len(top_100) if top_100 else 0.0

        return result

    def reset(self):
        """Reset tracking state."""
        self.best_scores = {metric: float("-inf") for metric in self.metrics}
        self.generations_without_improvement = 0

    def get_state(self) -> dict:
        """Get current state."""
        return {
            "best_scores": self.best_scores.copy(),
            "generations_without_improvement": self.generations_without_improvement,
        }


class BudgetStopping(StoppingCriterion):
    """Stop when evaluation budget is exhausted."""

    def __init__(self, budget: int):
        """
        Initialize budget stopping.

        Args:
            budget: Maximum number of evaluations
        """
        self.budget = budget
        self.evaluations_used = 0

    def should_stop(self, population: Population, objectives: List[Objective]) -> bool:
        """Check if budget is exhausted."""
        return self.evaluations_used >= self.budget

    def update_budget(self, evaluations: int):
        """Update budget usage."""
        self.evaluations_used += evaluations

    def reset(self):
        """Reset budget tracking."""
        self.evaluations_used = 0

    def get_state(self) -> dict:
        """Get current state."""
        return {
            "budget": self.budget,
            "evaluations_used": self.evaluations_used,
            "remaining": self.budget - self.evaluations_used,
        }


class CompositeStopping(StoppingCriterion):
    """Combine multiple stopping criteria with AND/OR logic."""

    def __init__(self, criteria: List[StoppingCriterion], mode: str = "OR"):
        """
        Initialize composite stopping.

        Args:
            criteria: List of stopping criteria to combine
            mode: "OR" (stop if any met) or "AND" (stop if all met)
        """
        self.criteria = criteria
        self.mode = mode.upper()

        if self.mode not in ["OR", "AND"]:
            raise ValueError("mode must be 'OR' or 'AND'")

    def should_stop(self, population: Population, objectives: List[Objective]) -> bool:
        """Check composite stopping condition."""
        results = [criterion.should_stop(population, objectives) for criterion in self.criteria]

        if self.mode == "OR":
            return any(results)
        else:  # AND
            return all(results)

    def reset(self):
        """Reset all criteria."""
        for criterion in self.criteria:
            criterion.reset()

    def get_state(self) -> dict:
        """Get state of all criteria."""
        return {
            "mode": self.mode,
            "criteria_states": [
                {
                    "type": type(criterion).__name__,
                    "state": criterion.get_state(),
                }
                for criterion in self.criteria
            ],
        }
