"""
Population data model for optimization frameworks.

This module defines data structures for managing collections of optimization candidates
with evolutionary and statistical analysis capabilities.
"""

from typing import List, Dict, Any, Optional, Tuple, Union
from pydantic import BaseModel, Field
import numpy as np


from .candidate import Candidate

from ...utils.logging import get_logger

logger = get_logger()


class Population(BaseModel):
    """
    Represents a population of optimization candidates with analysis capabilities.
    
    This class manages collections of candidates and provides methods for
    statistical analysis, selection, and evolutionary operations.
    """
    
    # Core population data
    candidates: List[Candidate] = Field(default_factory=list, description="List of candidates in the population")
    generation: Optional[int] = Field(default=None, description="Current generation number")
    
    # Properties and scores
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the candidate")
    scores: Dict[str, Optional[float]] = Field(default_factory=dict, description="Population-wise evaluation scores (only for population-wise objectives)")
    regular_scores_mean: Dict[str, Optional[float]] = Field(default_factory=dict, description="Average scores of the candidates")
    regular_scores_std: Dict[str, Optional[float]] = Field(default_factory=dict, description="Standard deviation of the average scores of the candidates")
    regular_scores_none_count: Dict[str, int] = Field(default_factory=dict, description="Number of candidates with incomplete evaluations")
    @property
    def size(self) -> int:
        """Get the current population size."""
        return len(self.candidates)
    
    @property
    def is_empty(self) -> bool:
        """Check if the population is empty."""
        return len(self.candidates) == 0
    
    def __len__(self) -> int:
        """Get the population size."""
        return len(self.candidates)
    
    def add_candidate(self, candidate: Candidate) -> None:
        """Add a candidate to the population."""
        self.candidates.append(candidate)
    
    def add_candidates(self, candidates: List[Candidate]) -> None:
        """Add multiple candidates to the population."""
        for candidate in candidates:
            self.add_candidate(candidate)
    
    def get_candidate(self, candidate_id: str) -> Optional[Candidate]:
        """Get a candidate by ID."""
        for candidate in self.candidates:
            if candidate.id == candidate_id:
                return candidate
        return None
    
    def get_score(self, objective_name: str) -> float:
        return self.scores[objective_name]
    
    def set_score(self, objective_name: str, score: float) -> None:
        self.scores[objective_name] = score

    def get_property(self, prop_name: str) -> Any:
        """Get a specific property by name."""
        return self.properties[prop_name]
    
    def set_property(self, prop_name: str, value: Any) -> None:
        """Set a specific property by name."""
        self.properties[prop_name] = value

    def get_evaluated_objectives(self) -> Dict[str, List[str]]:
        """Get the objectives that have been evaluated."""
        population_objectives = [o for o in self.scores.keys()]
        regular_objectives = [o for o in self.candidates[0].scores.keys()]
        return {
            "regular": regular_objectives,
            "population_wise": population_objectives,
        }

    def _calculate_regular_score_mean_and_std(self, objective_name: str) -> Tuple[Optional[float], Optional[float], int]:
        """Calculate the average score and standard deviation of the candidates for a given objective.

        Only supports candidate-wise objectives with numerical scores. Raises ValueError for filter or population-wise objectives.

        Args:
            objective_name: Name of the objective

        Returns:
            Tuple of (mean, std, none_count)

        Raises:
            ValueError: If called on a filter or population-wise objective, or if scores contain non-numerical values
        """
        scores = [c.get_score(objective_name) for c in self.candidates]
        none_count = 0
        tmp_scores = []
        for score in scores:
            if score is None:
                none_count += 1
            else:
                # Check if score is bool (filter objective) and raise error
                if isinstance(score, bool):
                    raise ValueError(
                        f"Cannot calculate mean and std for objective '{objective_name}': "
                        "This objective appears to be a filter objective (returns bool values). "
                        "Mean and std calculation only supports candidate-wise objectives with numerical scores."
                    )
                tmp_scores.append(score)
        if none_count == len(scores):
            return None, None, none_count
        return float(np.mean(tmp_scores)), float(np.std(tmp_scores)), none_count

    def get_regular_score_mean_and_std(self, objective_name: str) -> Tuple[Optional[float], Optional[float], int]:
        """Get the average score and standard deviation of the candidates for a given objective.

        Only supports candidate-wise objectives with numerical scores.

        Args:
            objective_name: Name of the objective

        Returns:
            Tuple of (mean, std, none_count)

        Raises:
            ValueError: If called on a filter or population-wise objective
        """
        if objective_name not in self.regular_scores_mean or objective_name not in self.regular_scores_std or objective_name not in self.regular_scores_none_count:
            mean, std, none_count = self._calculate_regular_score_mean_and_std(objective_name)
            self.regular_scores_mean[objective_name] = mean
            self.regular_scores_std[objective_name] = std
            self.regular_scores_none_count[objective_name] = none_count
        return self.regular_scores_mean[objective_name], self.regular_scores_std[objective_name], self.regular_scores_none_count[objective_name]
    
    def __get_candidates_and_scores(self, objective) -> Tuple[List[Candidate], List[Union[float, bool]]]:
        if objective.population_wise:
            raise ValueError("Cannot find best candidate for population-wise objective")
        if objective.type == "filter":
            raise ValueError("Cannot find best candidate for filter objective (filter objectives return bool, not numerical scores)")
        if self.is_empty:
            raise ValueError("Cannot find best candidate for empty population")

        if objective.name not in self.regular_scores_mean or objective.name not in self.regular_scores_std or objective.name not in self.regular_scores_none_count:
            self.get_regular_score_mean_and_std(objective.name)
        if self.regular_scores_none_count[objective.name] == self.size:
            raise ValueError(f"Cannot find best candidate for objective {objective.name} because evaluation was incomplete for all candidates")

        candidates_with_scores = [c for c in self.candidates if c.get_score(objective.name) is not None]
        scores_with_scores = [c.get_score(objective.name) for c in candidates_with_scores]

        return candidates_with_scores, scores_with_scores

    def find_best_candidate(self, objective) -> Tuple[Optional[Candidate], Optional[float], Optional[str]]:
        """Find the best candidate for a given objective."""
        try:
            candidates_with_scores, scores_with_scores = self.__get_candidates_and_scores(objective)
        except ValueError as e:
            logger.warning(f"Cannot find best candidate for objective {objective.name}: {e}. Returning None.")
            return None, None, None
        direction = objective.optimization_direction
        if direction == "maximize":
            best_candidate = candidates_with_scores[np.argmax(scores_with_scores)]
            return best_candidate, np.max(scores_with_scores), "maxima"
        else:
            best_candidate = candidates_with_scores[np.argmin(scores_with_scores)]
            return best_candidate, np.min(scores_with_scores), "minima"
        
    def find_worst_candidate(self, objective) -> Tuple[Optional[Candidate], Optional[float], Optional[str]]:
        """Find the worst candidate for a given objective."""
        try:
            candidates_with_scores, scores_with_scores = self.__get_candidates_and_scores(objective)
        except ValueError as e:
            logger.warning(f"Cannot find worst candidate for objective {objective.name}: {e}. Returning None.")
            return None, None, None
        direction = objective.optimization_direction
        if direction == "maximize":
            worst_candidate = candidates_with_scores[np.argmin(scores_with_scores)]
            return worst_candidate, np.min(scores_with_scores), "minima"
        else:
            worst_candidate = candidates_with_scores[np.argmax(scores_with_scores)]
            return worst_candidate, np.max(scores_with_scores), "maxima"
        
    def remove_candidate(self, candidate_id: str) -> bool:
        """Remove a candidate by ID. Returns True if removed, False if not found."""
        for i, candidate in enumerate(self.candidates):
            if candidate.id == candidate_id:
                self.candidates.pop(i)
                return True
        return False

    async def filter(self, filter_objective) -> "Population":
        """
        Filter the population using a filter objective and return a new population with passing candidates.

        Args:
            filter_objective: An Objective with type="filter" that returns bool scores (True=pass, False=fail)

        Returns:
            A new Population containing only candidates that passed the filter (score=True)

        Raises:
            ValueError: If the objective is not a filter type objective
        """
        # Validate that this is a filter objective
        if filter_objective.type != "filter":
            raise ValueError(
                f"Cannot use filter() with objective '{filter_objective.name}': "
                f"Expected type='filter', but got type='{filter_objective.type}'. "
                "Only filter objectives (returning bool values) can be used with this method."
            )

        if self.is_empty:
            return Population(candidates=[])

        # Evaluate the population with the filter objective if not already evaluated
        await filter_objective.score(self)

        # Filter candidates where the filter objective score is True
        passing_candidates = []
        for candidate in self.candidates:
            score = candidate.get_score(filter_objective.name)
            if score is True:
                passing_candidates.append(candidate)
            elif score is None:
                logger.warning(
                    f"Candidate {candidate.id[:8]}... has None score for filter objective '{filter_objective.name}'. "
                    "Excluding from filtered population."
                )

        # Create new population with passing candidates
        filtered_population = Population(
            candidates=passing_candidates
        )

        logger.debug(
            f"Filtered population using '{filter_objective.name}': "
            f"{len(passing_candidates)}/{len(self.candidates)} candidates passed"
        )

        return filtered_population

    async def evaluate(self, objectives: list) -> "Population":
        """Evaluate the population for a given objective (async)."""
        for objective in objectives:
            await objective.score(self)
        return self
    
    def get_evaluated_candidates(self) -> List[Candidate]:
        """Get all candidates that have been evaluated."""
        raise NotImplementedError("This method is not implemented yet")
    
    def get_unevaluated_candidates(self) -> List[Candidate]:
        """Get all candidates that have not been evaluated."""
        raise NotImplementedError("This method is not implemented yet")
    
    async def get_pareto_set(self, objectives: List) -> "Population":
        """
        Return a Population containing only those candidates on the Pareto frontier (async).

        Only regular objectives (which score each candidate) are used to calculate
        the Pareto frontier. Population-wise objectives are ignored with a warning.

        Args:
            objectives: List of objectives to consider for Pareto frontier calculation.

        Returns:
            A new Population containing only the Pareto-optimal candidates.
        """
        if self.is_empty:
            return Population(candidates=[])

        if len(objectives) == 0:
            raise ValueError("No objectives provided for Pareto frontier calculation.")

        # Filter objectives to only include candidate-wise objectives with optimization direction
        # Exclude population-wise and filter objectives
        population_wise_objectives = [obj for obj in objectives if obj.population_wise]
        filter_objectives = [obj for obj in objectives if obj.type == "filter"]

        if population_wise_objectives:
            logger.warning(
                f"Ignoring {len(population_wise_objectives)} population-wise objectives "
                f"({[obj.name for obj in population_wise_objectives]}) in Pareto frontier calculation. "
                "Only candidate-wise objectives with optimization direction are used."
            )

        if filter_objectives:
            logger.warning(
                f"Ignoring {len(filter_objectives)} filter objectives "
                f"({[obj.name for obj in filter_objectives]}) in Pareto frontier calculation. "
                "Only candidate-wise objectives with optimization direction are used."
            )

        regular_objectives = [obj for obj in objectives if obj.type == "candidate-wise"]
        objective_names = [obj.name for obj in regular_objectives]

        if not objective_names:
            logger.warning("No regular objectives found for Pareto frontier calculation. Returning original population.")
            return Population(candidates=self.candidates.copy())

        # Ensure all candidates have scores for all objectives
        await self.evaluate(regular_objectives)
        valid_candidates = self.candidates
        
        # Create objective mapping for optimization direction
        objective_directions = {}
        for obj in regular_objectives:
            objective_directions[obj.name] = obj.optimization_direction
        
        # Calculate Pareto frontier
        pareto_candidates = []
        
        for i, candidate in enumerate(valid_candidates):
            is_pareto_optimal = True
            
            # Compare with all other candidates
            for j, other_candidate in enumerate(valid_candidates):
                if i == j:
                    continue
                
                # Check if other_candidate dominates candidate
                dominates = True
                at_least_one_better = False
                
                for obj_name in objective_names:
                    candidate_score = candidate.scores[obj_name]
                    other_score = other_candidate.scores[obj_name]
                    direction = objective_directions[obj_name]
                    
                    # Skip comparison if either score is None
                    if candidate_score is None or other_score is None:
                        dominates = False
                        logger.warning(f"Skipping comparison for candidate {candidate.id} and objective {obj_name} because one of the scores is None")
                        break
                    
                    if direction == "maximize":
                        if other_score < candidate_score:
                            dominates = False
                            break
                        elif other_score > candidate_score:
                            at_least_one_better = True
                    else:  # minimize
                        if other_score > candidate_score:
                            dominates = False
                            break
                        elif other_score < candidate_score:
                            at_least_one_better = True
                
                # If other_candidate dominates candidate, then candidate is not Pareto optimal
                if dominates and at_least_one_better:
                    is_pareto_optimal = False
                    break
            
            if is_pareto_optimal:
                pareto_candidates.append(candidate)
        
        # Create new population with Pareto-optimal candidates
        pareto_population = Population(
            candidates=pareto_candidates
        )
        
        return pareto_population
    
    def clear(self) -> None:
        """Clear all candidates from the population."""
        raise NotImplementedError("This method is not implemented. Please re-instantiate a new population.")
    
    def __len__(self) -> int:
        """Get the population size."""
        return len(self.candidates)
    
    def __iter__(self):
        """Iterate over candidates."""
        return iter(self.candidates)
    
    def __str__(self) -> str:
        """String representation of the population."""
        return f"Population(size={self.size}, generation={self.generation})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Population(size={self.size}, generation={self.generation})" 
