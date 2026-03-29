from abc import abstractmethod
from typing import List, Optional, Dict, Any
import inspect

from ..data_models import Population, Objective, Candidate
from .base import BaseModule


class OptimizerModule(BaseModule):
    """
    Module that runs optimization algorithms until convergence.
    
    Given a well-defined optimization problem (objectives with scorers) and
    initial population, it runs optimization algorithms until convergence
    and returns the optimized candidates.
    """

    @property
    def requires_initial_population(self) -> bool:
        return True

    def check_objectives(self, objectives: List[Objective]):
        """Check if the given objectives cover all the required objectives, or are compatible with the optimizer. Raise an error if not."""
        pass

    @abstractmethod
    async def create_random_candidates(
        self,
        num_candidates: int,
        **additional_kwargs: Dict[str, Any]
    ) -> List[Candidate]:
        """
        Create random candidates.
        
        Args:
            num_candidates: Number of candidates to create
            
        Returns:
            List of randomly created candidates
        """
        pass

    @abstractmethod
    async def optimize(
        self,
        current_population: Optional[Population],
        objectives: List[Objective],
        **additional_kwargs: Dict[str, Any]
    ) -> Population:
        """
        Run optimization on the given population.
        
        Args:
            current_population: Current population of candidates, if None, this optimizer should create an initial population, usually in the first iteration
            objectives: List of objectives with scorer functions
            
        Returns:
            Population of optimized candidates with their scores
        """
        pass

    async def evaluate_population(
        self,
        population: Population,
        objectives: List[Objective],
        force_evaluation: bool = False
    ) -> Population:
        """
        Evaluate candidates based on objectives.
        
        Args:
            population: Population of candidates to evaluate
            objectives: List of objectives with scorer functions
            force_evaluation: Whether to force evaluation of candidates, otherwise skip already evaluated candidates/population

        Returns:
            Population of candidates with their scores
        """
        if population is None or population.is_empty:
            raise ValueError("Population is None or empty for evaluation.")

        for objective in objectives:
            population = await objective.score(population, force_evaluation=force_evaluation)
        return population

    async def process(
        self,
        current_population: Optional[Population],
        objectives: List[Objective],
        **additional_kwargs: Dict[str, Any]
    ) -> Population:
        """Process method wrapper for optimize."""
        self._increment_call_count()
        self.check_objectives(objectives)

        if self.requires_initial_population and (current_population is None or current_population.is_empty):
            raise ValueError(
                "Initial population is required for this optimizer.")

        # Inspect the concrete optimize signature
        signature = inspect.signature(self.optimize)
        parameters = signature.parameters
        accepts_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values())

        if accepts_var_kwargs:
            # If the optimize method accepts var-keyword arguments, pass all additional kwargs
            population = await self.optimize(current_population, objectives, **additional_kwargs)
        else:
            # Otherwise, only pass the basic arguments
            population = await self.optimize(current_population, objectives)

        await self.evaluate_population(population, objectives)
        return population
