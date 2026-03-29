"""
Objective data model for optimization frameworks.

This module defines the Objective class that represents an optimization objective
with name, description, and optional scorer function.
"""

import inspect
from typing import Optional, Callable, Any, Dict, List, Union, Literal
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
import uuid

from .candidate import Candidate
from .population import Population

class Objective(BaseModel):
    """
    Represents an optimization objective with name, description, and optional scorer.

    An objective defines what should be optimized, including a human-readable name,
    description of what it measures, and an optional callable scorer function.
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True  # Allow callable types for scorer in Pydantic
    )

    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str = Field(..., description="Name of the objective (e.g., 'docking_score', 'drug_likeness')")
    description: str = Field(..., description="Human-readable description of what this objective measures")
    type: Literal["candidate-wise", "population-wise", "filter"] = Field(
        default="candidate-wise",
        description="Type of objective: 'candidate-wise' evaluates individual candidates, 'population-wise' evaluates entire population, 'filter' filters out invalid candidates"
    )
    
    # Scorer function (not serialized - needs to be set programmatically)
    scorer: Optional[Callable[[List[Candidate]], Union[List[Optional[Union[float, bool]]], Optional[float]]]] = Field(default=None, exclude=True, description="Callable function that scores candidates in batch")
    
    # Objective metadata
    optimization_direction: Optional[str] = Field(default=None, description="Whether to 'maximize' or 'minimize' this objective")
    weight: Optional[float] = Field(default=None, description="Weight for multi-objective optimization")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata for the objective")

    def __init__(self, **data):
        """
        Initialize Objective with support for legacy population_wise and filter parameters.

        Accepts either:
        - type: str (directly specifying "candidate-wise", "population-wise", or "filter")
        - population_wise: bool and/or filter: bool (legacy parameters)

        Raises:
            ValueError: If population_wise and filter are both True, or if type conflicts with legacy parameters
        """
        # Extract legacy parameters if present
        population_wise = data.pop('population_wise', None)
        filter_flag = data.pop('filter', None)
        provided_type = data.get('type', None)

        # Validation: population_wise and filter cannot both be True
        if population_wise is True and filter_flag is True:
            raise ValueError(
                "Invalid objective configuration: 'population_wise' and 'filter' cannot both be True. "
                "An objective can only be one of: candidate-wise, population-wise, or filter."
            )

        # Determine the expected type from legacy parameters
        expected_type_from_legacy = None
        if filter_flag is True:
            expected_type_from_legacy = 'filter'
        elif population_wise is True:
            expected_type_from_legacy = 'population-wise'
        elif population_wise is False and filter_flag is False:
            expected_type_from_legacy = 'candidate-wise'
        elif population_wise is False and filter_flag is None:
            expected_type_from_legacy = 'candidate-wise'
        elif filter_flag is False and population_wise is None:
            expected_type_from_legacy = 'candidate-wise'

        # Validation: if type is provided along with legacy parameters, they must be consistent
        if provided_type is not None and expected_type_from_legacy is not None:
            if provided_type != expected_type_from_legacy:
                raise ValueError(
                    f"Invalid objective configuration: 'type' is '{provided_type}' but legacy parameters "
                    f"indicate '{expected_type_from_legacy}'. "
                    f"(population_wise={population_wise}, filter={filter_flag}). "
                    "Please use either the 'type' parameter or the legacy 'population_wise'/'filter' parameters, "
                    "but ensure they don't conflict."
                )

        # Set type based on legacy parameters if type wasn't provided
        if expected_type_from_legacy is not None and provided_type is None:
            data['type'] = expected_type_from_legacy

        super().__init__(**data)

    @field_validator('optimization_direction')
    @classmethod
    def check_optimization_direction(cls, v):
        if v is None:
            return v
        if v not in {"maximize", "minimize"}:
            raise ValueError("optimization_direction must be either 'maximize' or 'minimize'")
        return v

    @model_validator(mode='after')
    def validate_filter_no_direction(self):
        """Validate that filter objectives don't have optimization_direction."""
        if self.type == "filter" and self.optimization_direction is not None:
            raise ValueError(
                f"Filter objective '{self.name}' cannot have optimization_direction. "
                "Filter objectives only return pass/fail (True/False) and do not have a direction to optimize."
            )
        return self
    
    @field_validator('weight')
    @classmethod
    def check_weight(cls, v):
        if v is None:
            return v
        if not isinstance(v, (int, float)):
            raise ValueError("weight must be a number")
        if v <= 0:
            raise ValueError("weight must be positive")
        return v

    @field_validator('name')
    @classmethod
    def check_name(cls, v):
        if not isinstance(v, str):
            raise ValueError("name must be a string")
        return v
    
    @field_validator('description')
    @classmethod
    def check_description(cls, v):
        if not isinstance(v, str):
            raise ValueError("description must be a string")
        return v
    
    def has_scorer(self) -> bool:
        """Check if this objective has a scorer function."""
        return self.scorer is not None

    @property
    def population_wise(self) -> bool:
        """
        Legacy property for backward compatibility.
        Returns True if this is a population-wise objective.
        """
        return self.type == "population-wise"

    @property
    def filter(self) -> bool:
        """
        Legacy property for backward compatibility.
        Returns True if this is a filter objective.
        """
        return self.type == "filter"

    async def _score_candidate_wise(self, population: Population, force_evaluation: bool = False) -> List[Optional[Union[float, bool]]]:
        """
        Score candidate(s) using this objective's regular scorer function (async).

        Args:
            population: Population of candidates to score

        Returns:
            Score value(s) or None for failed evaluations
            - For candidate-wise objectives: List[Optional[float]]
            - For filter objectives: List[Optional[bool]]

        Raises:
            ValueError: If no scorer is available
        """
        if not self.has_scorer():
            raise ValueError(f"No scorer available for objective '{self.name}'")

        if self.type == "population-wise":
            return None

        # Check if scorer is async (coroutine function)
        if inspect.iscoroutinefunction(self.scorer):
            return await self.scorer(population, force_evaluation=force_evaluation)
        else:
            return self.scorer(population, force_evaluation=force_evaluation)

    async def _score_population_wise(self, population: Population, force_evaluation: bool = False) -> Optional[float]:
        """Score a population with this objective's population-wise scorer (async)."""

        if not self.has_scorer():
            raise ValueError(f"No scorer available for objective '{self.name}'")

        if self.type != "population-wise":
            return None

        # Check if scorer is async (coroutine function)
        if inspect.iscoroutinefunction(self.scorer):
            return await self.scorer(population.candidates, force_evaluation=force_evaluation)
        else:
            return self.scorer(population.candidates, force_evaluation=force_evaluation)

    async def _score(self, population: Population, force_evaluation: bool = False) -> Union[List[Optional[Union[float, bool]]], Optional[float]]:
        """
        Score a population with this objective's scorer (async).

        Returns:
            - For candidate-wise objectives: List[Optional[float]]
            - For filter objectives: List[Optional[bool]]
            - For population-wise objectives: Optional[float]
        """
        if self.type == "population-wise":
            return await self._score_population_wise(population, force_evaluation)
        elif self.type == "candidate-wise":
            return await self._score_candidate_wise(population, force_evaluation)
        elif self.type == "filter":
            return await self._score_candidate_wise(population, force_evaluation)
        else:
            raise ValueError(f"Unknown objective type: {self.type}. Must be one of: 'candidate-wise', 'population-wise', 'filter'")

    async def score(self, population: Population, force_evaluation: bool = False) -> Population:
        """Score a population with this objective's scorer and save the scores to the population (async)."""
        if population is None or population.is_empty:
            raise ValueError("Population is None or empty for scoring.")
        scores = await self._score(population, force_evaluation)
        if self.type == "population-wise":
            population.set_score(self.name, scores)
        else:
            for candidate, score in zip(population.candidates, scores):
                candidate.set_score(self.name, score)
        return population
    
    def set_scorer(self, scorer: Callable[[List[Candidate]], Union[List[Optional[Union[float, bool]]], Optional[float]]]) -> None:
        """Set the scorer function for this objective."""
        self.scorer = scorer
    
    @property
    def is_maximization(self) -> bool:
        """Check if this is a maximization objective."""
        return self.optimization_direction == "maximize"

    @property
    def is_minimization(self) -> bool:
        """Check if this is a minimization objective."""
        return self.optimization_direction == "minimize"

    @property
    def is_filter(self) -> bool:
        """Check if this is a filter objective."""
        return self.filter
    
    def __str__(self) -> str:
        """String representation of the objective."""
        return f"Objective({self.name}, {self.optimization_direction})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Objective(name={self.name}, type={self.type}, "
                f"direction={self.optimization_direction}, weight={self.weight}, has_scorer={self.has_scorer()})") 


class ObjectiveIndex:
    """
    Indexes and organizes a list of Objective objects for fast retrieval and classification.
    """
    def __init__(self, objectives: List[Objective]):
        self.objectives = objectives
        self._by_name = {obj.name: obj for obj in objectives}
        self._population_wise = [obj for obj in objectives if obj.type == "population-wise"]
        self._regular = [obj for obj in objectives if obj.type == "candidate-wise"]
        self._filter = [obj for obj in objectives if obj.type == "filter"]
        # Only non-filter objectives have optimization direction
        self._maximize = [obj for obj in objectives if obj.type != "filter" and obj.is_maximization]
        self._minimize = [obj for obj in objectives if obj.type != "filter" and obj.is_minimization]

    def get_by_name(self, name: str) -> Optional[Objective]:
        """Retrieve an objective by its name."""
        return self._by_name.get(name)

    def get_all_population_wise(self) -> List[Objective]:
        """Return all population-wise objectives."""
        return self._population_wise

    def get_all_regular(self) -> List[Objective]:
        """Return all regular (non-population-wise) objectives."""
        return self._regular

    def get_maximization_objectives(self) -> List[Objective]:
        """Return all maximization objectives."""
        return self._maximize

    def get_minimization_objectives(self) -> List[Objective]:
        """Return all minimization objectives."""
        return self._minimize

    def get_filter_objectives(self) -> List[Objective]:
        """Return all filter objectives."""
        return self._filter

    def get_multiple_by_names(self, names: List[str]) -> List[Objective]:
        """Return objectives whose names are in names."""
        return [self.get_by_name(name) for name in names]

    def get_required_objectives(self, required_names: List[str]) -> List[Objective]:
        """Return objectives whose names are in required_names."""
        return self.get_multiple_by_names(required_names)

    def get_other_objectives(self, required_names: List[str]) -> List[Objective]:
        """Return objectives whose names are NOT in required_names."""
        return [obj for obj in self.objectives if obj.name not in required_names]

    def as_dict(self, required_names: Optional[List[str]] = None) -> Dict[str, Dict[str, Dict[str, List[Objective]]]]:
        """
        Return a nested dict structure similar to OptimizerModule.organize_objectives.
        If required_names is not provided, treat all as 'other'.
        """
        if required_names is None:
            required_names = []
        organized = {
            "regular": {
                "required": {"maximize": [], "minimize": []},
                "other": {"maximize": [], "minimize": []},
            },
            "population_wise": {
                "required": {"maximize": [], "minimize": []},
                "other": {"maximize": [], "minimize": []},
            },
        }
        for obj in self.objectives:
            first_key = "population_wise" if obj.type == "population-wise" else "regular"
            second_key = "required" if obj.name in required_names else "other"
            third_key = "maximize" if obj.is_maximization else "minimize"
            organized[first_key][second_key][third_key].append(obj)
        return organized

    def filter_objectives(self, objectives: Optional[List[Objective]] = None, filter_fn=None, **kwargs) -> List[Objective]:
        """
        Filter objectives by attribute values or a custom filter function.
        
        Args:
            filter_fn: Optional[Callable[[Objective], bool]] -- a custom filter function.
            **kwargs: attribute-value pairs to filter by (e.g., population_wise=True, optimization_direction='maximize').
        Returns:
            List[Objective]: objectives matching all conditions.
        """
        if objectives is None:
            results = self.objectives
        else:
            results = objectives
        if kwargs:
            for attr, value in kwargs.items():
                results = [obj for obj in results if getattr(obj, attr, None) == value]
        if filter_fn is not None:
            results = [obj for obj in results if filter_fn(obj)]
        return results
