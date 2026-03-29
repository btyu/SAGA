"""
Data models for the SciLeo Agent framework.

This package contains the core data structures for representing optimization
components:

- Candidate: Individual optimization candidates with properties and scores
- Objective: Optimization objectives with scoring functions
- OptimizationResult: Complete optimization results and metadata

All models use Pydantic for validation and serialization.
"""

from .candidate import Candidate
from .population import Population
from .results import OptimizationResult
from .objective import Objective, ObjectiveIndex

__all__ = [
    "Candidate",
    "Population",
    "OptimizationResult",
    "Objective",
    "ObjectiveIndex",
] 