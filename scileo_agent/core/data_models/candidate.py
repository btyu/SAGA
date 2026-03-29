"""
Candidate data model for optimization frameworks.

This module defines the core data structures for representing optimization candidates
with flexible properties and metadata.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field
import uuid


class Candidate(BaseModel):
    """
    Represents a single optimization candidate with properties and metadata.
    
    This is a flexible data model that can represent various types of optimization
    candidates (molecules, designs, configurations, etc.) with associated properties
    and evaluation results.
    """
    
    # Core identification
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    representation: Any = Field(..., description="Representation of the candidate (e.g., SMILES, JSON, etc.)")
    
    # Properties and scores
    properties: Dict[str, Any] = Field(default_factory=dict, description="Properties of the candidate")
    scores: Dict[str, Optional[Union[float, bool]]] = Field(default_factory=dict, description="Evaluation scores from various criteria (float for numerical objectives, bool for filter objectives)")
    
    # Evaluation metadata
    evaluation_results: Dict[str, Any] = Field(default_factory=dict, description="Detailed evaluation results")
    
    # Flags and additional metadata
    flags: Dict[str, bool] = Field(default_factory=dict, description="Boolean flags (e.g., novelty, feasibility)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    # Validation methods
    def is_evaluated(self, objective_names: List[str]) -> List[Optional[bool]]:
        """Check if the candidate has been evaluated by each objective.
        
        Args:
            objective_names: List of objective names to check
            
        Returns:
            List of booleans indicating if the candidate has been evaluated by each objective
            Elements are None if the objective has been evaluated but the result is None (usually due to a failed evaluation)
        """
        raise NotImplementedError("This method is deprecated or needs recheck.")
        evaluated = []
        for objective_name in objective_names:
            if objective_name in self.evaluation_results:
                if self.evaluation_results[objective_name] is None:
                    evaluated.append(None)
                else:
                    evaluated.append(True)
            else:
                evaluated.append(False)
        return evaluated
    
    def get_score(self, objective_name: str) -> Optional[Union[float, bool]]:
        """Get a specific score by objective name.

        Returns:
            float for numerical objectives, bool for filter objectives, or None if not evaluated
        """
        return self.scores[objective_name]
    
    def get_property(self, prop_name: str) -> Any:
        """Get a specific property by name."""
        return self.properties[prop_name]
    
    def set_score(self, objective_name: str, score: Union[float, bool]) -> None:
        """Add or update a score for a specific objective.

        Args:
            objective: Name of the objective
            score: Score value (float for numerical objectives, bool for filter objectives)
        """
        self.scores[objective_name] = score
        
    def set_property(self, prop_name: str, value: Any) -> None:
        """Add or update a property."""
        self.properties[prop_name] = value
        
    def set_flag(self, flag_name: str, value: bool) -> None:
        """Set a boolean flag."""
        self.flags[flag_name] = value
        
    def get_flag(self, flag_name: str) -> bool:
        """Get a boolean flag value."""
        return self.flags[flag_name]
    
    def __str__(self) -> str:
        """String representation of the candidate."""
        return f"Candidate(id={self.id[:8]}..., repr={self.representation[:50]}...)"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Candidate(id={self.id}, scores={len(self.scores)}, properties={len(self.properties)})")
