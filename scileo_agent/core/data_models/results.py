"""
Results data model for optimization frameworks.

This module defines data structures for tracking optimization results
and performance metrics.
"""

from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from .candidate import Candidate
from .population import Population

class OptimizationResult(BaseModel):
    """
    Represents the results of an optimization run.
    
    This class tracks the essential information about the optimization process,
    including the final results, timing, and status.
    """
    
    # Run identification
    run_id: str = Field(..., description="Unique identifier for this optimization run")
    run_name: Optional[str] = Field(None, description="Human-readable name for this run")
    
    # Timing information
    start_time: Optional[datetime] = Field(None, description="When the optimization started")
    end_time: Optional[datetime] = Field(None, description="When the optimization ended")
    
    # Optimization context
    high_level_goal: Optional[str] = Field(None, description="The high-level optimization goal")
    context_information: Optional[str] = Field(None, description="Context information for the optimization")
    config: Dict[str, Any] = Field(default_factory=dict, description="Configuration parameters")
    
    # Results
    final_population: Optional[Population] = Field(None, description="Final population of candidates")
    all_candidates_population: Optional[Population] = Field(None, description="All candidates from all iterations (when return_all_candidates is enabled)")
    final_analysis_report: Optional[str] = Field(None, description="Final analysis report from the analyzer")
    
    # Progress tracking
    total_generations: int = Field(default=0, description="Total number of generations/iterations")
    
    # Status and termination
    status: str = Field(default="running", description="Optimization status")
    termination_reason: Optional[str] = Field(None, description="Reason for termination")
    error_message: Optional[str] = Field(None, description="Error message if optimization failed")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the total duration of the optimization run in seconds."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds()
    
    @property
    def is_finished(self) -> bool:
        """Check if the optimization run has finished."""
        return self.status in ["completed", "failed", "terminated"]
    
    @property
    def is_successful(self) -> bool:
        """Check if the optimization run was successful."""
        return self.status == "completed" and self.final_population is not None and self.final_population.size > 0
    
    @property
    def best_candidates(self) -> List[Candidate]:
        """Get the best candidates from the final population, sorted by their best score."""
        if not self.final_population or self.final_population.size == 0:
            return []
        
        candidates = list(self.final_population.candidates)
        
        # Sort by the best score (highest value) across all objectives
        def get_best_score(candidate: Candidate) -> float:
            if not candidate.scores:
                return 0.0
            return max(candidate.scores.values())
        
        return sorted(candidates, key=get_best_score, reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization run."""
        return {
            "run_info": {
                "run_id": self.run_id,
                "run_name": self.run_name,
                "goal": self.high_level_goal,
                "status": self.status,
                "termination_reason": self.termination_reason
            },
            "timing": {
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": self.end_time.isoformat() if self.end_time else None,
                "duration_seconds": self.duration
            },
            "results": {
                "total_generations": self.total_generations,
                "final_population_size": self.final_population.size if self.final_population else 0,
                "best_candidates_count": len(self.best_candidates),
                "has_analysis_report": self.final_analysis_report is not None
            }
        }
    
    def __str__(self) -> str:
        """String representation of the optimization result."""
        return f"OptimizationResult(run_id={self.run_id}, status={self.status}, generations={self.total_generations})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"OptimizationResult(run_id={self.run_id}, status={self.status}, "
                f"generations={self.total_generations}, final_pop_size={self.final_population.size if self.final_population else 0})") 