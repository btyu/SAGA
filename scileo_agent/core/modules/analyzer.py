from abc import abstractmethod
from typing import List, Dict, Any, Optional

from ..data_models import Candidate, Objective, Population
from .base import BaseModule

from ...utils.logging import get_logger

logger = get_logger()

class AnalyzerModule(BaseModule):
    """
    Module that analyzes optimization results and generates comprehensive reports.
    
    It analyzes the candidates and their scores, identifies trends and issues,
    and generates reports for the planner module to make decisions about the next iteration.
    """

    def __init__(self, module_id: str, config: Dict[str, Any] = None, llm_config=None):
        super().__init__(module_id, config, llm_config)

        if not self.has_llm():
            raise ValueError(f"Analyzer module {module_id} requires a LLM to be initialized.")
    
    @abstractmethod
    async def analyze(
        self,
        iteration_number: int,
        high_level_goal: str,
        current_population: Population,
        current_objectives: List[Objective],
        historical_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the current state and generate a comprehensive report.
        
        Args:
            iteration_number: The current iteration number
            high_level_goal: The overall optimization goal
            current_population: Current population of candidates
            current_objectives: List of current objectives used in this iteration
            historical_info: Optional historical information from previous iterations
            
        Returns:
            Comprehensive analysis report as a string
        """
        pass
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Process method wrapper for analyze."""
        self._increment_call_count()
        result = await self.analyze(*args, **kwargs)
        return result
