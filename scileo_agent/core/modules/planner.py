from abc import abstractmethod
from typing import List, Optional, Dict, Any, Tuple, TYPE_CHECKING

from ..data_models.objective import Objective
from .base import BaseModule
from ...utils.logging import get_logger

if TYPE_CHECKING:
    from ..config import LLMConfig


logger = get_logger()


class PlannerModule(BaseModule):
    """
    Module that plans optimization objectives for each iteration.
    
    The planner module analyzes the current state and decides whether to continue and which objectives
    to use for the new iteration based on the high-level goal and analysis reports.
    """

    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None, llm_config: Optional['LLMConfig']=None):
        super().__init__(module_id, config, llm_config)

        if not self.has_llm():
            raise ValueError(f"Planner module {module_id} requires an LLM client.")
    
    @abstractmethod
    async def plan_objectives(
        self,
        iteration_number: int,
        high_level_goal: str,
        context_information: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        analysis_report: Optional[str] = None
    ) -> Tuple[List[Objective], Dict[str, Any]]:
        """
        Plan the objectives for the new iteration.
        
        Args:
            iteration_number: The current iteration number
            high_level_goal: The overall optimization goal
            context_information: Optional context information for the task
            initial_objectives: The initial list of objectives provided by users
            analysis_report: Optional analysis report from the AnalyzerModule
            
        Returns:
            List of objectives to use for this iteration
        """
        pass

    @abstractmethod
    async def _process(
        self, 
        iteration_number: int,
        high_level_goal: str,
        context_information: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        analysis_report: Optional[str] = None,
        mode: Optional[str] = "normal",
        additional_information: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Do the whole planning process."""
        pass
    
    async def process(
        self, 
        iteration_number: int,
        high_level_goal: str,
        context_information: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        analysis_report: Optional[str] = None,
        mode: Optional[str] = "normal",
        additional_information: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Do the whole planning process."""

        self._increment_call_count()
        result = await self._process(
            iteration_number=iteration_number,
            high_level_goal=high_level_goal,
            context_information=context_information,
            initial_objectives=initial_objectives,
            analysis_report=analysis_report,
            mode=mode,
            additional_information=additional_information
        )

        return result
