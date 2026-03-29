from abc import abstractmethod
from typing import List, Any, Dict

from ..data_models.objective import Objective
from .base import BaseModule


class ScorerCreatorModule(BaseModule):
    """
    Module that creates/retrieves scoring functions for objectives.
    
    Given a list of objectives, it attaches the appropriate scorer functions
    to each objective, either by retrieving pre-defined scorers or generating
    new ones using LLMs or other methods.
    """
    
    @abstractmethod
    async def get_scorers(self, objectives: List[Objective]) -> Dict[str, Any]:
        """
        Create or retrieve scorers for the given objectives.
        
        Args:
            objectives: List of objectives that need scorers
            
        Returns:
            List of objectives with their scorer functions attached
        """
        pass
    
    async def process(self, *args, **kwargs) -> Dict[str, Any]:
        """Process method wrapper for create_scorers."""
        self._increment_call_count()
        return await self.get_scorers(*args, **kwargs)
