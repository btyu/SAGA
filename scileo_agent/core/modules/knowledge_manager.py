from abc import abstractmethod
from typing import List, Dict, Any, Optional

from ..data_models import Candidate, Population, Objective
from .base import BaseModule


class KnowledgeManagerModule(BaseModule):
    """
    Module that manages all data and knowledge for the optimization process.
    
    It serves as the single source of truth, storing candidates, objectives,
    analysis results, and providing access to historical data.
    """
    
    @abstractmethod
    def store_population(self, population: Population, iteration: int) -> None:
        """
        Store population in the knowledge base.
        
        Args:
            population: Population to store
            iteration_number: Iteration number
        """
        pass
    
    @abstractmethod
    def store_objectives(self, objectives: List[Objective], iteration: int) -> None:
        """
        Store objectives in the knowledge base.
        
        Args:
            objectives: List of objectives to store
        """
        pass
    
    @abstractmethod
    def store_analysis_report(self, report: str, iteration: int) -> None:
        """
        Store analysis report for a specific iteration.
        
        Args:
            report: Analysis report string
            iteration: Iteration number
        """
        pass
    
    @abstractmethod
    def get_population(
        self,
        iteration,
    ) -> Population:
        """
        Retrieve candidates based on filter criteria.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Population of candidates
        """
        pass
    
    @abstractmethod
    def get_objectives(self, iteration: int) -> List[Objective]:
        """
        Retrieve objectives for a specific iteration.
        
        Args:
            iteration: iteration number
            
        Returns:
            List of objectives
        """
        pass
    
    @abstractmethod
    def get_analysis_report(self, iteration: int) -> Optional[str]:
        """
        Retrieve analysis report for a specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Analysis report string or None if not found
        """
        pass
    
    @abstractmethod
    def get_historical_summary(self, iteration: int) -> Dict[str, Any]:
        """
        Get a summary of historical data for analysis.
        
        Returns:
            Dictionary containing historical summary
        """
        pass
    
    @abstractmethod
    def clear_data(self) -> None:
        """Clear all stored data."""
        pass
    
    def process(self, *args, **kwargs) -> Any:
        """Process method wrapper."""
        self._increment_call_count()
        # KnowledgeManagerModule doesn't have a single process method
        # Individual methods should be called directly
        return None
