"""
General knowledge manager module for the SciLeo Agent framework.

This module provides a general-purpose knowledge manager that stores and manages
optimization data in memory with optional persistence to disk.
"""

import os
import json
import pickle
from typing import List, Dict, Any, Optional, Callable, Set
from pydantic import Field
from datetime import datetime
from collections import defaultdict

from scileo_agent.core.modules import KnowledgeManagerModule
from scileo_agent.core.data_models import Candidate, Population, Objective
from scileo_agent.core.registry.module_registry import register_module


@register_module("basic_knowledge_manager", "0.1.0")
class BasicKnowledgeManager(KnowledgeManagerModule):
    """
    Basic knowledge manager module that provides in-memory storage.
    
    This module stores population, objectives, and analysis reports in memory
    with optional persistence to disk for data recovery.
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None, llm_config=None):
        """
        Initialize the basic knowledge manager module.
        
        Args:
            module_id: Unique identifier for this module
            config: Configuration parameters
            llm_config: LLM configuration
        """
        super().__init__(module_id, config, llm_config)
        
        # Storage configuration
        self.enable_persistence = config.get("enable_persistence", False)
        self.persistence_dir = config.get("persistence_dir", "knowledge_storage")
        
        # In-memory storage with indexing
        self.populations_by_iteration: Dict[int, Population] = {}
        self.objectives_by_iteration: Dict[int, List[Objective]] = {}
        self.analysis_reports_by_iteration: Dict[int, str] = {}
        self.metadata: Dict[str, Any] = {}
        
        # Initialize persistence directory
        if self.enable_persistence:
            self._setup_persistence()
            self._load_from_disk()
    
    def _setup_persistence(self):
        """Set up persistence directory."""
        if not os.path.exists(self.persistence_dir):
            os.makedirs(self.persistence_dir)
    
    def _load_from_disk(self):
        """Load data from disk if available."""
        try:
            # Load candidates
            populations_file = os.path.join(self.persistence_dir, "populations.pkl")
            if os.path.exists(populations_file):
                with open(populations_file, 'rb') as f:
                    self.populations_by_iteration = pickle.load(f)
            
            # Load objectives
            objectives_file = os.path.join(self.persistence_dir, "objectives.pkl")
            if os.path.exists(objectives_file):
                with open(objectives_file, 'rb') as f:
                    self.objectives_by_iteration = pickle.load(f)
            
            # Load analysis reports
            reports_file = os.path.join(self.persistence_dir, "reports.json")
            if os.path.exists(reports_file):
                with open(reports_file, 'r') as f:
                    # Convert string keys back to int
                    reports_data = json.load(f)
                    self.analysis_reports = {int(k): v for k, v in reports_data.items()}
            
            # Load metadata
            metadata_file = os.path.join(self.persistence_dir, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.metadata = json.load(f)
            
        except Exception as e:
            raise
    
    def _save_to_disk(self):
        """Save data to disk."""
        if not self.enable_persistence:
            return
        
        try:
            # Save candidates
            populations_file = os.path.join(self.persistence_dir, "populations.pkl")
            with open(populations_file, 'wb') as f:
                pickle.dump(self.populations_by_iteration, f)
            
            # Save objectives (without scorer functions)
            objectives_file = os.path.join(self.persistence_dir, "objectives.pkl")
            serializable_objectives = {}
            for iteration, objectives in self.objectives_by_iteration.items():
                # Remove scorer functions for serialization
                serializable_obj = []
                for obj in objectives:
                    obj_dict = obj.model_dump()
                    obj_dict.pop('scorer', None)  # Remove scorer function
                    serializable_obj.append(obj_dict)
                serializable_objectives[iteration] = serializable_obj
            
            with open(objectives_file, 'wb') as f:
                pickle.dump(serializable_objectives, f)
            
            # Save analysis reports
            reports_file = os.path.join(self.persistence_dir, "reports.json")
            with open(reports_file, 'w', encoding='utf-8') as f:
                # Convert int keys to string for JSON
                reports_data = {str(k): v for k, v in self.analysis_reports.items()}
                json.dump(reports_data, f, indent=2)
            
            # Save metadata
            self.metadata.update({
                "last_saved": datetime.now().isoformat()
            })
            
            metadata_file = os.path.join(self.persistence_dir, "metadata.json")
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
            
        except Exception as e:
            print(f"Error saving to disk: {e}")
    
    def store_population(self, population: Population, iteration: int) -> None:
        """
        Store populations in the knowledge base.
        
        Args:
            candidates: List of candidates to store
            iteration: Optional iteration number (uses current_iteration if None)
        """
        
        self.populations_by_iteration[iteration] = population
        
        # Save to disk if persistence is enabled
        self._save_to_disk()
    
    def store_objectives(self, objectives: List[Objective], iteration: int) -> None:
        """
        Store objectives in the knowledge base.
        
        Args:
            objectives: List of objectives to store
            iteration: Optional iteration number (uses current_iteration if None)
        """
        
        self.objectives_by_iteration[iteration] = objectives
        
        # Save to disk if persistence is enabled
        self._save_to_disk()
    
    def store_analysis_report(self, report: str, iteration: int) -> None:
        """
        Store analysis report for a specific iteration.
        
        Args:
            report: Analysis report string
            iteration: Optional iteration number (uses current_iteration if None)
        """
        
        self.analysis_reports_by_iteration[iteration] = report
        
        # Save to disk if persistence is enabled
        self._save_to_disk()
    
    def get_population(
        self,
        iteration: int
    ) -> Population:
        """
        Retrieve population for a specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Population for the specified iteration
        """
        return self.populations_by_iteration.get(iteration)
    
    def get_objectives(self, iteration: int) -> List[Objective]:
        """
        Retrieve objectives for a specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            List of objectives for the specified iteration
        """
        return self.objectives_by_iteration.get(iteration)
    
    def get_analysis_report(self, iteration: int) -> Optional[str]:
        """
        Retrieve analysis report for a specific iteration.
        
        Args:
            iteration: Iteration number
            
        Returns:
            Analysis report string or None if not found
        """
        return self.analysis_reports_by_iteration.get(iteration)
    
    def get_historical_summary(self, iteration: int) -> Dict[str, Any]:
        """
        Get a summary of historical data for analysis.
        
        Returns:
            Dictionary containing historical summary
        """
        return {'last_population': self.populations_by_iteration.get(iteration - 1, None)}
    
    def clear_data(self) -> None:
        """Clear all stored data."""
        self.populations_by_iteration.clear()
        self.objectives_by_iteration.clear()
        self.analysis_reports_by_iteration.clear()
        self.metadata.clear()
        
        # Clear disk storage if persistence is enabled
        if self.enable_persistence:
            try:
                for filename in ["populations.pkl", "objectives.pkl", "reports.json", "metadata.json"]:
                    file_path = os.path.join(self.persistence_dir, filename)
                    if os.path.exists(file_path):
                        os.remove(file_path)
            except Exception as e:
                print(f"Error clearing disk storage: {e}")
    
    def _estimate_memory_usage(self) -> str:
        """Estimate memory usage."""
        import sys
        
        total_size = 0
        total_size += sys.getsizeof(self.populations_by_iteration)
        total_size += sys.getsizeof(self.objectives_by_iteration)
        total_size += sys.getsizeof(self.analysis_reports_by_iteration)
        total_size += sys.getsizeof(self.metadata)
        
        # Convert to human-readable format
        for unit in ['B', 'KB', 'MB', 'GB']:
            if total_size < 1024:
                return f"{total_size:.1f} {unit}"
            total_size /= 1024
        
        return f"{total_size:.1f} TB"
