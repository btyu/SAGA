"""
SciLeo Agent: A modular framework for evolutionary optimization.

This framework provides a flexible, extensible architecture for implementing
evolutionary optimization algorithms with LLM-enhanced modules.

Key Components:
- OptimizationOrchestrator: Coordinates the optimization workflow
  Usage: OptimizationOrchestrator(config) - modules are auto-created from config
- Five core modules: Planner, ScorerCreator, Optimizer, Analyzer, KnowledgeManager
- Data models: Candidate, Objective, Population, OptimizationResult
- Registry system: For registering and retrieving custom implementations
- Configuration management: Comprehensive settings and LLM integration
- Utilities: LLM clients, logging, and common functionality

The framework follows a modular design where domain experts can implement
custom modules for specific optimization problems while leveraging the
common infrastructure and workflow orchestration.
"""

from .core.orchestrator import OptimizationOrchestrator
from .core.config import FrameworkConfig, create_config
from .core.modules.base import BaseModule
from .core.modules.planner import PlannerModule
from .core.modules.scorer_creator import ScorerCreatorModule
from .core.modules.optimizer import OptimizerModule
from .core.modules.analyzer import AnalyzerModule
from .core.modules.knowledge_manager import KnowledgeManagerModule
from .core.registry.module_registry import (
    register_module,
    get_module_class,
    list_registered_modules
)
from .core.data_models.candidate import Candidate
from .core.data_models.objective import Objective
from .core.data_models.results import OptimizationResult

from .__version__ import __version__

__all__ = [
    # Core orchestration
    "OptimizationOrchestrator",
    
    # Configuration
    "FrameworkConfig", 
    "create_config",
    
    # Base classes
    "BaseModule",
    
    # Framework modules
    "PlannerModule",
    "ScorerCreatorModule",
    "OptimizerModule", 
    "AnalyzerModule",
    "KnowledgeManagerModule",
    
    # Data models
    "Candidate",
    "Objective",
    "OptimizationResult",
    
    # Registry system
    "register_module",
    "get_module_class", 
    "list_registered_modules"
] 