"""
Module implementations for the SciLeo Agent framework.

This package contains the base classes and implementations for all modules
in the optimization framework, along with a registry system for easy
module management.
"""

# Type alias for module instances
from typing import TYPE_CHECKING, Union

from .base import BaseModule
from .planner import PlannerModule
from .scorer_creator import ScorerCreatorModule
from .optimizer import OptimizerModule
from .analyzer import AnalyzerModule
from .knowledge_manager import KnowledgeManagerModule


# === TYPE DEFINITIONS ===
if TYPE_CHECKING:
    ModuleType = Union[PlannerModule, ScorerCreatorModule, OptimizerModule, AnalyzerModule, KnowledgeManagerModule]
else:
    ModuleType = BaseModule

__all__ = [
    # Base classes
    "BaseModule",
    "ModuleType",
    
    # Framework modules
    "PlannerModule",
    "ScorerCreatorModule", 
    "OptimizerModule",
    "AnalyzerModule",
    "KnowledgeManagerModule",
] 
