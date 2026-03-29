"""
Core components of the SciLeo Agent framework.

This package contains the essential components for optimization orchestration,
configuration management, and data models:

- Orchestrator: Main workflow coordinator
- Configuration: Framework and module configuration management
- Data Models: Data structures for candidates, populations, objectives, and results
- Modules: Base classes and registry for framework modules

These components form the foundation of the optimization framework and provide
the interfaces that domain experts implement for specific optimization problems.
"""

from .orchestrator import OptimizationOrchestrator
from .config import FrameworkConfig, create_config

from . import data_models, modules, registry

__all__ = [
    # Orchestration
    "OptimizationOrchestrator",
    # Configuration
    "FrameworkConfig",
    "create_config",
    
    "data_models",
    "modules",
    "registry",
] 