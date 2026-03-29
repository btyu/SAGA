"""
Registry system for the SciLeo Agent framework.

This package contains the registry systems for managing modules, scorers, and
representation hash functions in the optimization framework.
"""

from .module_registry import (
    register_module,
    get_module_class,
    list_registered_modules
)

from .scorer_registry import (
    register_scorer,
    register_scorer_class,
    register_mcp_module,
    get_scorer,
    list_scorers,
    clear_scorers,
    reset_scorer_manager,
    get_scorer_metadata,
    ScorerManager
)

from .mcp_scorer_registry import (
    load_module_scorers,
    McpScorerManager
)

from .serializer_registry import (
    Serializer,
    SerializerManager,
    register_serializer_class,
    get_serializer,
    list_serializers,
    get_serializer_metadata,
    clear_serializers,
    reset_serializer_manager
)

__all__ = [
    "register_module",
    "get_module_class", 
    "list_registered_modules",
    "register_scorer",
    "register_scorer_class",
    "register_mcp_module",
    "get_scorer",
    "list_scorers",
    "clear_scorers",
    "reset_scorer_manager",
    "ScorerManager",
    "Serializer",
    "SerializerManager",
    "register_serializer_class",
    "get_serializer",
    "list_serializers",
    "get_serializer_metadata",
    "clear_serializers",
    "reset_serializer_manager"
]
