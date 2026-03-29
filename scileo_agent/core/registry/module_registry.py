"""
Module registry system for the SciLeo Agent framework.

This module provides a decorator-based registration system that allows users
to easily register and retrieve custom module implementations by name and version.

The registry supports the five core module types:
- planner: PlannerModule implementations
- scorer_creator: ScorerCreatorModule implementations  
- optimizer: OptimizerModule implementations
- analyzer: AnalyzerModule implementations
- knowledge_manager: KnowledgeManagerModule implementations

Example usage:
    @register_module("my_optimizer", "1.0.0")
    class MyOptimizer(OptimizerModule):
        def optimize(self, ...):
            pass
            
    # Later retrieve the module
    optimizer_class = get_module_class("optimizer", "my_optimizer", "1.0.0")
"""

from typing import Dict, Type, Optional, List, Tuple, Any
from packaging import version

from ..modules import BaseModule, PlannerModule, ScorerCreatorModule, OptimizerModule, AnalyzerModule, KnowledgeManagerModule
from ...utils.logging import get_logger

logger = get_logger()


class ModuleManager:
    """
    Registry for managing different module implementations.
    
    This class maintains a registry of module classes organized by module type,
    name, and version, allowing for easy retrieval and version management.
    
    This class implements the singleton pattern to ensure only one instance
    exists throughout the application.
    """
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the module registry (only once due to singleton pattern)."""
        # Only initialize if not already initialized
        if hasattr(self, '_registry'):
            return
            
        # Structure: {module_type: {module_name: {version: class}}}
        self._registry: Dict[str, Dict[str, Dict[str, Type[BaseModule]]]] = {
            'planner': {},
            'scorer_creator': {},
            'optimizer': {},
            'analyzer': {},
            'knowledge_manager': {}
        }
        
        # Map base classes to module types
        self._type_mapping = {
            PlannerModule: 'planner',
            ScorerCreatorModule: 'scorer_creator',
            OptimizerModule: 'optimizer',
            AnalyzerModule: 'analyzer',
            KnowledgeManagerModule: 'knowledge_manager'
        }
    
    def register(
        self,
        module_type: str,
        module_name: str,
        module_version: str,
        module_class: Type[BaseModule],
        override: bool = False
    ) -> None:
        """
        Register a module class.
        
        Args:
            module_type: Type of module ('planner', 'scorer_creator', 'optimizer', 'analyzer', 'knowledge_manager')
            module_name: Name identifier for the module
            module_version: Version string (e.g., '1.0.0', '2.1.0')
            module_class: The module class to register
            override: Whether to override existing registration
            
        Raises:
            ValueError: If module type is invalid or class already registered without override
            TypeError: If module_class is not a valid module type
        """
        # Validate module type
        if module_type not in self._registry:
            raise ValueError(f"Invalid module type '{module_type}'. Must be one of: {list(self._registry.keys())}")
        
        # Validate module class
        if not self._is_valid_module_class(module_class, module_type):
            expected_base = self._get_base_class_for_type(module_type)
            raise TypeError(f"Module class must inherit from {expected_base.__name__} for type '{module_type}'")
        
        # Check for existing registration
        if (module_name in self._registry[module_type] and 
            module_version in self._registry[module_type][module_name] and 
            not override):
            raise ValueError(f"Module '{module_name}' version '{module_version}' already registered for type '{module_type}'")
        
        # Register the module
        if module_name not in self._registry[module_type]:
            self._registry[module_type][module_name] = {}
        
        self._registry[module_type][module_name][module_version] = module_class
        logger.info(f"Registered {module_type} module {module_name} version {module_version}")
    
    def get_class(
        self,
        module_type: str,
        module_name: str,
        module_version: Optional[str] = None
    ) -> Optional[Type[BaseModule]]:
        """
        Retrieve a module class by type, name, and optionally version.
        
        Args:
            module_type: Type of module
            module_name: Name of the module
            module_version: Specific version (if None, returns latest version)
            
        Returns:
            Module class if found, None otherwise
        """

        if module_type not in self._registry:
            return None
        
        if module_name not in self._registry[module_type]:
            return None
        versions = self._registry[module_type][module_name]
        
        if module_version is None:
            # Return latest version
            if not versions:
                return None
            latest_version = max(versions.keys(), key=lambda v: version.parse(v))
            return versions[latest_version]
        else:
            # Return specific version
            return versions.get(module_version)
    
    def list_modules(self, module_type: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
        """
        List all registered modules.
        
        Args:
            module_type: Specific module type to list (if None, lists all)
            
        Returns:
            Dictionary mapping module_type to list of (name, version) tuples
        """
        result = {}
        
        types_to_check = [module_type] if module_type else self._registry.keys()
        
        for mtype in types_to_check:
            if mtype in self._registry:
                result[mtype] = []
                for name, versions in self._registry[mtype].items():
                    for ver in versions.keys():
                        result[mtype].append((name, ver))
        
        return result
    
    def get_latest_version(self, module_type: str, module_name: str) -> Optional[str]:
        """
        Get the latest version of a specific module.
        
        Args:
            module_type: Type of module
            module_name: Name of the module
            
        Returns:
            Latest version string if found, None otherwise
        """
        if (module_type not in self._registry or 
            module_name not in self._registry[module_type]):
            return None
        
        versions = list(self._registry[module_type][module_name].keys())
        if not versions:
            return None
        
        return max(versions, key=lambda v: version.parse(v))
    
    def unregister(self, module_type: str, module_name: str, module_version: Optional[str] = None) -> bool:
        """
        Unregister a module.
        
        Args:
            module_type: Type of module
            module_name: Name of the module
            module_version: Specific version (if None, removes all versions)
            
        Returns:
            True if module was unregistered, False if not found
        """
        if (module_type not in self._registry or 
            module_name not in self._registry[module_type]):
            return False
        
        if module_version is None:
            # Remove all versions
            del self._registry[module_type][module_name]
            return True
        else:
            # Remove specific version
            if module_version in self._registry[module_type][module_name]:
                del self._registry[module_type][module_name][module_version]
                # Clean up empty module name entry
                if not self._registry[module_type][module_name]:
                    del self._registry[module_type][module_name]
                return True
            return False
    
    def clear(self, module_type: Optional[str] = None) -> None:
        """
        Clear all registered modules.
        
        Args:
            module_type: Specific module type to clear (if None, clears all)
        """
        if module_type is None:
            # Clear all
            for mtype in self._registry:
                self._registry[mtype].clear()
        else:
            # Clear specific type
            if module_type in self._registry:
                self._registry[module_type].clear()
    
    def _is_valid_module_class(self, module_class: Type, module_type: str) -> bool:
        """Check if a class is valid for the given module type."""
        expected_base = self._get_base_class_for_type(module_type)
        return expected_base and issubclass(module_class, expected_base)
    
    def _get_base_class_for_type(self, module_type: str) -> Type[BaseModule]:
        """Get the expected base class for a module type."""
        type_map = {
            'planner': PlannerModule,
            'scorer_creator': ScorerCreatorModule,
            'optimizer': OptimizerModule,
            'analyzer': AnalyzerModule,
            'knowledge_manager': KnowledgeManagerModule
        }
        return type_map.get(module_type)


def register_module(
    module_name: str,
    module_version: str = "1.0.0",
    module_type: Optional[str] = None,
    override: bool = False
):
    """
    Decorator to register a module class.
    
    Args:
        module_name: Name identifier for the module
        module_version: Version string (default: "1.0.0")
        module_type: Type of module (auto-detected if None)
        override: Whether to override existing registration
        
    Returns:
        Decorator function
        
    Example:
        @register_module("my_optimizer", "1.0.0")
        class MyOptimizer(OptimizerModule):
            def optimize(self, ...):
                pass
    """
    def decorator(cls):
        # Auto-detect module type if not provided
        detected_type = module_type
        if detected_type is None:
            for base_class, type_name in ModuleManager()._type_mapping.items():
                if issubclass(cls, base_class):
                    detected_type = type_name
                    break
        
        if detected_type is None:
            raise ValueError(f"Could not auto-detect module type for {cls.__name__}. Please specify module_type explicitly.")
        
        cls.module_type = detected_type
        cls.module_name = module_name
        cls.module_version = module_version
        
        # Register the module
        ModuleManager().register(
            module_type=detected_type,
            module_name=module_name,
            module_version=module_version,
            module_class=cls,
            override=override
        )
        
        return cls
    
    return decorator


def get_module_class(
    module_type: str,
    module_name: str,
    module_version: Optional[str] = None
) -> Optional[Type[BaseModule]]:
    """
    Get a registered module class.
    
    Args:
        module_type: Type of module
        module_name: Name of the module
        module_version: Specific version (if None, returns latest version)
        
    Returns:
        Module class if found, None otherwise
    """
    return ModuleManager().get_class(module_type, module_name, module_version)


def list_registered_modules(module_type: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
    """List all registered modules."""
    return ModuleManager().list_modules(module_type)


def get_latest_version(module_type: str, module_name: str) -> Optional[str]:
    """Get the latest version of a specific module."""
    return ModuleManager().get_latest_version(module_type, module_name)


def unregister_module(
    module_type: str,
    module_name: str,
    module_version: Optional[str] = None
) -> bool:
    """Unregister a module."""
    return ModuleManager().unregister(module_type, module_name, module_version)


def clear_registry(module_type: Optional[str] = None) -> None:
    """Clear the module registry."""
    ModuleManager().clear(module_type)
