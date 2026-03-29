"""
Serializer registry for the SciLeo Agent framework.

This module provides a serialization system with a manager for organizing serializers
and decorators for marking classes and methods for registration. It follows the same
pattern as the scorer registry to maintain consistency.
"""

from typing import Dict, Any, Optional, List, Type
import inspect
from abc import ABC, abstractmethod

from ...utils.logging import get_logger
from ..data_models.candidate import Candidate


logger = get_logger()


class Serializer(ABC):
    """
    Abstract base class for serializers.
    
    Subclasses should implement serialize and deserialize methods to convert
    Candidate instances to/from basic data formats (str, dict, etc.) that
    can be used with MCP-based scorers.
    """
    
    @property
    @abstractmethod
    def sample_schema(self) -> str:
        """
        Return a string describing the data type/format of the serialized output.
        
        This information is used for prompt construction when working with MCP scorers,
        helping them understand what type of data they will receive.
        
        Examples:
            - "str" (for string representations like SMILES)
            - "dict" (for dictionary-based serialization)
            - "List[str]" (for list-based formats)
        
        Returns:
            String describing the data type/format of serialized data
        """
        pass
    
    @property
    @abstractmethod
    def sample_description(self) -> str:
        """
        Return a description of what the serialized data represents.
        
        This information is used for prompt construction when working with MCP scorers,
        helping them understand the meaning and content of the serialized data.
        
        Examples:
            - "the SMILES string of a molecule"
            - "a dictionary containing molecular properties and scores"
            - "the source code of a function"
        
        Returns:
            String describing what the serialized data represents
        """
        pass
    
    @abstractmethod
    def serialize(self, candidate: Candidate) -> Any:
        """
        Serialize a Candidate instance to a basic data format.
        
        Args:
            candidate: The Candidate instance to serialize
            
        Returns:
            Serialized representation (str, dict, list, etc.)
        """
        pass
    
    @abstractmethod
    def deserialize(self, data: Any) -> Candidate:
        """
        Deserialize data back to a Candidate instance.
        
        Args:
            data: The serialized data to convert back to Candidate
            
        Returns:
            Candidate instance
        """
        pass
    
    def serialize_batch(self, candidates: List[Candidate]) -> List[Any]:
        """
        Serialize a list of Candidate instances.
        
        Args:
            candidates: List of Candidate instances to serialize
            
        Returns:
            List of serialized representations
        """
        return [self.serialize(candidate) for candidate in candidates]
    
    def deserialize_batch(self, data_list: List[Any]) -> List[Candidate]:
        """
        Deserialize a list of data back to Candidate instances.
        
        Args:
            data_list: List of serialized data to convert back to Candidates
            
        Returns:
            List of Candidate instances
        """
        return [self.deserialize(data) for data in data_list]


class SerializerManager:
    """
    Manages all registered serializers for the SciLeo Agent framework.
    
    This class provides a centralized registry for serializer classes and
    manages their registration and retrieval. Serializers are automatically
    registered when using the @serializer_class decorator.
    
    Implements the singleton pattern to ensure only one instance exists.
    """
    
    _instance: Optional['SerializerManager'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """Create a new instance only if one doesn't exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the serializer manager (only once)."""
        if not self._initialized:
            self._serializers: Dict[str, Serializer] = {}
            self._serializer_metadata: Dict[str, Dict[str, Any]] = {}
            self._initialized = True
    
    def register_serializer(
        self,
        serializer: Serializer,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Register a serializer instance.
        
        Args:
            serializer: Serializer instance
            name: Name of the serializer (optional, uses class name if not provided)
            metadata: Metadata dictionary for the serializer
        """
        if name is None:
            if hasattr(serializer, '_serializer_name'):
                name = serializer._serializer_name
            else:
                name = serializer.__class__.__name__
        
        if metadata is None and hasattr(serializer, '_serializer_metadata'):
            metadata = serializer._serializer_metadata
        
        if name in self._serializers:
            raise ValueError(f"Serializer '{name}' already registered.")
        
        self._serializers[name] = serializer
        self._serializer_metadata[name] = metadata or {}
        logger.info(f"Registered serializer '{name}'")
    
    def unregister_serializer(self, name: str) -> bool:
        """
        Unregister a serializer.
        
        Args:
            name: Name of the serializer to unregister
            
        Returns:
            True if serializer was found and removed, False otherwise
        """
        if name in self._serializers:
            del self._serializers[name]
            del self._serializer_metadata[name]
            logger.info(f"Unregistered serializer '{name}'")
            return True
        return False
    
    def get_serializer(self, name: str) -> Optional[Serializer]:
        """
        Get a serializer by name.
        
        Args:
            name: Name of the serializer
            
        Returns:
            Serializer instance or None if not found
        """
        return self._serializers.get(name)
    
    def list_serializers(self) -> List[str]:
        """
        List all registered serializer names.
        
        Returns:
            List of serializer names
        """
        return list(self._serializers.keys())
    
    def get_serializer_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a serializer.
        
        Args:
            name: Name of the serializer
            
        Returns:
            Serializer metadata or None if not found
        """
        return self._serializer_metadata.get(name)
    
    def clear_serializers(self) -> None:
        """Clear all registered serializers."""
        self._serializers.clear()
        self._serializer_metadata.clear()
        logger.info("Cleared all registered serializers")
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (useful for testing).
        
        This method should be used with caution as it will clear all registered serializers.
        """
        if cls._instance is not None:
            cls._instance.clear_serializers()
            cls._instance = None
            cls._initialized = False
            logger.info("SerializerManager instance reset")


def register_serializer_class(
    name: Optional[str] = None,
    **kwargs
):
    """
    Decorator for automatically registering a Serializer class.
    
    This decorator ensures that:
    1. The class becomes a singleton
    2. The class can be instantiated without arguments  
    3. The class instance is automatically registered when the class is decorated
    
    Args:
        name: Name of the serializer (defaults to class name)
        **kwargs: Additional metadata for the serializer (e.g., description, category, etc.)
    
    Example:
        @register_serializer_class(name="dict_serializer", description="Serialize to dict")
        class DictSerializer(Serializer):
            def serialize(self, candidate: Candidate) -> dict:
                return candidate.model_dump()
            
            def deserialize(self, data: dict) -> Candidate:
                return Candidate.model_validate(data)
    """
    def decorator(cls: Type[Serializer]):
        if not issubclass(cls, Serializer):
            raise ValueError(f"Class '{cls.__name__}' must inherit from Serializer")
        
        # Store original __init__ method
        original_init = cls.__init__
        
        # Check if the class has required arguments
        sig = inspect.signature(original_init)
        required_params = [param for param in sig.parameters.values() 
                          if param.default == inspect.Parameter.empty and param.name != 'self'
                          and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
        
        if required_params:
            raise ValueError(f"Class '{cls.__name__}' has required arguments: {[p.name for p in required_params]}. "
                            f"Serializer classes must be instantiable without arguments.")
        
        # Add singleton functionality
        cls._instance = None
        
        def __new__(cls_instance, *args, **kwargs):
            if cls_instance._instance is None:
                cls_instance._instance = super(cls, cls_instance).__new__(cls_instance)
            return cls_instance._instance
        
        def __init__(self, *args, **kwargs):
            # Only initialize once
            if not hasattr(self, '_initialized'):
                original_init(self, *args, **kwargs)
                self._initialized = True
        
        # Replace the class methods
        cls.__new__ = __new__
        cls.__init__ = __init__
        
        # Store metadata on the class
        serializer_name = name if name is not None else cls.__name__
        cls._serializer_name = serializer_name
        cls._serializer_metadata = kwargs
        
        # Create instance and register it
        instance = cls()
        SerializerManager().register_serializer(
            serializer=instance,
            name=serializer_name,
            metadata=kwargs
        )
        
        return cls
    
    return decorator


# Convenience functions for accessing the singleton serializer manager
def get_serializer(name: str) -> Optional[Serializer]:
    """
    Get a serializer by name.
    
    Serializers are automatically registered when using the @register_serializer_class decorator.
    """
    return SerializerManager().get_serializer(name)


def list_serializers() -> List[str]:
    """
    List all registered serializer names.
    
    Serializers are automatically registered when using the @register_serializer_class decorator.
    """
    return SerializerManager().list_serializers()


def get_serializer_metadata(name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a serializer.
    
    Args:
        name: Name of the serializer
        
    Returns:
        Serializer metadata or None if not found
    """
    return SerializerManager().get_serializer_metadata(name)


def clear_serializers() -> None:
    """
    Clear all registered serializers.
    
    Note: This will clear all serializers, including those automatically registered
    by the @register_serializer_class decorator.
    """
    SerializerManager().clear_serializers()


def reset_serializer_manager() -> None:
    """
    Reset the singleton SerializerManager instance (useful for testing).
    
    This will clear all registered serializers and reset the manager to its initial state.
    """
    SerializerManager.reset_instance()