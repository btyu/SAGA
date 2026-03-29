"""
Scorer utility for the SciLeo Agent framework.

This module provides a scoring system with a manager for organizing scorers
and decorators for marking functions and classes for manual registration.
"""

from typing import Dict, Callable, Any, Optional, List, Type, Union, get_origin, get_args
from functools import wraps
import inspect
from copy import deepcopy

from ...utils.logging import get_logger
from ..data_models.candidate import Candidate
from ..data_models.population import Population
from .mcp_scorer_registry import McpScorerManager
from .serializer_registry import SerializerManager


logger = get_logger()


def convert_filter_result(result, scorer_type: str):
    """
    Convert filter scorer results to boolean list if needed.

    This is a common utility function for both MCP and conventional scorers.
    For filter-type scorers, it ensures the result is a list of boolean values,
    converting numeric values (0/1) to False/True if needed.

    Args:
        result: The scorer result to convert (typically List[Optional[float]] or List[Optional[bool]])
        scorer_type: Type of scorer ("candidate-wise", "population-wise", or "filter")

    Returns:
        Original result for non-filter scorers, converted boolean list for filter scorers
    """
    if scorer_type != "filter":
        return result

    if isinstance(result, list):
        # Convert each score to bool
        bool_result = []
        for score in result:
            if score is None:
                bool_result.append(None)
            elif isinstance(score, bool):
                # Already boolean, keep as is
                bool_result.append(score)
            else:
                # Convert numeric to bool: round to 0/1 then to False/True
                rounded = round(float(score))
                bool_result.append(bool(rounded))
        return bool_result
    return result


class ScorerManager:
    """
    Manages all registered scorers for the SciLeo Agent framework.
    
    This class provides a centralized registry for scoring functions and
    manages their registration and retrieval. Scorers are assigned to
    objectives and called from there, not directly from the manager.
    
    Scorers are automatically registered when using the @scorer and @scorer_class
    decorators. Manual registration is still supported for backward compatibility.
    
    Implements the singleton pattern to ensure only one instance exists.
    """
    
    _instance: Optional['ScorerManager'] = None
    _initialized: bool = False

    def __new__(cls, run_in_docker: bool = True):
        """Create a new instance only if one doesn't exist."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, run_in_docker: bool = True):
        """
        Initialize the scorer manager (only once).

        Args:
            run_in_docker: Whether to run MCP scorer modules in Docker containers (default: True)
        """
        if not self._initialized:
            self.run_in_docker = run_in_docker
            self._scorers: Dict[str, Callable[[List[Candidate]], List[Optional[float]]]] = {}
            self._scorer_metadata: Dict[str, Dict[str, Any]] = {}
            # Track MCP scorers and their associated modules for cleanup
            self._mcp_scorers: Dict[str, str] = {}  # scorer_name -> module_name
            self._mcp_modules: Dict[str, List[str]] = {}  # module_name -> [scorer_names]
            self._initialized = True
    
    @property
    def mcp_scorer_to_module(self) -> Dict[str, str]:
        """Get mapping of MCP scorer names to their module names."""
        return deepcopy(self._mcp_scorers)
    
    @property
    def mcp_module_to_scorers(self) -> Dict[str, List[str]]:
        """Get mapping of MCP module names to their scorer names."""
        return deepcopy(self._mcp_modules)
    
    def register_scorer(
        self,
        scorer_func: Callable,
        name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        population_wise: bool = False,
        type: Optional[str] = None,
        is_mcp_scorer: bool = False,
        mcp_module_name: Optional[str] = None
    ) -> None:
        """
        Register a scorer function.

        Args:
            scorer_func: Function that takes a List[Candidate] and returns List[Optional[float]],
                         or takes a List[Candidate] and returns Optional[float] if population_wise.
            name: Name of the scorer (optional, extracted from decorated function if not provided)
            metadata: Metadata dictionary for the scorer (optional, extracted from decorated function if not provided)
            population_wise: (Deprecated) Whether this scorer is a population objective scorer. Use 'type' instead.
            type: Type of scorer - "candidate-wise", "population-wise", or "filter". If not provided, inferred from population_wise.
            is_mcp_scorer: Whether this is an MCP-based scorer
            mcp_module_name: Name of the MCP module (required if is_mcp_scorer=True)
        """
        if not is_mcp_scorer:
            raise RuntimeError("Conventional scorers are not supported anymore. Please use MCP-based scorers.")

        # Extract name from decorated function if not provided
        if name is None:
            if hasattr(scorer_func, '_scorer_name'):
                name = scorer_func._scorer_name
            else:
                name = scorer_func.__name__
        # Extract metadata from decorated function if not provided
        if metadata is None and hasattr(scorer_func, '_scorer_metadata'):
            metadata = scorer_func._scorer_metadata
        # Extract population_wise from decorated function if not provided
        if hasattr(scorer_func, '_population_wise'):
            population_wise = scorer_func._population_wise
        # Extract type from decorated function if not provided
        if hasattr(scorer_func, '_type'):
            type = scorer_func._type

        if name in self._scorers:
            raise ValueError(f"Scorer '{name}' already registered.")

        self._scorers[name] = scorer_func
        self._scorer_metadata[name] = metadata or {}

        # Determine scorer type with priority: explicit type > population_wise
        if type is not None:
            # Validate type
            if type not in ("candidate-wise", "population-wise", "filter"):
                raise ValueError(f"Invalid scorer type '{type}'. Must be 'candidate-wise', 'population-wise', or 'filter'.")
            scorer_type = type
        else:
            # Infer from population_wise for backward compatibility
            scorer_type = "population-wise" if population_wise else "candidate-wise"

        # Store type internally
        self._scorer_metadata[name]['type'] = scorer_type
        # Also store population_wise for backward compatibility
        self._scorer_metadata[name]['population_wise'] = (scorer_type == "population-wise")
        
        # Track MCP scorers
        if is_mcp_scorer and mcp_module_name:
            self._mcp_scorers[name] = mcp_module_name
            if mcp_module_name not in self._mcp_modules:
                self._mcp_modules[mcp_module_name] = []
            self._mcp_modules[mcp_module_name].append(name)
            logger.info(f"Registered MCP scorer '{name}' (type={scorer_type}) from module '{mcp_module_name}' with serializer '{metadata.get('serializer', 'unknown') if metadata else 'unknown'}'")
        else:
            logger.info(f"Registered conventional scorer '{name}' (type={scorer_type})")
    
    def register_scorer_class(self, cls: Type) -> None:
        """
        Register all scorer methods from a class.
        
        Args:
            cls: Class that contains methods decorated with @scorer
            
        Note: This method is deprecated. Use the @scorer_class decorator instead,
        which automatically registers scorer methods when the class is instantiated.
        """
        import warnings
        warnings.warn(
            "register_scorer_class is deprecated. Use the @scorer_class decorator instead, "
            "which automatically registers scorer methods when the class is instantiated.",
            DeprecationWarning,
            stacklevel=2
        )
        
        if not hasattr(cls, '_instance'):
            raise ValueError(f"Class '{cls.__name__}' must be decorated with @scorer_class")
        
        # Create instance if it doesn't exist
        instance = cls()
        
        # Find and register all methods decorated with @scorer
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            if hasattr(attr, '_is_scorer') and attr._is_scorer:
                # Bind the method to the instance
                bound_method = attr.__get__(instance, cls)
                
                # Register the bound method
                self.register_scorer(
                    name=attr._scorer_name,
                    scorer_func=bound_method,
                    metadata=attr._scorer_metadata
                )
    
    def unregister_scorer(self, name: str) -> bool:
        """
        Unregister a scorer.
        
        Args:
            name: Name of the scorer to unregister
            
        Returns:
            True if scorer was found and removed, False otherwise
        """
        if name in self._scorers:
            del self._scorers[name]
            del self._scorer_metadata[name]
            
            # Remove from MCP tracking if it's an MCP scorer
            if name in self._mcp_scorers:
                module_name = self._mcp_scorers[name]
                del self._mcp_scorers[name]
                if module_name in self._mcp_modules:
                    self._mcp_modules[module_name].remove(name)
                    if not self._mcp_modules[module_name]:  # Remove empty module
                        del self._mcp_modules[module_name]

                # Also unregister from McpScorerManager
                try:
                    mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
                    mcp_manager.unregister_scorer(name)
                    logger.debug(f"Unregistered MCP scorer '{name}' from module '{module_name}'")
                except Exception as e:
                    logger.warning(f"Error unregistering MCP scorer '{name}': {e}")
            else:
                logger.debug(f"Unregistered scorer '{name}'")
            return True
        return False
    
    def get_scorer(self, name: str, case_sensitive=False) -> Optional[Callable[[List[Candidate]], List[Optional[float]]]]:
        """
        Get a scorer function by name.
        
        Args:
            name: Name of the scorer
            
        Returns:
            Scorer function or None if not found
        """
        scorer = self._scorers.get(name)
        if scorer is None and not case_sensitive:
            # Try case-insensitive match
            for scorer_name, scorer_func in self._scorers.items():
                if scorer_name.lower() == name.lower():
                    scorer = scorer_func
                    break
        return scorer
    
    def list_scorers(self) -> List[str]:
        """
        List all registered scorer names.
        
        Returns:
            List of scorer names
        """
        return list(self._scorers.keys())
    
    def get_scorer_metadata(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get metadata for a scorer.

        Args:
            name: Name of the scorer

        Returns:
            Scorer metadata or None if not found
        """
        return self._scorer_metadata.get(name)

    def get_module_path(self, scorer_name: str) -> Optional[str]:
        """
        Get the absolute path of the module containing the scorer.

        For MCP scorers, returns the module path from McpScorerManager.
        For non-MCP scorers, returns None.

        Args:
            scorer_name: Name of the scorer

        Returns:
            Absolute path string to the module, or None if not found or not an MCP scorer
        """
        # Check if this is an MCP scorer
        if scorer_name in self._mcp_scorers:
            module_name = self._mcp_scorers[scorer_name]
            try:
                mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
                return mcp_manager.get_module_path(module_name)
            except Exception as e:
                logger.warning(f"Error getting module path for MCP scorer '{scorer_name}': {e}")
                return None
        else:
            logger.warning(f"Scorer '{scorer_name}' is not an MCP scorer, and therefore has no module path recorded")
        return None
    
    def set_run_in_docker(self, run_in_docker: bool) -> None:
        """
        Set whether to run MCP scorer modules in Docker containers.

        Args:
            run_in_docker: Whether to run MCP scorer modules in Docker containers

        Note:
            This updates both ScorerManager and McpScorerManager settings.
            It only affects new MCP servers started after this setting is changed.
            Existing running servers will not be affected.
        """
        self.run_in_docker = run_in_docker

        # Also update McpScorerManager if it exists
        try:
            mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
            mcp_manager.set_run_in_docker(run_in_docker)
        except Exception as e:
            logger.error(f"Error updating McpScorerManager run_in_docker setting: {e}")
        
        logger.debug(f"Set ScorerManager run_in_docker to {run_in_docker}")

    def stop_all_mcp_servers(self) -> None:
        """
        Stop all running MCP servers without unregistering scorers.

        This is useful for freeing up memory at the end of iterations while keeping
        scorer registrations intact. Servers will be restarted automatically when needed.
        """
        try:
            mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
            mcp_manager.stop_all_mcp_servers()
        except Exception as e:
            logger.error(f"Error stopping MCP servers: {e}")

    def clear_scorers(self) -> None:
        """Clear all registered scorers and their MCP counterparts."""
        # Clear regular scorers
        self._scorers.clear()
        self._scorer_metadata.clear()

        # Clear MCP scorers from McpScorerManager
        if self._mcp_scorers:
            try:
                mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)

                # Try to unregister modules gracefully first
                modules_to_clear = list(self._mcp_modules.keys())
                failed_modules = []

                for module_name in modules_to_clear:
                    try:
                        mcp_manager.unregister_module(module_name)
                        # logger.debug(f"Unregistered MCP module '{module_name}' during clear")
                    except Exception as e:
                        logger.warning(f"Error unregistering MCP module '{module_name}': {e}")
                        failed_modules.append(module_name)

                # If some modules failed to unregister gracefully, use force clear
                if failed_modules:
                    logger.warning(f"Some modules failed to unregister gracefully, using force clear")
                    try:
                        mcp_manager.clear_all_scorers()
                    except Exception as e:
                        logger.error(f"Error during force clear of MCP scorers: {e}")

                logger.debug(f"Cleared {len(self._mcp_scorers)} MCP scorers from {len(modules_to_clear)} modules")
            except Exception as e:
                logger.warning(f"Error clearing MCP scorers: {e}")
                # Fallback: try force clear
                try:
                    mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
                    mcp_manager.clear_all_scorers()
                    logger.debug("Used fallback force clear for MCP scorers")
                except Exception as e2:
                    logger.error(f"Fallback force clear also failed: {e2}")

        # Clear MCP tracking
        self._mcp_scorers.clear()
        self._mcp_modules.clear()

        logger.debug("Cleared all registered scorers and MCP scorers")
    
    def register_mcp_module(self, module_path: str, serializer_name: str) -> int:
        """
        Register MCP scorers from a module with the specified Serializer.

        Args:
            module_path: Path to the MCP module directory
            serializer_name: Name of the Serializer to use for serializing candidates

        Returns:
            Number of successfully registered MCP scorers

        Raises:
            ValueError: If serializer not found
            RuntimeError: If MCP module registration fails
        """
        # Get the serializer
        serializer_manager = SerializerManager()
        serializer = serializer_manager.get_serializer(serializer_name)
        if serializer is None:
            raise ValueError(f"Serializer '{serializer_name}' not found")
        
        # Register the module with McpScorerManager
        mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
        success = mcp_manager.add_module(module_path)
        if not success:
            raise RuntimeError(f"Failed to add MCP module at path: {module_path}")
        
        # Get all scorers from the added module
        all_scorers = mcp_manager.get_scorers()
        
        # Filter scorers that belong to this module
        from pathlib import Path
        module_name = Path(module_path).name
        registered_count = 0
        
        for scorer_name in all_scorers:
            scorer_info = mcp_manager.get_scorer_info(scorer_name)
            if scorer_info and scorer_info['module'] == module_name:
                # Create wrapper function for this MCP scorer
                def create_mcp_scorer_wrapper(mcp_scorer_name: str, mcp_serializer, scorer_type: str):
                    async def mcp_scorer_wrapper(*args, force_evaluation: bool = False, **kwargs):
                        """Async wrapper function that accepts Population or List[Candidate] and calls MCP scorer."""
                        is_population_wise = (scorer_type == "population-wise")
                        # Handle Population input by extracting candidates
                        # The first arg is the data (Population or List[Candidate])
                        if len(args) > 0:
                            first_arg = args[0]
                            if isinstance(first_arg, Population):
                                if first_arg.is_empty:
                                    raise ValueError(f"Population is empty for MCP scorer '{mcp_scorer_name}'")

                                if is_population_wise:
                                    # For population scorers with Population input
                                    if not force_evaluation and mcp_scorer_name in first_arg.scores:
                                        # Return cached score
                                        logger.debug(f"Using cached score for population MCP scorer '{mcp_scorer_name}'")
                                        return first_arg.scores[mcp_scorer_name]
                                    else:
                                        # Call MCP scorer (await async call)
                                        candidates = first_arg.candidates
                                        serialized_samples = mcp_serializer.serialize_batch(candidates)
                                        result = await mcp_manager.call_scorer(mcp_scorer_name, serialized_samples, **kwargs)
                                        result = convert_filter_result(result, scorer_type)
                                        logger.debug(f"Computed score for population MCP scorer '{mcp_scorer_name}': {result}")
                                        return result
                                else:
                                    # For regular scorers with Population input, extract candidates
                                    candidates = first_arg.candidates
                                    
                                    # Check each candidate for cached scores
                                    if not force_evaluation:
                                        cached_scores = []
                                        uncached_candidates = []
                                        uncached_indices = []
                                        
                                        for i, candidate in enumerate(candidates):
                                            if mcp_scorer_name in candidate.scores:
                                                cached_scores.append((i, candidate.scores[mcp_scorer_name]))
                                            else:
                                                uncached_candidates.append(candidate)
                                                uncached_indices.append(i)
                                        
                                        if uncached_candidates:
                                            # Compute scores for uncached candidates
                                            serialized_samples = mcp_serializer.serialize_batch(uncached_candidates)
                                            new_scores = await mcp_manager.call_scorer(mcp_scorer_name, serialized_samples, **kwargs)
                                            new_scores = convert_filter_result(new_scores, scorer_type)

                                            # Combine cached and new scores in original order
                                            result = [None] * len(candidates)
                                            for i, score in cached_scores:
                                                result[i] = score
                                            for i, score in zip(uncached_indices, new_scores):
                                                result[i] = score

                                            # logger.debug(f"Used {len(cached_scores)} cached scores and computed {len(new_scores)} new scores for MCP scorer '{mcp_scorer_name}'")
                                            return result
                                        else:
                                            # All candidates have cached scores
                                            result = [candidate.scores[mcp_scorer_name] for candidate in candidates]
                                            # logger.debug(f"Used all cached scores for MCP scorer '{mcp_scorer_name}'")
                                            return result
                                    else:
                                        # Force evaluation: compute all scores
                                        serialized_samples = mcp_serializer.serialize_batch(candidates)
                                        result = await mcp_manager.call_scorer(mcp_scorer_name, serialized_samples, **kwargs)
                                        result = convert_filter_result(result, scorer_type)
                                        logger.debug(f"Force evaluated scores for MCP scorer '{mcp_scorer_name}'")
                                        return result
                            elif isinstance(first_arg, list) and first_arg and isinstance(first_arg[0], Candidate):
                                # Handle List[Candidate] input
                                candidates = first_arg
                                if len(candidates) == 0:
                                    raise ValueError(f"Candidate list is empty for MCP scorer '{mcp_scorer_name}'")
                                
                                if is_population_wise:
                                    # For population scorers with List[Candidate] input, always call scorer
                                    # (no storage mechanism for lists)
                                    logger.debug(f"Calling population MCP scorer '{mcp_scorer_name}' on candidate list (no caching available)")
                                    serialized_samples = mcp_serializer.serialize_batch(candidates)
                                    result = await mcp_manager.call_scorer(mcp_scorer_name, serialized_samples, **kwargs)
                                    return convert_filter_result(result, scorer_type)
                                else:
                                    # For regular scorers with List[Candidate] input
                                    if not force_evaluation:
                                        cached_scores = []
                                        uncached_candidates = []
                                        uncached_indices = []
                                        
                                        for i, candidate in enumerate(candidates):
                                            if mcp_scorer_name in candidate.scores:
                                                cached_scores.append((i, candidate.scores[mcp_scorer_name]))
                                            else:
                                                uncached_candidates.append(candidate)
                                                uncached_indices.append(i)
                                        
                                        if uncached_candidates:
                                            # Compute scores for uncached candidates
                                            serialized_samples = mcp_serializer.serialize_batch(uncached_candidates)
                                            new_scores = await mcp_manager.call_scorer(mcp_scorer_name, serialized_samples, **kwargs)
                                            new_scores = convert_filter_result(new_scores, scorer_type)

                                            # Combine cached and new scores in original order
                                            result = [None] * len(candidates)
                                            for i, score in cached_scores:
                                                result[i] = score
                                            for i, score in zip(uncached_indices, new_scores):
                                                result[i] = score

                                            # logger.debug(f"Used {len(cached_scores)} cached scores and computed {len(new_scores)} new scores for MCP scorer '{mcp_scorer_name}'")
                                            return result
                                        else:
                                            # All candidates have cached scores
                                            result = [candidate.scores[mcp_scorer_name] for candidate in candidates]
                                            # logger.debug(f"Used all cached scores for MCP scorer '{mcp_scorer_name}'")
                                            return result
                                    else:
                                        # Force evaluation: compute all scores
                                        serialized_samples = mcp_serializer.serialize_batch(candidates)
                                        result = await mcp_manager.call_scorer(mcp_scorer_name, serialized_samples, **kwargs)
                                        result = convert_filter_result(result, scorer_type)
                                        logger.debug(f"Force evaluated scores for MCP scorer '{mcp_scorer_name}'")
                                        return result
                            else:
                                # Forbid other input types
                                raise ValueError(f"MCP scorer '{mcp_scorer_name}' must accept Population or List[Candidate] as input.")
                        else:
                            # Forbid other input types
                            raise ValueError(f"MCP scorer '{mcp_scorer_name}' must accept Population or List[Candidate] as input.")
                    
                    return mcp_scorer_wrapper
                
                # Create the wrapper function
                # Get type from scorer_info, with fallback to population_wise for backward compatibility
                scorer_type = scorer_info.get('type')
                if scorer_type is None:
                    # Fallback to population_wise for backward compatibility
                    is_population_wise = scorer_info.get('population_wise', False)
                    scorer_type = "population-wise" if is_population_wise else "candidate-wise"

                wrapper_func = create_mcp_scorer_wrapper(scorer_name, serializer, scorer_type)

                # Register the wrapper function as a regular scorer
                self.register_scorer(
                    scorer_func=wrapper_func,
                    name=scorer_name,
                    metadata={
                        'description': scorer_info.get('description', ''),
                        'mcp_module': module_name,
                        'serializer': serializer_name,
                        'original_metadata': scorer_info.get('metadata', {})
                    },
                    type=scorer_type,
                    is_mcp_scorer=True,
                    mcp_module_name=module_name
                )
                
                registered_count += 1
        
        return registered_count
    
    @classmethod
    def reset_instance(cls) -> None:
        """
        Reset the singleton instance (useful for testing).
        
        This method should be used with caution as it will clear all registered scorers
        and reset the MCP scorer manager.
        """
        if cls._instance is not None:
            # Clear all scorers (including MCP scorers)
            cls._instance.clear_scorers()
            
            # Reset the MCP scorer manager singleton
            try:
                McpScorerManager.reset_instance()
                logger.debug("Reset MCP scorer manager instance")
            except Exception as e:
                logger.warning(f"Error resetting MCP scorer manager: {e}")
            
            cls._instance = None
            cls._initialized = False
            logger.debug("ScorerManager instance reset")


def register_scorer(
    name: Optional[str] = None,
    population_wise: bool = False,
    type: Optional[str] = None,
    **kwargs
):
    """
    Decorator for marking a function as a scorer and automatically registering it.

    The decorated function must accept a List[Candidate] and return List[Optional[float]],
    or if population_wise=True (or type="population-wise"), accept a List[Candidate] and return Optional[float].
    For class methods, the function should have 'self' as the first parameter.

    The registered scorer function will accept either List[Candidate] or Population as input.
    If Population is passed, it will extract population.candidates and pass them to the original function.

    The function is automatically registered with the ScorerManager when the decorator is applied.

    Args:
        name: Name of the scorer (defaults to function name)
        population_wise: (Deprecated) Whether this scorer is a population objective scorer. Use 'type' instead.
        type: Type of scorer - "candidate-wise", "population-wise", or "filter". If not provided, inferred from population_wise.
        **kwargs: Additional metadata for the scorer (e.g., description, category, etc.)
    """
    def decorator(func: Callable):
        sig = inspect.signature(func)
        params = list(sig.parameters.values())
        is_class_method = (len(params) == 2 and params[0].name == 'self')
        is_function = (len(params) == 1)
        
        def is_valid_annotation(ann, expected_types):
            """Check if annotation matches any of the expected types or is a Union containing them."""
            if ann == inspect._empty:
                return True
            if ann in expected_types:
                return True
            # Check for Union types
            origin = get_origin(ann)
            if origin is Union:
                union_args = get_args(ann)
                return any(arg in expected_types for arg in union_args)
            return False
        
        # Validate signature - original function must accept List[Candidate]
        valid = False
        if is_function:
            ann = params[0].annotation
            if is_valid_annotation(ann, (List[Candidate],)):
                valid = True
        elif is_class_method:
            ann = params[1].annotation
            if is_valid_annotation(ann, (List[Candidate],)):
                valid = True
        if not valid:
            raise ValueError(f"Scorer '{func.__name__}' must accept List[Candidate] as input.")
        
        # Determine scorer type
        if type is not None:
            if type not in ("candidate-wise", "population-wise", "filter"):
                raise ValueError(f"Invalid scorer type '{type}'. Must be 'candidate-wise', 'population-wise', or 'filter'.")
            scorer_type = type
            is_population_wise = (type == "population-wise")
        else:
            # Infer from population_wise for backward compatibility
            scorer_type = "population-wise" if population_wise else "candidate-wise"
            is_population_wise = population_wise

        scorer_name = name if name is not None else func.__name__
        func._is_scorer = True
        func._scorer_name = scorer_name
        func._scorer_metadata = kwargs
        func._population_wise = is_population_wise
        func._type = scorer_type
        
        @wraps(func)
        def wrapper(*args, force_evaluation: bool = False, **kwargs):
            # Handle Population input by extracting candidates
            # For class methods, the first arg is 'self', second arg is data
            # For regular functions, the first arg is data
            data_arg_index = 1 if is_class_method else 0
            
            if len(args) > data_arg_index:
                first_arg = args[data_arg_index]
                if isinstance(first_arg, Population):
                    if first_arg.is_empty:
                        raise ValueError(f"Population is empty for scorer '{scorer_name}'")
                    if is_population_wise:
                        # For population scorers with Population input
                        if not force_evaluation and scorer_name in first_arg.scores:
                            # Return cached score
                            logger.debug(f"Using cached score for population scorer '{scorer_name}'")
                            return first_arg.scores[scorer_name]
                        else:
                            # Call scorer
                            if is_class_method:
                                result = func(args[0], first_arg.candidates, **kwargs)
                            else:
                                result = func(first_arg.candidates, **kwargs)
                            result = convert_filter_result(result, scorer_type)
                            logger.debug(f"Computed score for population scorer '{scorer_name}': {result}")
                            return result
                    else:
                        # For regular scorers with Population input, extract candidates
                        candidates = first_arg.candidates
                        
                        # Check each candidate for cached scores
                        if not force_evaluation:
                            cached_scores = []
                            uncached_candidates = []
                            uncached_indices = []
                            
                            for i, candidate in enumerate(candidates):
                                if scorer_name in candidate.scores:
                                    cached_scores.append((i, candidate.scores[scorer_name]))
                                else:
                                    uncached_candidates.append(candidate)
                                    uncached_indices.append(i)
                            
                            if uncached_candidates:
                                # Compute scores for uncached candidates
                                if is_class_method:
                                    new_scores = func(args[0], uncached_candidates, **kwargs)
                                else:
                                    new_scores = func(uncached_candidates, **kwargs)
                                new_scores = convert_filter_result(new_scores, scorer_type)

                                # Combine cached and new scores in original order
                                result = [None] * len(candidates)
                                for i, score in cached_scores:
                                    result[i] = score
                                for i, score in zip(uncached_indices, new_scores):
                                    result[i] = score

                                # logger.debug(f"Used {len(cached_scores)} cached scores and computed {len(new_scores)} new scores for scorer '{scorer_name}'")
                                return result
                            else:
                                # All candidates have cached scores
                                result = [candidate.scores[scorer_name] for candidate in candidates]
                                # logger.debug(f"Used all cached scores for scorer '{scorer_name}'")
                                return result
                        else:
                            # Force evaluation: compute all scores
                            if is_class_method:
                                result = func(args[0], candidates, **kwargs)
                            else:
                                result = func(candidates, **kwargs)
                            result = convert_filter_result(result, scorer_type)

                            logger.debug(f"Force evaluated and cached scores for scorer '{scorer_name}'")
                            return result
                elif isinstance(first_arg, list) and first_arg and isinstance(first_arg[0], Candidate):
                    # Handle List[Candidate] input
                    candidates = first_arg
                    if len(candidates) == 0:
                        raise ValueError(f"Candidate list is empty for scorer '{scorer_name}'")

                    if is_population_wise:
                        # For population scorers with List[Candidate] input, always call scorer
                        # (no storage mechanism for lists)
                        logger.debug(f"Calling population scorer '{scorer_name}' on candidate list (no caching available)")
                        if is_class_method:
                            result = func(args[0], candidates, **kwargs)
                        else:
                            result = func(candidates, **kwargs)
                        return convert_filter_result(result, scorer_type)
                    else:
                        # For regular scorers with List[Candidate] input
                        if not force_evaluation:
                            cached_scores = []
                            uncached_candidates = []
                            uncached_indices = []
                            
                            for i, candidate in enumerate(candidates):
                                if scorer_name in candidate.scores:
                                    cached_scores.append((i, candidate.scores[scorer_name]))
                                else:
                                    uncached_candidates.append(candidate)
                                    uncached_indices.append(i)
                            
                            if uncached_candidates:
                                # Compute scores for uncached candidates
                                if is_class_method:
                                    new_scores = func(args[0], uncached_candidates, **kwargs)
                                else:
                                    new_scores = func(uncached_candidates, **kwargs)
                                new_scores = convert_filter_result(new_scores, scorer_type)

                                # Combine cached and new scores in original order
                                result = [None] * len(candidates)
                                for i, score in cached_scores:
                                    result[i] = score
                                for i, score in zip(uncached_indices, new_scores):
                                    result[i] = score

                                # logger.debug(f"Used {len(cached_scores)} cached scores and computed {len(new_scores)} new scores for scorer '{scorer_name}'")
                                return result
                            else:
                                # All candidates have cached scores
                                result = [candidate.scores[scorer_name] for candidate in candidates]
                                # logger.debug(f"Used all cached scores for scorer '{scorer_name}'")
                                return result
                        else:
                            # Force evaluation: compute all scores
                            if is_class_method:
                                result = func(args[0], candidates, **kwargs)
                            else:
                                result = func(candidates, **kwargs)
                            result = convert_filter_result(result, scorer_type)

                            logger.debug(f"Force evaluated and cached scores for scorer '{scorer_name}'")
                            return result
                else:
                    # Forbid other input types
                    raise ValueError(f"Scorer '{scorer_name}' must accept Population or List[Candidate] as input.")
            
            else:
                # Forbid other input types
                raise ValueError(f"Scorer '{scorer_name}' must accept Population or List[Candidate] as input.")
        
        wrapper._is_scorer = True
        wrapper._scorer_name = scorer_name
        wrapper._scorer_metadata = kwargs
        wrapper._population_wise = is_population_wise
        wrapper._type = scorer_type

        if is_function:
            ScorerManager().register_scorer(
                scorer_func=wrapper,
                name=scorer_name,
                metadata=kwargs,
                type=scorer_type
            )
        return wrapper
    return decorator


def register_scorer_class(cls: Type):
    """
    Decorator for making a class a singleton and automatically registering its scorer methods.
    
    This decorator ensures that:
    1. The class becomes a singleton
    2. The class can be instantiated without arguments
    3. Individual methods decorated with @scorer are automatically registered when the class is instantiated
    
    Args:
        cls: The class to decorate
    
    Example:
        @scorer_class
        class MyScorers:
            @scorer(name="score_a", description="Score A")
            def score_a(self, candidates: List[Candidate]) -> List[Optional[float]]:
                return [candidate.get_score("metric_a", 0.0) for candidate in candidates]
            
            @scorer(name="score_b", description="Score B")
            def score_b(self, candidates: List[Candidate]) -> List[Optional[float]]:
                return [candidate.get_score("metric_b", 0.0) for candidate in candidates]
        
        # Methods are automatically registered when class is instantiated
        # No manual registration needed
    """
    # Store original __init__ method
    original_init = cls.__init__
    
    # Check if the class has required arguments
    sig = inspect.signature(original_init)
    required_params = [param for param in sig.parameters.values() 
                      if param.default == inspect.Parameter.empty and param.name != 'self'
                      and param.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)]
    
    if required_params:
        raise ValueError(f"Class '{cls.__name__}' has required arguments: {[p.name for p in required_params]}. "
                        f"Scorer classes must be instantiable without arguments.")
    
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
            
            # Automatically register all scorer methods
            self._register_scorer_methods()
    
    def _register_scorer_methods(self):
        """Register all methods decorated with @scorer."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, '_is_scorer') and attr._is_scorer:
                # Bind the method to the instance
                bound_method = attr.__get__(self, type(self))
                
                # Register the bound method
                ScorerManager().register_scorer(
                    name=attr._scorer_name,
                    scorer_func=bound_method,
                    metadata=attr._scorer_metadata
                )
    
    # Replace the class methods
    cls.__new__ = __new__
    cls.__init__ = __init__
    cls._register_scorer_methods = _register_scorer_methods
    
    cls()
    return cls


# Convenience functions for accessing the singleton scorer manager
def get_scorer(name: str, case_sensitive: bool = False) -> Optional[Callable[[List[Candidate]], List[Optional[float]]]]:
    """
    Get a scorer function by name.
    
    Scorers are automatically registered when using the @scorer and @scorer_class decorators.
    """
    return ScorerManager().get_scorer(name, case_sensitive=case_sensitive)


def list_scorers() -> List[str]:
    """
    List all registered scorer names.

    Scorers are automatically registered when using the @scorer and @scorer_class decorators.
    """
    return ScorerManager().list_scorers()


def get_scorer_metadata(name: str) -> Optional[Dict[str, Any]]:
    """
    Get metadata for a scorer by name.

    Args:
        name: Name of the scorer

    Returns:
        Scorer metadata dict or None if not found
    """
    return ScorerManager().get_scorer_metadata(name)


def clear_scorers() -> None:
    """
    Clear all registered scorers.
    
    Note: This will clear all scorers, including those automatically registered
    by the @scorer and @scorer_class decorators.
    """
    ScorerManager().clear_scorers()


def reset_scorer_manager() -> None:
    """
    Reset the singleton ScorerManager instance (useful for testing).
    
    This will clear all registered scorers and reset the manager to its initial state.
    """
    ScorerManager.reset_instance()


def register_mcp_module(module_path: str, serializer_name: str) -> int:
    """
    Register MCP scorers from a module with the specified Serializer.
    
    Args:
        module_path: Path to the MCP module directory
        serializer_name: Name of the Serializer to use for serializing candidates
        
    Returns:
        Number of successfully registered MCP scorers
        
    Raises:
        ValueError: If serializer not found
        RuntimeError: If MCP module registration fails
    """
    return ScorerManager().register_mcp_module(module_path, serializer_name)