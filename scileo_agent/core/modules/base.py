"""
Abstract base classes for modules in the SciLeo Agent framework.

This module defines the abstract interfaces that all modules must implement,
providing a consistent API for the optimization orchestrator.

The framework uses five core modules:
- PlannerModule: Plans optimization objectives for each iteration
- ScorerCreatorModule: Creates/retrieves scoring functions for objectives  
- OptimizerModule: Runs optimization algorithms until convergence
- AnalyzerModule: Analyzes results and generates comprehensive reports
- KnowledgeManagerModule: Manages all data and knowledge storage
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Union, Callable, TYPE_CHECKING
from pydantic import Field

from ...utils.llm import LLMClient, create_client
from ...utils.logging import get_logger

if TYPE_CHECKING:
    from ..config import LLMConfig


class BaseModule(ABC):
    """
    Abstract base class for all modules in the optimization framework.
    
    This class defines the common interface and functionality that all modules
    must implement, ensuring consistent behavior across the framework.
    """
    
    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None, llm_config: Optional['LLMConfig'] = None):
        """
        Initialize the base module.
        
        Args:
            module_id: Unique identifier for this module instance
            config: Configuration parameters for the module
            llm_config: LLM configuration for this module (optional)
        """
        self.module_id = module_id
        self.config = config or {}
        self.llm_config = llm_config
        self.call_count = 0
        self.metadata = {}
        
        # Initialize LLM client if configuration is provided
        self.llm_client: Optional[LLMClient] = None
        if self.llm_config:
            try:
                self.llm_client = create_client(
                    model_name=self.llm_config.model_name,
                    models_file=self.llm_config.models_file,
                    credentials_file=self.llm_config.credentials_file,
                    max_retries=self.llm_config.max_retries,
                    retry_delay=self.llm_config.retry_delay,
                    **self.llm_config.config
                )
            except Exception as e:
                # Log the error but don't fail module initialization
                print(f"Warning: Failed to initialize LLM client for module {module_id}: {e}")
                self.llm_client = None
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """
        Main processing method that each module must implement.
        
        This method defines the core functionality of the module.
        """
        pass
    
    def has_llm(self) -> bool:
        """Check if this module has an LLM client available."""
        return self.llm_client is not None
    
    def call_llm(self, messages: List[Dict[str, str]], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make an LLM call if the module has an LLM client.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the LLM call

        Returns:
            LLM response or None if no LLM client is available
        """
        if not self.llm_client:
            return None

        return self.llm_client.call(messages, **kwargs)

    async def call_llm_async(self, messages: List[Dict[str, str]], **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make an async LLM call if the module has an LLM client.

        Args:
            messages: List of message dictionaries
            **kwargs: Additional parameters for the LLM call

        Returns:
            LLM response or None if no LLM client is available
        """
        if not self.llm_client:
            return None

        return await self.llm_client.call_async(messages, **kwargs)

    def call_llm_with_prompt(self, user_prompt: str, system_prompt: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make an LLM call with a system prompt and user message.

        Args:
            user_prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the LLM call

        Returns:
            LLM response or None if no LLM client is available
        """
        if not self.llm_client:
            return None

        return self.llm_client.call_with_prompt(user_prompt, system_prompt, **kwargs)

    async def call_llm_with_prompt_async(self, user_prompt: str, system_prompt: str = None, **kwargs) -> Optional[Dict[str, Any]]:
        """
        Make an async LLM call with a system prompt and user message.

        Args:
            user_prompt: User prompt
            system_prompt: Optional system prompt
            **kwargs: Additional parameters for the LLM call

        Returns:
            LLM response or None if no LLM client is available
        """
        if not self.llm_client:
            return None

        return await self.llm_client.call_with_prompt_async(user_prompt, system_prompt, **kwargs)

    def get_llm_stats(self) -> Optional[Dict[str, Any]]:
        """Get LLM usage statistics for this module."""
        if not self.llm_client:
            return None
            
        return self.llm_client.get_stats()
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the module."""
        status = {
            "module_id": self.module_id,
            "module_type": self.module_type,
            "module_name": self.module_name,
            "module_version": self.module_version,
            "module_class_name": self.__class__.__name__,
            "call_count": self.call_count,
            "has_llm": self.has_llm(),
            "llm_name": self.llm_config.model_name if self.llm_config else None,
            "metadata": self.metadata
        }
        
        # Add LLM stats if available
        llm_stats = self.get_llm_stats()
        if llm_stats:
            status["llm_stats"] = llm_stats
            
        return status
    
    def __str__(self) -> str:
        """Return a string representation of the module."""
        status = self.get_status()
        module_class_name = status["module_class_name"]
        # Create a clean representation without the class name in the parameters
        params = {k: v for k, v in status.items() if k != "module_class_name"}
        return f"{module_class_name}({', '.join([f'{k}={v}' for k, v in params.items()])})"
    
    def update_metadata(self, key: str, value: Any) -> None:
        """Update module metadata."""
        self.metadata[key] = value
    
    def _increment_call_count(self) -> None:
        """Increment the call counter."""
        self.call_count += 1
