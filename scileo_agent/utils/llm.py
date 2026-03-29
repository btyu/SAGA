"""
LiteLLM integration utilities for the SciLeo Agent framework.

This module provides a unified interface for calling various LLM providers
through LiteLLM, with support for retries, caching, error handling, and
YAML-based configuration management.
"""

import os
from typing import Dict, Any, List, Optional
from pathlib import Path
import asyncio
from litellm import completion, acompletion
import time
import yaml
from copy import deepcopy

from .logging import get_logger

logger = get_logger()


class BaseLLMConfig:
    """Configuration for LLM calls."""
    
    def __init__(
        self,
        provider: str,
        model: str,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs
    ):
        self.provider = provider
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.extra_args = kwargs
    
    def get_call_args(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for litellm."""
        config = {
            "model": f'{self.provider}/{self.model}',
        }
        
        # Add extra arguments
        config.update(self.extra_args)

        return config


class YAMLLLMConfig:
    """Configuration manager for YAML-based LLM setup."""
    
    def __init__(
        self,
        models_file: str = "llm_configs/models.yaml",
        credentials_file: str = "llm_configs/credentials.yaml"
    ):
        self.models_file = Path(models_file)
        self.credentials_file = Path(credentials_file)
        self.models = {}
        self.credentials = {}
        self.load_configurations()
    
    def load_configurations(self) -> None:
        """Load models and credentials from YAML files."""

        # Load models configuration
        if self.models_file.exists():
            with open(self.models_file, 'r') as f:
                self.models = yaml.safe_load(f) or {}
            # logger.debug(f"Loaded {len(self.models)} models from {self.models_file}")
        else:
            logger.warning(f"Models file not found: {self.models_file}")
            raise FileNotFoundError(f"Models file not found: {self.models_file}")
        
        # Load credentials configuration
        if self.credentials_file.exists():
            with open(self.credentials_file, 'r') as f:
                self.credentials = yaml.safe_load(f) or {}
            # logger.debug(f"Loaded credentials for {len(self.credentials)} providers from {self.credentials_file}")
        else:
            logger.warning(f"Credentials file not found: {self.credentials_file}")
            raise FileNotFoundError(f"Credentials file not found: {self.credentials_file}")
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific model."""
        return self.models.get(model_name)
    
    def get_credentials(self, credential_tag: str) -> Optional[Dict[str, Any]]:
        """Get credentials for a specific provider."""
        return self.credentials.get(credential_tag)
    
    def list_available_models(self) -> List[str]:
        """List all available model names."""
        return list(self.models.keys())
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """Get all models for a specific provider."""
        return [
            name for name, config in self.models.items()
            if config.get("provider") == provider
        ]
    
    def create_llm_config(self, model_name: str, **override_args) -> Optional[BaseLLMConfig]:
        """Create an LLMConfig from YAML configuration."""
        model_config = self.get_model_config(model_name)
        if not model_config:
            logger.error(f"Model '{model_name}' not found in configuration")
            return None
        
        # Get credentials if specified
        credentials = {}
        credential_tag = model_config.get("credentials")
        if credential_tag:
            cred_config = self.get_credentials(credential_tag)
            if cred_config:
                credentials = cred_config.copy()
                # Replace placeholder values with environment variables
                for key, value in credentials.items():
                    if isinstance(value, str) and value.isupper():
                        credentials[key] = os.getenv(value, value)
        
        # Merge configurations
        config_args = {
            "provider": model_config.get("provider"),
            "model": model_config.get("model", model_name),
            "max_retries": 3,
            "retry_delay": 1.0,
        }
        
        # Add credentials
        config_args.update(credentials)
        
        # Add call arguments from model config
        call_args = model_config.get("__call_args", {})
        config_args.update(call_args)
        
        # Apply overrides
        config_args.update(override_args)
        
        return BaseLLMConfig(**config_args)


class LLMClient:
    """
    Client for making LLM calls with retry logic and error handling.
    Supports dynamic model switching when yaml_config is provided.
    """

    def __init__(self, config: BaseLLMConfig, yaml_config: Optional['YAMLLLMConfig'] = None):
        self.config = config
        self.yaml_config = yaml_config
        self.stats = {}
        self.responses = []

    def _resolve_model_config(self, model_name: Optional[str] = None, **kwargs) -> BaseLLMConfig:
        """
        Resolve the model configuration to use for a call.

        Args:
            model_name: Optional model name from YAML config to switch to
            **kwargs: Additional override arguments

        Returns:
            BaseLLMConfig to use for the call
        """
        if model_name and self.yaml_config:
            # Create new config from YAML model_name
            config = self.yaml_config.create_llm_config(model_name, **kwargs)
            if not config:
                raise ValueError(f"Model '{model_name}' not found in YAML configuration")
            return config
        else:
            # Use default config
            return self.config

    def _retry_call(self, func, *args, **kwargs):
        """Execute a function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}")
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"LLM call failed after {self.config.max_retries + 1} attempts: {e}")
                    raise last_exception

        raise last_exception

    async def _retry_call_async(self, func, *args, **kwargs):
        """Execute an async function with retry logic."""
        last_exception = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.config.max_retries:
                    logger.warning(f"LLM call failed (attempt {attempt + 1}/{self.config.max_retries + 1}): {e}")
                    await asyncio.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    logger.error(f"LLM call failed after {self.config.max_retries + 1} attempts: {e}")
                    raise last_exception

        raise last_exception
    
    def call(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a completion call to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: Optional model name from YAML config to use for this call
            **kwargs: Additional parameters to override config

        Returns:
            Dictionary containing the response and metadata
        """
        # logger.debug(f"LLM call: {json.dumps(messages, indent=2)}")

        def _make_call():
            # Resolve which model config to use
            resolved_config = self._resolve_model_config(model_name, **kwargs)

            # Merge config with kwargs
            call_config = resolved_config.get_call_args()
            call_config.update(kwargs)

            actual_model = call_config.get("model", "unknown")
            if actual_model not in self.stats:
                self.stats[actual_model] = {"call_count": 0, "total_tokens": 0, "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            model_name_stats = self.stats[actual_model]
            
            # Make the API call
            response = completion(
                messages=messages,
                **call_config
            )
            
            # Update statistics
            model_name_stats['call_count'] += 1
            response_total_tokens = None
            response_input_tokens = None
            response_cache_read_input_tokens = 0
            response_cache_creation_input_tokens = 0
            response_output_tokens = None
            if hasattr(response, 'usage') and response.usage:
                response_total_tokens = response.usage.total_tokens
                response_input_tokens = response.usage.prompt_tokens
                response_output_tokens = response.usage.completion_tokens
                prompt_tokens_details = response.usage.prompt_tokens_details
                if prompt_tokens_details:
                    response_cache_read_input_tokens = getattr(prompt_tokens_details, "cached_tokens", 0)
                else:
                    response_cache_read_input_tokens = 0
                response_cache_creation_input_tokens = response.usage.get("cache_creation_input_tokens", 0)

                # This is to fix the inconsistent counting of LiteLLM
                if response_input_tokens >= response_cache_read_input_tokens:
                    response_input_tokens -= response_cache_read_input_tokens
                # After this fix, the input_tokens should not include cache_read and cache_creation
                
                if response_total_tokens != (response_input_tokens + response_cache_read_input_tokens + response_cache_creation_input_tokens + response_output_tokens):
                    logger.warning(f"Total tokens mismatch for '{actual_model}': {response.usage}. Please upgrade LiteLLM with `pip install \"litellm >=1.78.0\".")

                model_name_stats['total_tokens'] += response_total_tokens
                model_name_stats['input_tokens'] += response_input_tokens
                model_name_stats['cache_read_input_tokens'] += response_cache_read_input_tokens
                model_name_stats['cache_creation_input_tokens'] += response_cache_creation_input_tokens
                model_name_stats['output_tokens'] += response_output_tokens
            else:
                logger.warning(f"Response does not contain usage information for '{actual_model}'. Skipped token counting.")

            response_cost = response._hidden_params.get('response_cost', None)
            if response_cost is None:
                logger.warning(f"Response does not contain cost information for '{actual_model}'.")
            else:
                model_name_stats['cost'] += response_cost

            response_dict = response.model_dump()
            response_dict['model_name'] = actual_model
            response_dict['_hidden_params'] = response._hidden_params
            response_dict['messages'] = deepcopy(messages)
            self.responses.append(response_dict)
            
            # Extract response content
            content = response.choices[0].message.content
            
            return {
                "content": content,
                "response": response,
                "model": response.model,
                "total_tokens": response_total_tokens, 
                "input_tokens": response_input_tokens, 
                "cache_read_input_tokens": response_cache_read_input_tokens,
                "cache_creation_input_tokens": response_cache_creation_input_tokens,
                "output_tokens": response_output_tokens,
                "cost": response_cost
            }
        
        return self._retry_call(_make_call)

    async def call_async(
        self,
        messages: List[Dict[str, str]],
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an async completion call to the LLM.

        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model_name: Optional model name from YAML config to use for this call
            **kwargs: Additional parameters to override config

        Returns:
            Dictionary containing the response and metadata
        """
        # logger.debug(f"LLM async call: {json.dumps(messages, indent=2)}")

        async def _make_call():
            # Resolve which model config to use
            resolved_config = self._resolve_model_config(model_name, **kwargs)

            # Merge config with kwargs
            call_config = resolved_config.get_call_args()
            call_config.update(kwargs)

            actual_model = call_config.get("model", "unknown")
            if actual_model not in self.stats:
                self.stats[actual_model] = {"call_count": 0, "total_tokens": 0, "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            model_name_stats = self.stats[actual_model]

            # Make the async API call
            response = await acompletion(
                messages=messages,
                **call_config
            )

            # Update statistics
            model_name_stats['call_count'] += 1
            response_total_tokens = None
            response_input_tokens = None
            response_cache_read_input_tokens = 0
            response_cache_creation_input_tokens = 0
            response_output_tokens = None
            if hasattr(response, 'usage') and response.usage:
                response_total_tokens = response.usage.total_tokens
                response_input_tokens = response.usage.prompt_tokens
                response_output_tokens = response.usage.completion_tokens
                prompt_tokens_details = response.usage.prompt_tokens_details
                if prompt_tokens_details:
                    response_cache_read_input_tokens = getattr(prompt_tokens_details, "cached_tokens", 0)
                else:
                    response_cache_read_input_tokens = 0
                response_cache_creation_input_tokens = response.usage.get("cache_creation_input_tokens", 0)

                # This is to fix the inconsistent counting of LiteLLM
                if response_input_tokens >= response_cache_read_input_tokens:
                    response_input_tokens -= response_cache_read_input_tokens
                # After this fix, the input_tokens should not include cache_read and cache_creation

                if response_total_tokens != (response_input_tokens + response_cache_read_input_tokens + response_cache_creation_input_tokens + response_output_tokens):
                    logger.warning(f"Total tokens mismatch for '{actual_model}': {response.usage}. Please upgrade LiteLLM with `pip install \"litellm >=1.78.0\".")

                model_name_stats['total_tokens'] += response_total_tokens
                model_name_stats['input_tokens'] += response_input_tokens
                model_name_stats['cache_read_input_tokens'] += response_cache_read_input_tokens
                model_name_stats['cache_creation_input_tokens'] += response_cache_creation_input_tokens
                model_name_stats['output_tokens'] += response_output_tokens
            else:
                logger.warning(f"Response does not contain usage information for '{actual_model}'. Skipped token counting.")

            response_cost = response._hidden_params.get('response_cost', None)
            if response_cost is None:
                logger.warning(f"Response does not contain cost information for '{actual_model}'.")
            else:
                model_name_stats['cost'] += response_cost

            response_dict = response.model_dump()
            response_dict['model_name'] = actual_model
            response_dict['_hidden_params'] = response._hidden_params
            response_dict['messages'] = deepcopy(messages)
            self.responses.append(response_dict)

            # Extract response content
            content = response.choices[0].message.content

            return {
                "content": content,
                "response": response,
                "model": response.model,
                "total_tokens": response_total_tokens,
                "input_tokens": response_input_tokens,
                "cache_read_input_tokens": response_cache_read_input_tokens,
                "cache_creation_input_tokens": response_cache_creation_input_tokens,
                "output_tokens": response_output_tokens,
                "cost": response_cost
            }

        return await self._retry_call_async(_make_call)

    def call_with_prompt(
        self,
        user_prompt: str,
        system_prompt: str = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make a call with system prompt and user message.

        Args:
            user_prompt: User prompt
            system_prompt: System prompt to set context
            model_name: Optional model name from YAML config to use for this call
            **kwargs: Additional parameters

        Returns:
            Dictionary containing the response and metadata
        """
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return self.call(messages, model_name=model_name, **kwargs)

    async def call_with_prompt_async(
        self,
        user_prompt: str,
        system_prompt: str = None,
        model_name: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Make an async call with system prompt and user message.

        Args:
            user_prompt: User prompt
            system_prompt: System prompt to set context
            model_name: Optional model name from YAML config to use for this call
            **kwargs: Additional parameters

        Returns:
            Dictionary containing the response and metadata
        """
        messages = [
            {"role": "user", "content": user_prompt}
        ]
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})

        return await self.call_async(messages, model_name=model_name, **kwargs)

    def get_stats(self) -> Dict[str, Any]:
        """Get client statistics."""
        return deepcopy(self.stats)
    
    def reset_stats(self) -> None:
        """Reset client statistics and responses."""
        self.stats = {}
        self.responses = []


class LLMFactory:
    """
    Factory for creating LLM clients with different configurations.
    Supports YAML-based configuration with required models and credentials files.
    Uses singleton pattern to ensure only one instance exists.
    """
    
    _instance = None
    _initialized = False
    
    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(LLMFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(
        self,
        models_file: str = "llm_configs/models.yaml",
        credentials_file: str = "llm_configs/credentials.yaml",
        force_initialize: bool = False
    ):
        # Only initialize if forced or if first time
        if self._initialized and not force_initialize:
            return
            
        # Store file paths
        self.models_file = models_file
        self.credentials_file = credentials_file
        
        # Initialize YAML configuration
        self.yaml_config = YAMLLLMConfig(models_file, credentials_file)
        
        self._initialized = True
    
    def create_client(
        self,
        model_name: str,
        **override_args
    ) -> LLMClient:
        """
        Create an LLM client from YAML configuration.
        Supports dynamic model switching through the model_name parameter in call methods.

        Args:
            model_name: Name of the model in YAML config (default model for this client)
            **override_args: Arguments to override from YAML config

        Returns:
            New LLMClient instance with yaml_config attached for model switching
        """
        config = self.yaml_config.create_llm_config(model_name, **override_args)
        if not config:
            raise ValueError(f"Failed to create configuration for model '{model_name}'")

        return LLMClient(config, yaml_config=self.yaml_config)
    
    def create_client_from_config(self, config: BaseLLMConfig) -> LLMClient:
        """
        Create an LLM client with programmatic configuration.
        
        Args:
            config: LLM configuration
            
        Returns:
            New LLMClient instance
        """
        return LLMClient(config)
    
    def list_available_models(self) -> List[str]:
        """List all available models from YAML configuration."""
        return self.yaml_config.list_available_models()
    
    def get_models_by_provider(self, provider: str) -> List[str]:
        """Get all models for a specific provider."""
        return self.yaml_config.get_models_by_provider(provider)


def list_available_models(models_file: str = "llm_configs/models.yaml", credentials_file: str = "llm_configs/credentials.yaml", force_initialize: bool = False) -> List[str]:
    """List all available models from YAML configuration."""
    llm_factory = LLMFactory(models_file, credentials_file, force_initialize)
    return llm_factory.list_available_models()


def create_client(model_name: str, models_file: str = "llm_configs/models.yaml", credentials_file: str = "llm_configs/credentials.yaml", force_initialize: bool = False, **override_args) -> LLMClient:
    """Create an LLM client from YAML configuration."""
    llm_factory = LLMFactory(models_file, credentials_file, force_initialize)
    return llm_factory.create_client(model_name, **override_args)


def get_models_by_provider(provider: str, models_file: str = "llm_configs/models.yaml", credentials_file: str = "llm_configs/credentials.yaml", force_initialize: bool = False) -> List[str]:
    """Get all models for a specific provider from YAML configuration."""
    llm_factory = LLMFactory(models_file, credentials_file, force_initialize)
    return llm_factory.get_models_by_provider(provider)


def list_providers(models_file: str = "llm_configs/models.yaml", credentials_file: str = "llm_configs/credentials.yaml", force_initialize: bool = False) -> List[str]:
    """List all available providers from YAML configuration."""
    llm_factory = LLMFactory(models_file, credentials_file, force_initialize)
    providers = set()
    for model_config in llm_factory.yaml_config.models.values():
        provider = model_config.get("provider")
        if provider:
            providers.add(provider)
    return list(providers)
