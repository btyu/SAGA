"""
Configuration management for the SciLeo Agent framework.

This module provides comprehensive configuration management for the framework,
including:

- LLM configuration for each module
- Module-specific configurations
- Optimization parameters
- Framework-wide settings

The configuration system uses Pydantic models for validation and type safety,
and supports loading from files or environment variables.
"""

from typing import Dict, Any, Optional, List, Union
from pydantic import BaseModel, Field, ConfigDict
from pydantic_settings import BaseSettings
from pathlib import Path
import warnings
import json
from datetime import datetime

from .registry.module_registry import get_module_class
from ..__version__ import __version__ as framework_version


# Default to dev mode
DEV_DEFAULT = False


class LLMConfig(BaseModel):
    """Configuration for LLM clients."""

    models_file: str = Field(default="llm_configs/models.yaml", description="Path to models YAML file")
    credentials_file: str = Field(default="llm_configs/credentials.yaml", description="Path to credentials YAML file")
    
    model_name: str = Field(default="openai/gpt-4.1-nano-2025-04-14", description="LLM model name")
    
    max_retries: int = Field(default=3, ge=0, description="Maximum number of retries for failed calls")
    retry_delay: float = Field(default=1.0, ge=0.0, description="Delay between retries in seconds")

    config: Dict[str, Any] = Field(default_factory=dict, description="LLM configuration")

    def __init__(self, **data):
        # Auto-detect defined fields from the model class
        defined_fields = set(self.__class__.model_fields.keys())
        
        # Extract config if provided, otherwise start with empty dict
        config = data.pop("config", {})
        
        # Move all undefined items into config
        undefined_items = {k: v for k, v in data.items() if k not in defined_fields}
        config.update(undefined_items)
        
        # Remove undefined items from data
        for key in undefined_items:
            data.pop(key)
        
        # Add config back to data
        data['config'] = config
        
        super().__init__(**data)

    
    def __str__(self) -> str:
        """Return a nicely formatted string representation."""
        indent = "    "
        lines = ["LLMConfig("]
        
        # Add basic fields
        basic_fields = [
            f"models_file='{self.models_file}'",
            f"credentials_file='{self.credentials_file}'",
            f"model_name='{self.model_name}'",
            f"max_retries={self.max_retries}",
            f"retry_delay={self.retry_delay}"
        ]
        
        # Add fields with indentation
        for field in basic_fields:
            lines.append(f"{indent}{field},")
        
        # Add config if present
        if self.config:
            lines.append(f"{indent}config={self.config},")
        
        lines.append(")")
        return "\n".join(lines)


class ModuleConfig(BaseModel):
    """Base configuration for modules."""
    
    module_id: Optional[str] = Field(default=None, description="Unique identifier for the module")
    module_type: str = Field(..., description="Type of module")
    module_name: str = Field(..., description="Name of the module")
    module_version: Optional[str] = Field(default=None, description="Version of the module")
    # enabled: bool = Field(default=True, description="Whether the module is enabled")
    config: Dict[str, Any] = Field(default_factory=dict, description="Module-specific configuration")
    llm_config: Optional[LLMConfig] = Field(default=None, description="LLM configuration for this module")

    def __init__(self, ensure_module_existence: bool = True, auto_fill_module_version: bool = True, **data):
        """Initialize with automatic module ID generation if not provided."""
        # Get module class first to determine version
        module_type = data.get('module_type')
        module_name = data.get('module_name')
        module_version = data.get('module_version')
        
        if module_type and module_name:
            module_class = get_module_class(module_type, module_name, module_version)
            if ensure_module_existence and module_class is None:
                raise ValueError(f"Module class not found for module type: {module_type}, module name: {module_name}, module version: {module_version}, possibly because the module is not imported.")
            
            # Auto-fill module version if requested
            if auto_fill_module_version and module_class:
                data['module_version'] = module_class.module_version
                module_version = module_class.module_version
            
            # Set module_id if not provided
            if 'module_id' not in data or data['module_id'] is None:
                data['module_id'] = f"{module_name}-{module_version}"
        
        super().__init__(**data)
    
    def __str__(self) -> str:
        """Return a nicely formatted string representation."""
        indent = "    "
        lines = ["ModuleConfig("]
        
        # Add basic fields
        basic_fields = [
            f"module_id='{self.module_id}'",
            f"module_type='{self.module_type}'",
            f"module_name='{self.module_name}'"
        ]
        
        if self.module_version:
            basic_fields.append(f"module_version='{self.module_version}'")
        
        # basic_fields.append(f"enabled={self.enabled}")
        
        # Add basic fields with indentation
        for field in basic_fields:
            lines.append(f"{indent}{field},")
        
        # Add config if present
        if self.config:
            lines.append(f"{indent}config={self.config},")
        
        # Add llm_config if present
        if self.llm_config is not None:
            llm_str = str(self.llm_config).replace('\n', f'\n{indent}')
            lines.append(f"{indent}llm_config={llm_str},")
        
        lines.append(")")
        return "\n".join(lines)


class FrameworkConfig(BaseSettings):
    """Main configuration class for the SciLeo Agent framework."""
    
    model_config = ConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    framework_name: str = Field(default="SciLeo Agent", description="Name of the framework")
    framework_version: str = Field(default=framework_version, description="Version of the framework")
    
    modules: Dict[str, ModuleConfig] = Field(default_factory=dict, description="Module configurations")
    loop_config: Dict[str, Any] = Field(default_factory=dict, description="Optimization loop parameters")
    
    # output_directory: str = Field(default=None, description="Directory for output files")
    # save_final_results: bool = Field(default=True, description="Whether to save final results")
    
    # def __init__(self, **data):
    #     """Initialize with automatic run ID generation if not provided."""
    #     if 'run_id' not in data or data['run_id'] is None:
    #         run_name = data.get('run_name')
    #         if not run_name:
    #             run_name = "run"
    #         run_id = f"{run_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    #         data['run_id'] = run_id
    #     if 'output_directory' not in data or data['output_directory'] is None:
    #         data['output_directory'] = f"runs/{data['run_id']}/outputs"
    #     super().__init__(**data)
    
    def __str__(self) -> str:
        """Return a nicely formatted string representation with indentation."""
        indent = "    "
        lines = ["FrameworkConfig("]
        
        # Add basic fields
        basic_fields = [
            f"framework_name='{self.framework_name}'",
            f"framework_version='{self.framework_version}'",
        ]
        
        # if self.run_name:
        #     basic_fields.append(f"run_name='{self.run_name}'")
        # if self.high_level_goal:
        #     basic_fields.append(f"high_level_goal='{self.high_level_goal}'")
        # if self.context_information:
        #     basic_fields.append(f"context_information='{self.context_information}'")
        
        # Add basic fields with indentation
        for field in basic_fields:
            lines.append(f"{indent}{field},")
        
        # Add modules if present
        if self.modules:
            lines.append(f"{indent}modules={{")
            for module_id, module_config in self.modules.items():
                module_str = str(module_config).replace('\n', f'\n{indent}{indent}')
                lines.append(f"{indent}{indent}'{module_id}': {module_str},")
            lines.append(f"{indent}}},")
        
        # Add loop_config if present
        if self.loop_config:
            lines.append(f"{indent}loop_config={self.loop_config},")
        
        # Add remaining fields
        # remaining_fields = [
        #     f"output_directory='{self.output_directory}'",
        #     f"save_final_results={self.save_final_results}"
        # ]
        
        # for field in remaining_fields:
        #     lines.append(f"{indent}{field},")
        
        lines.append(")")
        return "\n".join(lines)
    
    def add_module(self, module_config: ModuleConfig) -> None:
        """Add a module configuration."""
        self.modules[module_config.module_id] = module_config
    
    def get_module_config(self, module_id: str) -> Optional[ModuleConfig]:
        """Get configuration for a specific module."""
        return self.modules.get(module_id)
    
    def get_module_configs_by_type(self, module_type: str) -> List[ModuleConfig]:
        """Get all module configurations for a specific type."""
        return [config for config in self.modules.values() if config.module_type == module_type]
    
    # def setup_directories(self) -> None:
    #     """Create necessary directories."""
    #     Path(self.output_directory).mkdir(parents=True, exist_ok=True)
    
    def save_to_file(self, filepath: Union[str, Path]) -> None:
        """Save configuration to a file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.model_dump(), f, indent=2, default=str)
    
    @classmethod
    def from_file(cls, filepath: Union[str, Path]) -> 'FrameworkConfig':
        """Load configuration from a file."""
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {filepath}")
        
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls(**data)


# Default module configurations
DEFAULT_MODULE_CONFIGS = {
    "planner": {
        "module_id": "planner",
        "module_type": "planner",
        "module_name": "general_planner",
        "config": {},
        "llm_config": {}
    },
    "scorer_creator": {
        "module_id": "scorer_creator",
        "module_type": "scorer_creator",
        "module_name": "general_scorer_creator",
        "config": {},
        "llm_config": {}
    },
    "optimizer": {
        "module_id": "optimizer",
        "module_type": "optimizer",
        "module_name": "optimizer",
        "config": {},
        "llm_config": {}
    },
    "analyzer": {
        "module_id": "analyzer",
        "module_type": "analyzer",
        "module_name": "basic_analyzer",
        "config": {},
        "llm_config": {}
    },
    "knowledge_manager": {
        "module_id": "knowledge_manager",
        "module_type": "knowledge_manager",
        "module_name": "basic_knowledge_manager",
        "module_version": "0.1.0",
        "config": {},
        "llm_config": None
    }
}


def create_config(
    run_id: Optional[str] = None,
    run_name: Optional[str] = None,
    model_name: Optional[str] = None,
    loop_config: Optional[Dict[str, Any]] = None,
    module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    models_file: str = "llm_configs/models.yaml",
    credentials_file: str = "llm_configs/credentials.yaml",
    ensure_module_existence: bool = True,
    auto_fill_module_version: bool = True,
    default_module_configs: Optional[Dict[str, Dict[str, Any]]] = None
) -> FrameworkConfig:
    """
    Create a default framework configuration.
    
    Args:
        run_name: Optional run name
        loop_config: Dict mapping loop config keys to values
            Example: {
                "max_iterations": 10,
            }
        model_name: Default model name from YAML config (fallback for modules without specific config)
        module_configs: Dict mapping module_type to module configuration overrides
            Example: {
                "planner": {
                    "module_name": "Planner",
                    "module_version": "1.0.0",
                    "config": {"strategy": "adaptive"},
                    "llm_config": {"temperature": 0.7}
                }
            }
        high_level_goal: The optimization goal
        context_information: Optional context information for the optimization
        models_file: Path to models YAML file
        credentials_file: Path to credentials YAML file
    """
    config = FrameworkConfig(
        run_id=run_id,
        run_name=run_name,
        loop_config=loop_config or {}
    )
    
    # Helper function to create LLM config for a module
    def create_module_llm_config(module_llm_config: Optional[Dict[str, Any]] = None) -> Optional[LLMConfig]:
        if module_llm_config is None:
            return None
        
        # Start with base configuration
        llm_config_data = {
            "models_file": models_file,
            "credentials_file": credentials_file,
            "model_name": model_name or "openai/gpt-4.1-nano-2025-04-14"
        }
        
        # Apply module-specific LLM overrides if provided
        if module_llm_config:
            llm_config_data.update(module_llm_config)
        
        return LLMConfig(**llm_config_data)
    
    # Process module configurations
    # Always start with default configurations
    final_module_configs = default_module_configs or DEFAULT_MODULE_CONFIGS
    final_module_configs = final_module_configs.copy()
    
    # Update with provided configurations if any
    if module_configs:
        for module_type, module_config_data in module_configs.items():
            if module_type in final_module_configs:
                # Update existing default configuration
                final_module_configs[module_type].update(module_config_data)
            else:
                warnings.warn(f"Module type {module_type} not allowed. Skipped.")
    
    # Create modules from final configurations
    for module_type, base_config in final_module_configs.items():
        # Extract LLM config if provided
        module_llm_config = base_config.pop("llm_config", {})
        
        # Create LLM config
        llm_config = create_module_llm_config(module_llm_config)

        # Create and add ModuleConfig
        module_config = ModuleConfig(
            llm_config=llm_config,
            ensure_module_existence=ensure_module_existence,
            auto_fill_module_version=auto_fill_module_version,
            **base_config
        )
        config.add_module(module_config)
    
    return config 