"""
Utility modules for the SciLeo Agent framework.

This package contains helper utilities for framework operations:

- LLM integration: Multi-provider LLM client with configuration management
- Logging: Structured logging for optimization runs and agent activities
- Scorer: Objective scoring system with decorators for custom scorers
- Common utilities: Shared functionality across the framework

These utilities support the core framework functionality and provide
consistent interfaces for external integrations.
"""

from .llm import (
    BaseLLMConfig,
    LLMClient,
    LLMFactory,
    list_available_models,
    create_client,
    get_models_by_provider,
    list_providers
)

from .logging import (
    SciLeoLogger,
    setup_logging,
    get_logger,
)

from .human_feedback import (
    get_multiline_input,
    validate_json,
    confirm_input,
    get_validated_json_input,
    display_objectives_for_feedback,
    validate_objectives_dict,
    get_human_feedback_on_objectives
)

__all__ = [
    "BaseLLMConfig",
    "LLMClient",
    "LLMFactory",
    "list_available_models",
    "create_client",
    "get_models_by_provider",
    "list_providers",
    "SciLeoLogger",
    "setup_logging",
    "get_logger",
    "reset_logger_configuration",
    "enable_monitoring",
    "disable_monitoring",
    "get_multiline_input",
    "validate_json",
    "confirm_input",
    "get_validated_json_input",
    "display_objectives_for_feedback",
    "validate_objectives_dict",
    "get_human_feedback_on_objectives"
] 