"""
Logging utilities for the SciLeo Agent framework.

This module provides structured logging capabilities with different levels,
formatters, and output destinations.
"""

import sys
from typing import Dict, Any, Optional, Union, List
from loguru import logger as base_logger
from datetime import datetime
import json
from contextlib import contextmanager
import copy


class SciLeoLogger:
    """
    Enhanced logger for the SciLeo Agent framework with structured logging.
    """

    def __init__(self, level: str = "INFO", log_dir: Optional[str] = None):
        self.logger = base_logger.bind(app="SciLeoAgent")
        self.level = level
        self.log_dir = log_dir
        self.context = {}
        self._setup_logger()

    def _setup_logger(self):
        """Set up the logger with appropriate configuration."""

        self.logger.remove()

        # Console handler
        self.logger.add(
            sys.stdout,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
                    "<level>{level: <8}</level> | "
                    "<level>{message}</level>",
            filter=lambda record: record["extra"].get("app") == "SciLeoAgent",
            level=self.level,
            colorize=True,
            backtrace=True,
            diagnose=True
        )

        # File handlers (if log_dir is specified)
        if self.log_dir:
            from pathlib import Path
            log_path = Path(self.log_dir)
            log_path.mkdir(parents=True, exist_ok=True)

            # All logs file (DEBUG and above)
            self.logger.add(
                log_path / "all_logs.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
                filter=lambda record: record["extra"].get("app") == "SciLeoAgent",
                level="DEBUG",
                rotation="100 MB",
                retention="10 days",
                backtrace=True,
                diagnose=True
            )

            # Important logs file (INFO and above)
            self.logger.add(
                log_path / "important_logs.log",
                format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {message}",
                filter=lambda record: record["extra"].get("app") == "SciLeoAgent",
                level="INFO",
                rotation="100 MB",
                retention="10 days",
                backtrace=True,
                diagnose=True
            )
    
    def set_context(self, context: Dict[str, Any]) -> None:
        """Set logging context that will be included in all messages."""
        self.context.update(context)
    
    def clear_context(self) -> None:
        """Clear the logging context."""
        self.context.clear()
    
    def _format_message(self, message: str, extra: Optional[Dict[str, Any]] = None) -> str:
        """Format message with context and extra data."""
        # Combine context and extra data
        all_extra = {**self.context}
        if extra:
            all_extra.update(extra)
        
        if all_extra:
            # Format extra data as JSON string
            extra_str = json.dumps(all_extra, default=str)
            return f"{message} | {extra_str}"
        return message
    
    def debug(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log debug message."""
        self.logger.debug(self._format_message(message, extra))
    
    def info(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log info message."""
        self.logger.info(self._format_message(message, extra))
    
    def warning(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log warning message."""
        self.logger.warning(self._format_message(message, extra))
    
    def error(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log error message."""
        self.logger.error(self._format_message(message, extra))
    
    def critical(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log critical message."""
        self.logger.critical(self._format_message(message, extra))

    def exception(self, message: str, extra: Optional[Dict[str, Any]] = None) -> None:
        """Log exception with traceback."""
        self.logger.exception(self._format_message(message, extra))
    
    def log_module_activity(
        self,
        module_id: str,
        activity: str,
        details: Optional[Dict[str, Any]] = None,
        level: str = "INFO"
    ) -> None:
        """Log module activity with structured information."""
        extra = {
            "module_id": module_id,
            "activity": activity
        }
        
        if details:
            extra.update(details)
        
        message = f"Module {module_id}: {activity}"
        
        if level.upper() == "DEBUG":
            self.debug(message, extra)
        elif level.upper() == "INFO":
            self.info(message, extra)
        elif level.upper() == "WARNING":
            self.warning(message, extra)
        elif level.upper() == "ERROR":
            self.error(message, extra)
        elif level.upper() == "CRITICAL":
            self.critical(message, extra)
    
    def log_optimization_step(
        self,
        step: str,
        generation: int,
        population_size: int,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Log optimization step with structured information."""
        extra = {
            "step": step,
            "generation": generation,
            "population_size": population_size
        }
        
        if details:
            extra.update(details)
        
        message = f"Optimization step: {step} (Gen {generation}, Pop {population_size})"
        self.info(message, extra)
    
    def log_performance_metrics(
        self,
        metrics: Dict[str, Union[int, float]],
        context: Optional[str] = None
    ) -> None:
        """Log performance metrics."""
        message = f"Performance metrics{' for ' + context if context else ''}"
        self.info(message, metrics)
    
    def log_llm_call(
        self,
        model: str,
        tokens_used: int,
        duration: float,
        cost: Optional[float] = None
    ) -> None:
        """Log LLM call information."""
        extra = {
            "model": model,
            "tokens_used": tokens_used,
            "duration": duration
        }
        
        if cost is not None:
            extra["cost"] = cost
        
        message = f"LLM call to {model} ({tokens_used} tokens, {duration:.2f}s)"
        self.debug(message, extra)
    
    def log_exception(self, exception: Exception, context: Optional[str] = None) -> None:
        """Log exception with full traceback."""
        message = f"Exception occurred{' in ' + context if context else ''}: {str(exception)}"
        self.logger.exception(message)


@contextmanager
def avoid_logging_change():
    """
    Context manager to protect logging configuration from external library changes.
    
    This is particularly useful when importing libraries like grelu that may modify
    the global logging configuration, including loguru handlers.
    
    Usage:
        with avoid_logging_change():
            import grelu.lightning
            import grelu.data.dataset
    """
    # Save current SciLeoLogger configuration if it exists
    global default_logger
    original_scileo_config = None
    
    if default_logger is not None:
        # Save the current logger configuration
        original_scileo_config = {
            'level': default_logger.level,
            'context': copy.deepcopy(default_logger.context)
        }
    
    try:
        yield
    finally:
        # If we had a SciLeoLogger, re-setup it with original configuration
        if original_scileo_config is not None and default_logger is not None:
            default_logger.level = original_scileo_config['level']
            default_logger.context = original_scileo_config['context']
            # Force re-setup of the logger to restore original format
            default_logger._setup_logger()
        elif default_logger is not None:
            # If no original config saved but logger exists, just re-setup
            default_logger._setup_logger()


# Global logger instance
default_logger = None


def setup_logging(level: str = "INFO", log_dir: Optional[str] = None) -> SciLeoLogger:
    """
    Set up the global logging configuration.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Optional directory to save log files. If provided, creates:
                - all_logs.log: All logs (DEBUG and above)
                - important_logs.log: Important logs (INFO and above)

    Returns:
        Configured logger instance
    """
    global default_logger

    if default_logger is None:
        # Create the logger for the first time
        default_logger = SciLeoLogger(level=level, log_dir=log_dir)
    else:
        # Update existing logger configuration
        default_logger.level = level
        default_logger.log_dir = log_dir
        default_logger._setup_logger()

    return default_logger


def get_logger() -> SciLeoLogger:
    """Get the global logger instance."""
    global default_logger

    if default_logger is None:
        # Initialize with default settings if not already done
        default_logger = SciLeoLogger()

    return default_logger


 