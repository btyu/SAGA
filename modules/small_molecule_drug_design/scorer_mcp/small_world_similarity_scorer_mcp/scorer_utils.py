from typing import List, Dict, Any, Callable
from loguru import logger
from functools import wraps


def scorer(name: str, population_wise: bool, description: str):
    """
    Decorator to register a method as a scorer function.

    Args:
        name: Name of the scorer
        population_wise: Whether the scorer operates on population level
        description: Description of the scorer
    """

    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)

        wrapper._scorer_name = name
        wrapper._scorer_population_wise = population_wise
        wrapper._scorer_description = description
        wrapper._is_scorer = True

        return wrapper

    return decorator


class BaseScorer:
    """
    Base class that provides scorer registry functionality.
    Classes inheriting from this can use the @scorer decorator to register scoring methods.
    """

    def __init__(self):
        self.scorers: Dict[str, Dict[str, Any]] = {}
        self._register_scorers()

    def _register_scorers(self):
        """Automatically register all methods decorated with @scorer."""
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if hasattr(attr, "_is_scorer") and attr._is_scorer:
                self.scorers[attr._scorer_name] = {
                    "method": attr,
                    "function_name": attr_name,
                    "population_wise": attr._scorer_population_wise,
                    "description": attr._scorer_description,
                    "tool_description": attr.__doc__,
                }
                logger.info(f"Registered scorer: {attr._scorer_name}")

    def get_available_scorers(self) -> Dict[str, Dict[str, Any]]:
        """Return metadata for all registered scorers."""
        return {
            name: {k: v for k, v in info.items() if k != "method"}
            for name, info in self.scorers.items()
        }

    def get_scorer_names(self) -> List[str]:
        """Return list of scorer names."""
        return list(self.scorers.keys())

