"""
Parallel execution utilities for genetic operations.

Provides consistent parallel execution with error handling for LLM-based operations.
"""

import concurrent.futures
import logging
from typing import Any, Callable, List, Optional, Tuple


class ParallelGeneticExecutor:
    """Manages parallel execution of genetic operations with consistent error handling."""

    def __init__(self, max_workers: int = 200):
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum number of concurrent workers
        """
        self.max_workers = max_workers

    def execute_operations(
        self,
        operation: Callable,
        inputs: List[Any],
        error_handler: Optional[Callable[[Exception], None]] = None,
        return_artifacts: bool = False,
    ) -> List[Any]:
        """
        Execute operations in parallel and collect results.

        Args:
            operation: Function to execute on each input
            inputs: List of inputs to process
            error_handler: Optional function to handle errors (receives exception)
            return_artifacts: Whether operation returns artifacts (for logging)

        Returns:
            List of results from successful operations
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [executor.submit(operation, inp) for inp in inputs]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if error_handler:
                        error_handler(e)
                    else:
                        logging.error(f"Operation failed: {e}")
                        # Re-raise LiteLLM errors to propagate API issues
                        if "litellm" in str(e).lower():
                            raise

        return results

    def execute_with_args(
        self,
        operation: Callable,
        arg_tuples: List[Tuple],
        error_handler: Optional[Callable[[Exception], None]] = None,
    ) -> List[Any]:
        """
        Execute operations with multiple arguments in parallel.

        Args:
            operation: Function to execute
            arg_tuples: List of argument tuples to unpack
            error_handler: Optional error handler

        Returns:
            List of results from successful operations
        """
        results = []

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [executor.submit(operation, *args) for args in arg_tuples]

            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    if error_handler:
                        error_handler(e)
                    else:
                        logging.error(f"Operation failed: {e}")
                        if "litellm" in str(e).lower():
                            raise

        return results

    def execute_crossover_batch(
        self,
        crossover_fn: Callable,
        parent_pairs: List[Tuple[Any, Any]],
        objectives: List[Any],
        return_artifacts: bool = False,
    ) -> Tuple[List[Any], List[Tuple]]:
        """
        Execute crossover operations in parallel.

        Args:
            crossover_fn: Crossover function
            parent_pairs: List of (parent_a, parent_b) tuples
            objectives: Optimization objectives
            return_artifacts: Whether to collect prompts/responses

        Returns:
            Tuple of (offspring_list, artifacts_list)
        """
        offspring = []
        artifacts = []

        def crossover_wrapper(parents):
            return crossover_fn(parents, objectives, return_artifacts)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(crossover_wrapper, parents) for parents in parent_pairs
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    if return_artifacts:
                        child, prompt, response = future.result()
                        offspring.append(child)
                        artifacts.append((prompt, response, child.representation))
                    else:
                        child = future.result()
                        offspring.append(child)
                except Exception as e:
                    logging.error(f"Crossover failed: {e}")
                    if "litellm" in str(e).lower():
                        raise

        return offspring, artifacts

    def execute_mutation_batch(
        self,
        mutation_fn: Callable,
        candidates: List[Any],
        objectives: List[Any],
        return_artifacts: bool = False,
    ) -> Tuple[List[Any], List[Tuple]]:
        """
        Execute mutation operations in parallel.

        Args:
            mutation_fn: Mutation function
            candidates: List of candidates to mutate
            objectives: Optimization objectives
            return_artifacts: Whether to collect prompts/responses

        Returns:
            Tuple of (mutant_list, artifacts_list)
        """
        mutants = []
        artifacts = []

        def mutation_wrapper(candidate):
            return mutation_fn(candidate, objectives, return_artifacts)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.max_workers
        ) as executor:
            futures = [
                executor.submit(mutation_wrapper, candidate)
                for candidate in candidates
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    if return_artifacts:
                        mutant, prompt, response = future.result()
                        mutants.append(mutant)
                        artifacts.append((prompt, response, mutant.representation))
                    else:
                        mutant = future.result()
                        mutants.append(mutant)
                except Exception as e:
                    logging.error(f"Mutation failed: {e}")
                    if "litellm" in str(e).lower():
                        raise

        return mutants, artifacts
