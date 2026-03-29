"""
General scorer creator module for the SciLeo Agent framework.

This module provides a general-purpose scorer creator that retrieves scorers
from the ScorerManager and can create new ones using LLM when needed.
"""

from typing import List, Dict, Any, Callable, Optional, Tuple
from scileo_agent.core.data_models.candidate import Candidate
import json
import re
import time
import os
import asyncio

from scileo_agent.core.modules import ScorerCreatorModule
from scileo_agent.core.data_models.objective import Objective
from scileo_agent.core.registry.module_registry import register_module
from scileo_agent.core.registry.scorer_registry import ScorerManager, get_scorer, list_scorers
from scileo_agent.core.registry.serializer_registry import get_serializer
from scileo_agent.utils import get_logger, LLMClient
from scileo_agent.core.config import DEV_DEFAULT

from .scorer_implementor import ScorerImplementor


logger = get_logger()


# Pairwise comparison prompt template
PAIRWISE_COMPARISON_SYSTEM_PROMPT = "You are an expert at evaluating whether a scoring function exactly matches an optimization objective. You must be very strict and only match if the scorer does exactly what the objective measures."

PAIRWISE_COMPARISON_USER_PROMPT_TEMPLATE = """Analyze whether this scorer exactly matches the given objective:

OBJECTIVE:
- Name: {objective_name}
- Description: {objective_description}

SCORER TO EVALUATE:
- Name: {scorer_name}
- Description: {scorer_description}

CRITICAL MATCHING REQUIREMENTS:
1. The scorer must do EXACTLY what the objective measures - not just something related
2. Pay special attention to VALUE RANGES - they must be compatible
3. Pay special attention to OPTIMIZATION DIRECTION - check the descriptions carefully: Both objective and scorer descriptions should indicate the same direction (higher/better vs lower/better)
4. The domains and contexts must align perfectly
5. The measurement units and scales should be compatible

Respond with a JSON in <answer>...</answer> tags:
- If the scorer EXACTLY matches: {{"match": true, "reasoning": "explanation"}}
- If related but not exact: {{"match": false, "reasoning": "explanation", "related": true}}
- If not related: {{"match": false, "reasoning": "explanation", "related": false}}

Example:
<answer>
{{"match": false, "reasoning": "Your detailed analysis here", "related": false}}
</answer>"""

# Multi-scorer selection prompt template
MULTI_SCORER_SELECTION_SYSTEM_PROMPT = "You are an expert at ranking multiple scoring functions that all potentially match an objective. Select the best one with confidence scores."

MULTI_SCORER_SELECTION_USER_PROMPT_TEMPLATE = """Multiple scorers have been identified as potential matches for the objective. Evaluate each scorer and assign confidence scores:

OBJECTIVE:
- Name: {objective_name}
- Description: {objective_description}

CANDIDATE SCORERS:
{matched_scorers}

CRITICAL MATCHING REQUIREMENTS:
1. The scorer must do EXACTLY what the objective measures - not just something related
2. Pay special attention to VALUE RANGES - they must be compatible
3. Pay special attention to OPTIMIZATION DIRECTION - check the descriptions carefully: Both objective and scorer descriptions should indicate the same direction (higher/better vs lower/better)
4. The domains and contexts must align perfectly
5. The measurement units and scales should be compatible

SCORING GUIDELINES:
- 100: EXACT MATCH ONLY - scorer does precisely what the objective measures with identical requirements (value ranges, optimization direction, domain, units)
- 80-99: Related scorer - close match but with differences in requirements, ranges, or direction
- 50-79: Related scorer - same domain but different measurement focus
- 20-49: Related scorer - some conceptual overlap but significant differences
- 0-19: Not related or poor match

IMPORTANT:
- Only assign confidence 100 if the scorer EXACTLY matches all requirements above
- If you find a scorer does actually not match the objective, give it a confidence score of 0
- Each scorer MUST receive a unique confidence score - no duplicates allowed (except multiple 0s)

Example:
<answer>
{{
    "scorer_confidences": [
        {{"scorer_name": "scorer1", "confidence": 100, "reasoning": "detailed explanation for this confidence level"}},
        {{"scorer_name": "scorer2", "confidence": 90, "reasoning": "detailed explanation for this confidence level"}}
    ],
    "selected_scorer": "scorer1",
    "selection_reasoning": "Overall reasoning for why this scorer is the best choice"
}}
</answer>"""


@register_module("general_scorer_creator", "0.8.0")
class GeneralScorerCreator(ScorerCreatorModule):
    """
    General scorer creator module that retrieves scorers from ScorerManager.
    
    This module uses LLM to intelligently match objectives with available scorers
    based on their names and descriptions, providing better matching than simple
    name-based lookup.

    v0.8.0: Subagent. Support for test example input.
    v0.7.0: Added deployment verification in ScorerImplementor, and added runnable scorer emphasis in the coding agent prompt
    v0.4.0: Added async support and parallel processing for better efficiency
    v0.3.1: Added usage tracking
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None, llm_config=None):
        """
        Initialize the general scorer creator module.
        
        Args:
            module_id: Unique identifier for this module
            config: Configuration parameters
            llm_config: LLM configuration
        """
        super().__init__(module_id, config, llm_config)
        
        # LLM-based scorer creation configuration
        self.enable_llm_scorer_creation = config.get("enable_llm_scorer_creation", True)
        self.coding_workspace_path = config.get("coding_workspace_path", "coding_workspace")
        self.generated_scorer_library_path = config.get("generated_scorer_library_path", "generated_scorers")
        self.scorer_library_subfolder = config.get("scorer_library_subfolder", None)
        self.dev = config.get("dev", DEV_DEFAULT)
        self.coding_agent_model = config.get("coding_agent_model_name", "claude_code/claude-sonnet-4-5-20250929")
        self.coding_agent_models_file = config.get("coding_agent_models_file", os.path.join('llm_configs', 'claude_code.yaml'))
        self.coding_agent_credentials_file = config.get("coding_agent_credentials_file", os.path.join('llm_configs', 'credentials.yaml'))
        self.coding_agent_run_in_docker = config.get("coding_agent_run_in_docker", False if self.coding_agent_model.startswith("claude_code") else True)
        self.max_llm_retries = config.get("max_llm_retries", 3)

        # Parallel scorer creation configuration
        self.coding_agent_max_parallel_scorer_creation = config.get("coding_agent_max_parallel_scorer_creation", 3)

        # Initialize semaphore for controlling parallel scorer creations
        self._scorer_creation_semaphore = None  # Will be initialized when needed

        # Name-based matching configuration
        self.enable_name_matching = config.get("enable_name_matching", True)
        
        # LLM-based matching configuration
        self.enable_llm_matching = config.get("enable_llm_matching", True)
        
        if self.enable_llm_scorer_creation or self.enable_llm_matching:
            if not self.has_llm():
                logger.error("LLM is not available but LLM-based matching is enabled")
                raise ValueError("LLM is not available but LLM scorer creation or LLM-based matching is enabled.")

        # Reference module paths
        self.reference_module_paths = config.get("reference_module_paths", None) or []
        self.use_potential_matched_scorers_as_references = config.get("use_potential_matched_scorers_as_references", True)
        
        # Get ScorerManager instance
        self.scorer_manager = ScorerManager()

        # Initialize the scorer implementor
        self.scorer_implementor = ScorerImplementor(
            config={
                "coding_workspace_path": self.coding_workspace_path,
                "generated_scorer_library_path": self.generated_scorer_library_path,
                "scorer_library_subfolder": self.scorer_library_subfolder,
                "dev": self.dev,
                "coding_agent_model": self.coding_agent_model,
                "coding_agent_models_file": self.coding_agent_models_file,
                "coding_agent_credentials_file": self.coding_agent_credentials_file,
                "coding_agent_run_in_docker": self.coding_agent_run_in_docker,
            }
        ) if self.enable_llm_scorer_creation else None
    
    async def get_scorers(self, objectives: List[Objective], serializer_name: str = None, test_candidates: Optional[List[Candidate]] = None) -> dict[str, Any]:
        """
        Async implementation of get_scorers that processes objectives in parallel.

        Args:
            objectives: List of objectives that need scorers
            serializer_name: Name of the serializer to use for getting sample schema and description
            test_candidates: Optional list of example candidates for testing. These will be serialized
                           and saved as test_examples.json for reference during scorer implementation.

        Returns:
            Dictionary of results, including objectives with their scorers and the information about the matching
        """
        logger.debug(f"Processing {len(objectives)} objectives for scorer retrieval (async)")

        # Process all objectives in parallel
        results = await asyncio.gather(
            *[self._process_single_objective_async(objective, serializer_name, test_candidates) for objective in objectives],
            return_exceptions=True
        )

        # Separate successful results from failures
        objectives_with_scorers = []
        objectives_without_scorers = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Error processing objective '{objectives[i].name}': {result}")
                objectives_without_scorers.append(objectives[i])
            elif result is not None:
                objectives_with_scorers.append(result)
            else:
                objectives_without_scorers.append(objectives[i])

        # Build available objectives list
        scorer_names_in_registry = list_scorers()
        scorer_info_in_registry = {
            name: self.scorer_manager.get_scorer_metadata(name)
            for name in scorer_names_in_registry
        }
        already_matched_scorer_ids = set([id(objective.scorer) for objective in objectives_with_scorers])
        available_objectives = []
        for scorer_name, scorer_metadata in scorer_info_in_registry.items():
            if id(self.scorer_manager.get_scorer(scorer_name)) in already_matched_scorer_ids:
                continue
            else:
                # Get type directly from metadata (with fallback for backward compatibility)
                scorer_type = scorer_metadata.get("type")
                if scorer_type is None:
                    # Fallback: infer from population_wise if type not available
                    population_wise = scorer_metadata.get("population_wise", False)
                    scorer_type = "population-wise" if population_wise else "candidate-wise"

                available_objective = Objective(
                    name=scorer_name,
                    description=scorer_metadata["description"],
                    type=scorer_type,
                    scorer=self.scorer_manager.get_scorer(scorer_name)
                )
                available_objectives.append(available_objective)

        return_dict = {
            "matched_objectives": objectives_with_scorers,
            "unmatched_objectives": objectives_without_scorers,
            "available_objectives": available_objectives,
        }

        return return_dict

    async def _process_single_objective_async(self, objective: Objective, serializer_name: str = None, test_candidates: Optional[List[Candidate]] = None) -> Optional[Objective]:
        """
        Process a single objective asynchronously.

        Args:
            objective: Objective to process
            serializer_name: Name of the serializer to use
            test_candidates: Optional list of example candidates for testing

        Returns:
            Objective with scorer if successful, None otherwise
        """
        logger.debug(f"Processing objective: '{objective.name}'")

        # Skip if already has scorer
        if objective.has_scorer():
            logger.debug(f"Objective '{objective.name}' already has a scorer, skipping")
            return objective

        # Try to find existing scorer using LLM-based matching
        scorer, potential_matches = await self._find_scorer_for_objective_async(objective)

        if scorer:
            # Create new objective with scorer
            new_objective = Objective(
                name=objective.name,
                description=objective.description,
                optimization_direction=objective.optimization_direction,
                type=objective.type,
                weight=objective.weight,
                metadata=objective.metadata.copy()
            )
            new_objective.set_scorer(scorer)
            return new_objective
        else:
            logger.debug(f"No existing scorer found for objective '{objective.name}'")
            # Try to create new scorer using LLM (async with semaphore control)
            if self.enable_llm_scorer_creation:
                logger.debug(f"Attempting to create new scorer using LLM for objective '{objective.name}'")

                # Initialize semaphore on first use (in async context)
                if self._scorer_creation_semaphore is None:
                    self._scorer_creation_semaphore = asyncio.Semaphore(self.coding_agent_max_parallel_scorer_creation)
                    logger.debug(f"Initialized scorer creation semaphore with limit: {self.coding_agent_max_parallel_scorer_creation}")

                # Use semaphore to limit parallel scorer creations
                async with self._scorer_creation_semaphore:
                    scorer = await self._create_scorer_with_llm_async(objective, potential_matches, serializer_name, test_candidates)

                if scorer:
                    logger.debug(f"Successfully created new scorer using LLM for objective '{objective.name}'")
                    new_objective = Objective(
                        name=objective.name,
                        description=objective.description,
                        optimization_direction=objective.optimization_direction,
                        type=objective.type,
                        weight=objective.weight,
                        metadata=objective.metadata.copy()
                    )
                    new_objective.set_scorer(scorer)
                    return new_objective
                else:
                    logger.debug(f"Failed to create scorer using LLM for objective '{objective.name}'")
                    return None
            else:
                logger.debug(f"No scorer available for objective '{objective.name}' and LLM creation is disabled")
                return None
    
    async def _find_scorer_for_objective_async(self, objective: Objective) -> Tuple[Optional[Callable], Optional[List[str]]]:
        """
        Find an existing scorer for the given objective using LLM-based matching (async version).

        This method uses an LLM to intelligently match objectives with available
        scorers based on their names and descriptions, rather than relying on
        exact name matches.

        Args:
            objective: The objective to find a scorer for

        Returns:
            Tuple of (scorer function, potential matches)
        """
        if self.enable_name_matching:
            # First try exact name match for efficiency
            scorer = get_scorer(objective.name)
            if scorer:
                meta = self.scorer_manager.get_scorer_metadata(objective.name)
                # Get scorer type directly from metadata (with fallback to population_wise)
                scorer_type = meta.get("type") if meta else None
                if scorer_type is None and meta:
                    scorer_is_population_wise = meta.get("population_wise", False)
                    scorer_type = "population-wise" if scorer_is_population_wise else "candidate-wise"
                if scorer_type == objective.type:
                    logger.debug(f"Found exact name match for objective '{objective.name}'")
                    return scorer, None

            # Check metadata for scorer hints
            if "scorer_name" in objective.metadata:
                scorer_name = objective.metadata["scorer_name"]
                logger.debug(f"Checking metadata scorer hint: '{scorer_name}'")
                scorer = get_scorer(scorer_name)
                if scorer:
                    meta = self.scorer_manager.get_scorer_metadata(scorer_name)
                    # Get scorer type directly from metadata (with fallback to population_wise)
                    scorer_type = meta.get("type") if meta else None
                    if scorer_type is None and meta:
                        scorer_is_population_wise = meta.get("population_wise", False)
                        scorer_type = "population-wise" if scorer_is_population_wise else "candidate-wise"
                    if scorer_type == objective.type:
                        logger.debug(f"Found scorer via metadata hint for objective '{objective.name}': '{scorer_name}'")
                        return scorer, None

        if not self.enable_llm_matching:
            return None, None

        # Use LLM-based matching if enabled and LLM is available
        return await self._find_scorer_with_llm_async(objective)

    def _find_scorer_for_objective(self, objective: Objective) -> Tuple[Optional[Callable], Optional[List[str]]]:
        """
        Find an existing scorer for the given objective using LLM-based matching.

        This method uses an LLM to intelligently match objectives with available
        scorers based on their names and descriptions, rather than relying on
        exact name matches.

        Args:
            objective: The objective to find a scorer for

        Returns:
            Tuple of (scorer function, potential matches)
        """
        if self.enable_name_matching:
            # First try exact name match for efficiency
            scorer = get_scorer(objective.name)
            if scorer:
                meta = self.scorer_manager.get_scorer_metadata(objective.name)
                # Get scorer type directly from metadata (with fallback to population_wise)
                scorer_type = meta.get("type") if meta else None
                if scorer_type is None and meta:
                    scorer_is_population_wise = meta.get("population_wise", False)
                    scorer_type = "population-wise" if scorer_is_population_wise else "candidate-wise"
                if scorer_type == objective.type:
                    logger.debug(f"Found exact name match for objective '{objective.name}'")
                    return scorer, None

            # Check metadata for scorer hints
            if "scorer_name" in objective.metadata:
                scorer_name = objective.metadata["scorer_name"]
                logger.debug(f"Checking metadata scorer hint: '{scorer_name}'")
                scorer = get_scorer(scorer_name)
                if scorer:
                    meta = self.scorer_manager.get_scorer_metadata(scorer_name)
                    # Get scorer type directly from metadata (with fallback to population_wise)
                    scorer_type = meta.get("type") if meta else None
                    if scorer_type is None and meta:
                        scorer_is_population_wise = meta.get("population_wise", False)
                        scorer_type = "population-wise" if scorer_is_population_wise else "candidate-wise"
                    if scorer_type == objective.type:
                        logger.debug(f"Found scorer via metadata hint for objective '{objective.name}': '{scorer_name}'")
                        return scorer, None

        if not self.enable_llm_matching:
            return None, None

        # Use LLM-based matching if enabled and LLM is available
        return self._find_scorer_with_llm(objective)
    
    async def _find_scorer_with_llm_async(self, objective: Objective) -> Tuple[Optional[Callable], Optional[List[str]]]:
        """
        Use LLM to find the best matching scorer for an objective using pairwise comparison (async version).

        This version performs pairwise comparisons in parallel for better performance.

        Args:
            objective: The objective to find a scorer for

        Returns:
            Best matching scorer function if found, None otherwise
        """
        # Get all available scorers and their metadata
        available_scorers = list_scorers()
        if not available_scorers:
            logger.warning("No available scorers found for LLM matching")
            return None, None

        # Filter scorers by type setting and collect their metadata
        candidate_scorers = []
        for scorer_name in available_scorers:
            meta = self.scorer_manager.get_scorer_metadata(scorer_name)
            # Get scorer type directly from metadata (with fallback to population_wise)
            scorer_type = meta.get("type") if meta else None
            if scorer_type is None and meta:
                scorer_is_population_wise = meta.get("population_wise", False)
                scorer_type = "population-wise" if scorer_is_population_wise else "candidate-wise"
            if scorer_type == objective.type:
                candidate_scorers.append((scorer_name, meta))

        if not candidate_scorers:
            return None, None

        logger.debug(f"Evaluating {len(candidate_scorers)} candidate scorers for objective '{objective.name}' using pairwise comparison (parallel)")

        # Perform pairwise comparison for all scorers in parallel
        comparison_tasks = [
            self._compare_objective_scorer_pair_async(objective, scorer_name, scorer_meta)
            for scorer_name, scorer_meta in candidate_scorers
        ]

        comparison_results = await asyncio.gather(*comparison_tasks, return_exceptions=True)

        # Process results
        exact_matches = []
        related_matches = []

        for i, comparison_result in enumerate(comparison_results):
            scorer_name, scorer_meta = candidate_scorers[i]

            if isinstance(comparison_result, Exception):
                logger.warning(f"Failed to compare scorer '{scorer_name}' with objective '{objective.name}': {comparison_result}")
                continue

            if comparison_result["match"]:
                # Exact match found
                match_info = {
                    "scorer_name": scorer_name,
                    "description": scorer_meta.get("description", ""),
                    "reasoning": comparison_result["reasoning"]
                }
                exact_matches.append(match_info)
                logger.debug(f"Exact match found: {scorer_name} for objective '{objective.name}'")
            elif comparison_result.get("related", False):
                # Related but not exact match (as labeled by LLM)
                related_info = {
                    "scorer_name": scorer_name,
                    "description": scorer_meta.get("description", ""),
                    "reasoning": comparison_result["reasoning"]
                }
                related_matches.append(related_info)
                logger.debug(f"Related match found: {scorer_name} for objective '{objective.name}'")

        # Handle results
        if exact_matches:
            if len(exact_matches) == 1:
                # Single exact match - use it
                selected_scorer_name = exact_matches[0]["scorer_name"]
                logger.debug(f"Single exact match found: {selected_scorer_name} for objective '{objective.name}'")
            else:
                # Multiple exact matches - use LLM to select the one perfect match (confidence 100)
                logger.debug(f"Multiple exact matches found ({len(exact_matches)}) for objective '{objective.name}', checking for perfect match")
                selected_scorer_name = await self._select_best_from_multiple_matches_async(objective, exact_matches)

            if selected_scorer_name:
                scorer = get_scorer(selected_scorer_name)
                if scorer:
                    # Collect related scorers: LLM-labeled related + other non-selected scorers
                    llm_related_names = [m["scorer_name"] for m in related_matches]
                    other_exact_matches = [m["scorer_name"] for m in exact_matches if m["scorer_name"] != selected_scorer_name]
                    # Combine LLM-labeled related scorers with any other scorers that weren't selected
                    all_related_names = llm_related_names + other_exact_matches

                    logger.debug(f"Successfully matched scorer '{selected_scorer_name}' to objective '{objective.name}' via pairwise comparison")
                    return scorer, all_related_names
                else:
                    logger.warning(f"Selected scorer '{selected_scorer_name}' not found in registry")

        # No exact matches found - return LLM-labeled related scorers
        if related_matches:
            related_scorer_names = [m["scorer_name"] for m in related_matches]
            logger.debug(f"No exact matches for objective '{objective.name}', found {len(related_matches)} LLM-labeled related scorers: {related_scorer_names}")
            return None, related_scorer_names
        else:
            logger.debug(f"No matches (exact or related) found for objective '{objective.name}'")
            return None, None

    async def _compare_objective_scorer_pair_async(self, objective: Objective, scorer_name: str, scorer_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare a single objective-scorer pair to determine exact match (async version).

        Args:
            objective: The objective to match
            scorer_name: Name of the scorer to evaluate
            scorer_meta: Metadata of the scorer

        Returns:
            Dictionary with match result, reasoning, and related flag
        """
        # Create prompt for pairwise comparison
        prompt = PAIRWISE_COMPARISON_USER_PROMPT_TEMPLATE.format(
            objective_name=objective.name,
            objective_description=objective.description,
            scorer_name=scorer_name,
            scorer_description=scorer_meta.get("description", "")
        )

        try:
            for k in range(self.max_llm_retries + 1):
                try:
                    # Call LLM for pairwise comparison
                    # Note: LLM client call is synchronous, but we can still benefit from parallel execution
                    response = await asyncio.to_thread(
                        self.llm_client.call_with_prompt,
                        user_prompt=prompt,
                        system_prompt=PAIRWISE_COMPARISON_SYSTEM_PROMPT
                    )

                    # Parse LLM response
                    result = self._parse_pairwise_response(response['content'])
                    break
                except Exception as e:
                    if k >= self.max_llm_retries:
                        logger.error(f"Pairwise comparison failed for objective '{objective.name}' and scorer '{scorer_name}' after {k + 1} attempts: {e}")
                        raise e
                    else:
                        logger.debug(f"Pairwise comparison failed for objective '{objective.name}' and scorer '{scorer_name}' on attempt {k + 1}/{self.max_llm_retries + 1}: {e}")
                        await asyncio.sleep(self.llm_config.retry_delay * (2 ** k))

            if result:
                result["scorer_name"] = scorer_name
                return result
            else:
                logger.warning(f"Failed to parse pairwise comparison response for objective '{objective.name}' and scorer '{scorer_name}'")
                return {"match": False, "reasoning": "Failed to parse LLM response", "related": False, "scorer_name": scorer_name}

        except Exception as e:
            logger.error(f"Pairwise comparison failed for objective '{objective.name}' and scorer '{scorer_name}': {e}")
            if self.dev:
                raise
            return {"match": False, "reasoning": f"Error during comparison: {e}", "related": False, "scorer_name": scorer_name}

    def _compare_objective_scorer_pair(self, objective: Objective, scorer_name: str, scorer_meta: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare a single objective-scorer pair to determine exact match.

        Args:
            objective: The objective to match
            scorer_name: Name of the scorer to evaluate
            scorer_meta: Metadata of the scorer

        Returns:
            Dictionary with match result, reasoning, and related flag
        """
        # Create prompt for pairwise comparison
        prompt = PAIRWISE_COMPARISON_USER_PROMPT_TEMPLATE.format(
            objective_name=objective.name,
            objective_description=objective.description,
            scorer_name=scorer_name,
            scorer_description=scorer_meta.get("description", "")
        )

        try:
            k = 0
            while True:
                try:
                    # Call LLM for pairwise comparison
                    response = self.llm_client.call_with_prompt(
                        user_prompt=prompt,
                        system_prompt=PAIRWISE_COMPARISON_SYSTEM_PROMPT,
                    )

                    # Parse LLM response
                    result = self._parse_pairwise_response(response['content'])
                    break
                except Exception as e:
                    k += 1
                    if k > self.max_llm_retries:
                        logger.error(f"Pairwise comparison failed for objective '{objective.name}' and scorer '{scorer_name}' after {k} attempts: {e}")
                        raise e
                    else:
                        logger.debug(f"Pairwise comparison failed for objective '{objective.name}' and scorer '{scorer_name}' on attempt {k}/{self.max_llm_retries + 1}: {e}")
                        time.sleep(self.llm_config.retry_delay * (2 ** k))

            if result:
                result["scorer_name"] = scorer_name
                return result
            else:
                logger.warning(f"Failed to parse pairwise comparison response for objective '{objective.name}' and scorer '{scorer_name}'")
                return {"match": False, "reasoning": "Failed to parse LLM response", "related": False, "scorer_name": scorer_name}

        except Exception as e:
            logger.error(f"Pairwise comparison failed for objective '{objective.name}' and scorer '{scorer_name}': {e}")
            if self.dev:
                raise
            return {"match": False, "reasoning": f"Error during comparison: {e}", "related": False, "scorer_name": scorer_name}

    async def _select_best_from_multiple_matches_async(self, objective: Objective, matched_scorers: List[Dict[str, Any]]) -> Optional[str]:
        """
        Select the best scorer from multiple matches using LLM confidence scoring (async version).

        Args:
            objective: The objective being matched
            matched_scorers: List of matched scorer dictionaries with metadata

        Returns:
            Name of the best scorer, or None if selection fails
        """
        if not matched_scorers:
            return None

        if len(matched_scorers) == 1:
            return matched_scorers[0]["scorer_name"]

        # Format matched scorers for prompt
        scorer_descriptions = []
        for i, scorer in enumerate(matched_scorers):
            scorer_descriptions.append(f"{i+1}. {scorer['scorer_name']}")
            scorer_descriptions.append(f"   Description: {scorer.get('description', '')}")
            scorer_descriptions.append(f"   Reasoning for match: {scorer.get('reasoning', '')}")
            scorer_descriptions.append("")

        # Create prompt for multi-scorer selection
        prompt = MULTI_SCORER_SELECTION_USER_PROMPT_TEMPLATE.format(
            objective_name=objective.name,
            objective_description=objective.description,
            matched_scorers='\n'.join(scorer_descriptions)
        )

        try:
            for k in range(self.max_llm_retries + 1):
                try:
                    # Call LLM for confidence scoring
                    response = await asyncio.to_thread(
                        self.llm_client.call_with_prompt,
                        user_prompt=prompt,
                        system_prompt=MULTI_SCORER_SELECTION_SYSTEM_PROMPT
                    )

                    # Parse LLM response
                    result = self._parse_multi_scorer_response(response['content'])
                    break
                except Exception as e:
                    if k >= self.max_llm_retries:
                        logger.error(f"Multi-scorer selection failed for objective '{objective.name}' after {k + 1} attempts: {e}")
                        raise e
                    else:
                        logger.debug(f"Multi-scorer selection failed for objective '{objective.name}' on attempt {k + 1}/{self.max_llm_retries + 1}: {e}")
                        await asyncio.sleep(self.llm_config.retry_delay * (2 ** k))

            if result and result.get("scorer_confidences"):
                # Verify and select best scorer (only confidence 100 is considered a valid match)
                selected_scorer = self._verify_and_select_best_scorer(
                    result["scorer_confidences"],
                    matched_scorers,
                    objective.name,
                    result.get("selected_scorer"),
                    result.get("selection_reasoning", "")
                )
                if selected_scorer:
                    return selected_scorer
                else:
                    logger.debug(f"No perfect match (confidence 100) found in multi-scorer selection for objective '{objective.name}'")
                    return None  # No valid exact match found
            else:
                logger.warning(f"Failed to get confidence scores for objective '{objective.name}'")
                return None

        except Exception as e:
            logger.error(f"Multi-scorer selection failed for objective '{objective.name}': {e}")
            if self.dev:
                raise
            # Fallback to first match
            logger.warning(f"Using first match as fallback for objective '{objective.name}'")
            return matched_scorers[0]["scorer_name"]

    def _verify_and_select_best_scorer(self, confidence_scores: List[Dict[str, Any]], matched_scorers: List[Dict[str, Any]], objective_name: str, llm_selected_scorer: Optional[str] = None, llm_selection_reasoning: str = "") -> Optional[str]:
        """
        Verify confidence scores and select the scorer with highest confidence.

        Args:
            confidence_scores: List of scorer confidence dictionaries
            matched_scorers: Original list of matched scorers
            objective_name: Name of the objective for logging
            llm_selected_scorer: LLM's explicit selection
            llm_selection_reasoning: LLM's reasoning for the selection

        Returns:
            Name of the best scorer, or None if verification fails
        """
        try:
            # Create a map of scorer names for validation
            valid_scorer_names = {scorer["scorer_name"] for scorer in matched_scorers}

            # Validate and extract confidence scores
            valid_confidences = []
            seen_confidences = set()

            for score_info in confidence_scores:
                scorer_name = score_info.get("scorer_name")
                confidence = score_info.get("confidence")

                # Validate scorer name
                if scorer_name not in valid_scorer_names:
                    logger.warning(f"LLM returned invalid scorer name '{scorer_name}' for objective '{objective_name}'")
                    continue

                # Validate confidence score
                if not isinstance(confidence, (int, float)) or confidence < 0 or confidence > 100:
                    logger.warning(f"LLM returned invalid confidence score {confidence} for scorer '{scorer_name}' in objective '{objective_name}'")
                    continue

                # Check for duplicate confidence scores
                if confidence in seen_confidences:
                    logger.warning(f"LLM returned duplicate confidence score {confidence} for objective '{objective_name}' - this violates the unique score requirement")
                    continue

                seen_confidences.add(confidence)
                valid_confidences.append((scorer_name, confidence, score_info.get("reasoning", "")))

            # Verify we have scores for all scorers
            if len(valid_confidences) != len(matched_scorers):
                logger.warning(f"LLM didn't provide valid confidence scores for all scorers for objective '{objective_name}' ({len(valid_confidences)}/{len(matched_scorers)})")

            if not valid_confidences:
                logger.error(f"No valid confidence scores received for objective '{objective_name}'")
                return None

            # Only accept confidence 100 as a valid exact match
            perfect_matches = [conf for conf in valid_confidences if conf[1] == 100]

            if perfect_matches:
                if len(perfect_matches) == 1:
                    best_scorer_name, best_confidence, best_reasoning = perfect_matches[0]

                    # Cross-validate with LLM's explicit selection
                    if llm_selected_scorer and llm_selected_scorer != best_scorer_name:
                        logger.warning(f"LLM selected '{llm_selected_scorer}' but highest confidence (100) is '{best_scorer_name}' for objective '{objective_name}'")
                        # Trust the confidence 100 scorer over LLM's explicit selection

                    logger.debug(f"Found perfect match: scorer '{best_scorer_name}' with confidence {best_confidence} for objective '{objective_name}': {best_reasoning}")
                    if llm_selection_reasoning:
                        logger.debug(f"LLM selection reasoning: {llm_selection_reasoning}")
                    return best_scorer_name
                else:
                    # Multiple perfect matches - this shouldn't happen but handle gracefully
                    logger.warning(f"Multiple perfect matches (confidence 100) found for objective '{objective_name}', selecting first one")
                    best_scorer_name, best_confidence, best_reasoning = perfect_matches[0]
                    return best_scorer_name
            else:
                # No perfect matches - all are related scorers
                logger.debug(f"No perfect matches (confidence 100) found for objective '{objective_name}', all scorers are related only")
                return None

        except Exception as e:
            logger.error(f"Error verifying confidence scores for objective '{objective_name}': {e}")
            if self.dev:
                raise
            return None

    def _parse_pairwise_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response for pairwise comparison.

        Args:
            response: LLM response string

        Returns:
            Parsed comparison result or None if parsing fails
        """
        try:
            # Extract JSON from response
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if not match:
                logger.debug("No <answer> tags found in pairwise comparison response")
                raise ValueError("No <answer> tags found in LLM response")

            response_json = match.group(1)
            result = json.loads(response_json)

            # Validate required fields
            parsed_result = {
                "match": result["match"],
                "reasoning": result.get("reasoning", ""),
                "related": result.get("related", False)
            }

            return parsed_result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse pairwise comparison response: {e}")
            if self.dev:
                raise e  # Raise error in dev mode for debugging

        return None

    def _parse_multi_scorer_response(self, response: str) -> Optional[Dict[str, Any]]:
        """
        Parse the LLM response for multi-scorer confidence scoring.

        Args:
            response: LLM response string

        Returns:
            Parsed confidence scores result or None if parsing fails
        """
        try:
            # Extract JSON from response
            match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if not match:
                logger.debug("No <answer> tags found in multi-scorer selection response")
                raise ValueError("No <answer> tags found in LLM response")

            response_json = match.group(1)
            result = json.loads(response_json)

            # Validate required fields
            parsed_result = {
                "scorer_confidences": result.get("scorer_confidences", []),
                "selected_scorer": result.get("selected_scorer"),
                "selection_reasoning": result.get("selection_reasoning", "")
            }

            return parsed_result

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.debug(f"Failed to parse multi-scorer selection response: {e}")
            if self.dev:
                raise e  # Raise error in dev mode for debugging

        return None

    async def _create_scorer_with_llm_async(self, objective: Objective, potential_matches: Optional[List[str]] = None, serializer_name: str = None, test_candidates: Optional[List[Candidate]] = None) -> Optional[Callable]:
        """
        Create a new scorer using LLM via the coding agent (async version).

        Args:
            objective: The objective to create a scorer for
            potential_matches: List of potential matching scorer names for context
            serializer_name: Name of the serializer to use for getting sample schema and description (required)
            test_candidates: Optional list of example candidates for testing

        Returns:
            Created scorer function if successful, None otherwise
        """
        logger.info(f"Creating new scorer using coding agent (async) for objective '{objective.name}'")

        if not self.enable_llm_scorer_creation:
            logger.critical(f"LLM scorer creation is disabled when initializing the scorer creator")
            raise ValueError(f"LLM scorer creation is disabled when initializing the scorer creator")

        if not serializer_name:
            logger.error(f"serializer_name is required for scorer creation but was not provided for objective '{objective.name}'")
            return None

        try:
            # Get sample schema and description from serializer
            try:
                serializer = get_serializer(serializer_name)
                if not serializer:
                    logger.error(f"Serializer '{serializer_name}' not found")
                    return None

                sample_schema = serializer.sample_schema
                sample_description = serializer.sample_description
                logger.debug(f"Using serializer '{serializer_name}' - schema: {sample_schema}, description: {sample_description}")
            except Exception as e:
                logger.error(f"Error getting serializer '{serializer_name}': {e}")
                return None

            reference_module_paths = []
            for path in self.reference_module_paths:
                reference_module_paths.append(os.path.abspath(path))

            # Add potential matched scorer module paths to reference paths
            potential_matched_scorer_module = None
            if self.use_potential_matched_scorers_as_references and potential_matches:
                potential_matched_scorer_module = {}
                for scorer_name in potential_matches:
                    module_path = self.scorer_manager.get_module_path(scorer_name)
                    if module_path:
                        module_name = os.path.splitext(os.path.basename(module_path))[0]
                        potential_matched_scorer_module[scorer_name] = module_name
                        abs_path = os.path.abspath(module_path)
                        if abs_path not in reference_module_paths:
                            reference_module_paths.append(abs_path)

            # Call the coding agent to implement the scorer (async)
            result = await self.scorer_implementor.process(
                name=objective.name,
                description=objective.description,
                type=objective.type,
                serializer_name=serializer_name,
                sample_schema=sample_schema,
                sample_description=sample_description,
                reference_module_paths=reference_module_paths,
                potential_matched_scorer_module=potential_matched_scorer_module,
                test_candidates=test_candidates,
            )

            generated_library_path, _, scorer_name, implementation_success, coding_agent_usage_stats = result

            # Update Coding Agent usage stats to LLMClient stats
            if self.llm_client is None:
                self.llm_client = LLMClient(None)
            coding_agent_model_name = coding_agent_usage_stats.get("model_name") + '[coding agent]'
            if coding_agent_model_name not in self.llm_client.stats:
                self.llm_client.stats[coding_agent_model_name] = {"call_count": 0, "total_tokens": 0, "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            for key in ('total_tokens', 'input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens', 'cost'):
                self.llm_client.stats[coding_agent_model_name][key] += coding_agent_usage_stats[key]

            if not implementation_success:
                logger.info(f"Scorer creation failed for objective '{objective.name}' - coding agent could not implement it")
                return None

            # Load the generated scorer module
            module_path = os.path.join(generated_library_path, scorer_name)
            self.scorer_manager.register_mcp_module(module_path, serializer_name)

            logger.info(f"Successfully created and registered scorer for objective '{objective.name}'")
            scorer_function = self.scorer_manager.get_scorer(scorer_name)
            return scorer_function

        except Exception as e:
            logger.error(f"Error creating scorer with LLM for objective '{objective.name}': {e}")
            if self.dev:
                raise
            return None