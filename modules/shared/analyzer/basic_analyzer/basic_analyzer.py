"""
General analyzer module for the SciLeo Agent framework.

This module provides a general-purpose analyzer that uses LLM to analyze
optimization results and generate comprehensive reports.
"""

from calendar import c
from typing import List, Dict, Any, Optional
import os
import re
import json
from pathlib import Path
from datetime import datetime

from scileo_agent.core.modules import AnalyzerModule
from scileo_agent.core.data_models import Population, Objective
from scileo_agent.core.registry.module_registry import register_module
from scileo_agent.core.registry.serializer_registry import get_serializer
from scileo_agent.utils.logging import get_logger
from scileo_agent.utils import LLMClient
from scileo_agent.utils.human_feedback import (
    get_multiline_input,
    validate_json,
    confirm_input
)
from .candidate_analyzer import CandidateAnalyzer

logger = get_logger()


ANALYZER_SYSTEM_PROMPT = """You are an expert optimization analyst. You will be given information about an optimization process, including its high-level goal, current objective(s), and population results. Your task is to:

1. Analyze the current state of the optimization and provide a clear, concise report that will help guide future decisions.
2. Decide whether the optimization has reached a good stopping point and should be terminated.

Both outputs are critical for guiding the optimization process.

Note: When you see "calculation failed" in the results, this means that some objective evaluations could not be completed for certain candidates. This can happen when objectives are not applicable to certain candidates, or when there are technical limitations/issues in the evaluation process.

Your analysis report must include the following four sections:

1. **Overview**: 
   - Briefly summarize the current iteration and progress so far.
   - Highlight key objectives and notable characteristics of the population.

2. **Performance Analysis**:
   - Analyze how the population is performing on the objectives.
   - Highlight improvements, regressions, and notable trends.
   - Include any useful statistics or observations.

3. **Issues and Concerns**:
   - Point out potential problems such as stagnation, poor diversity, or conflicting objectives.

4. **Strategic Recommendations**:
   - Provide actionable, realistic suggestions for the next iteration, focusing only on things that can be done at the objective level (e.g., adding/removing objectives or minor focus adjustments).
   - Avoid suggesting actions outside of what is possible (e.g., filtering candidates, altering the optimization algorithm).

Output format specification (strict):

1) Wrap the analysis report in <report>...</report> tags.
2) After </report>, output exactly these two tags, in this order, one per line:
   <should_stop>true</should_stop> or <should_stop>false</should_stop>
   <reasoning>One to three sentences explaining why to stop or continue</reasoning>

Formatting rules:
- The value inside <should_stop> must be a lowercase literal: "true" or "false" only.
- Do not include any other tags, code fences, or markdown outside the specified tags.
- Do not repeat fields; each tag must appear exactly once.
"""


ANALYZER_USER_PROMPT_TEMPLATE = """# Optimization Analysis Request

## Optimization Goal
{high_level_goal}
{context_information_section}
## Current Optimization Status
{optimization_status}
{random_replacement_context}

## Objectives Configuration
{objectives_info}

## Performance Results
{results}

{candidate_analysis_section}

## Analysis Instructions
Follow the output format strictly:
1) Provide the analysis report inside <report>...</report>.
2) After </report>, output exactly:
   <should_stop>true</should_stop> or <should_stop>false</should_stop>
   <reasoning>One to three sentences explaining the decision</reasoning>
"""


ANALYZER_TOOL_SELECTION_SYSTEM_PROMPT = "You are a helpful assistant that selects relevant scientific tools based on optimization goals."


ANALYZER_TOOL_SELECTION_PROMPT_TEMPLATE = """You are a tool selection expert for scientific optimization tasks. You will be given:
1. An optimization goal
2. Context information about the task
3. Information about the candidates being analyzed
4. A complete list of available scientific tools from ToolUniverse

Your task is to select ALL tools that could be potentially useful for analyzing candidates in this optimization task. Be inclusive rather than restrictive - it's better to include a tool that might be useful than to exclude one that could help.

# Optimization Goal
{high_level_goal}

# Context Information
{context_information}

# Candidates Being Analyzed
Each candidate representation ({candidate_type}) is {candidate_description}{example_candidate}

# Available Tools from ToolUniverse
{tools_text}

# Your Task
Analyze the optimization goal and candidate type to select ALL tools that could be relevant for:
- Evaluating candidate quality and properties
- Computing domain-specific metrics
- Running simulations or predictions
- Validating candidates against scientific criteria
- Any other analysis that could provide insights into candidate performance

Think through your reasoning step by step, then provide your final selection.

# Output Format
After your analysis, provide your selection inside <tool_selection>...</tool_selection> tags as a JSON object with:
1. "selected_tools": A list of the EXACT tool names (as shown above) that are relevant
2. "reasoning": A brief explanation of why these categories of tools were selected

Example:
First, think about the domain and what tools would be useful...
[your analysis here]

<tool_selection>
{{
  "selected_tools": ["Tool_Name_1", "Tool_Name_2", "Tool_Name_3"],
  "reasoning": "Selected tools for drug discovery include ADMET prediction tools, toxicity assessment, molecular property calculators, and binding affinity predictors."
}}
</tool_selection>

IMPORTANT:
- Use the EXACT tool names from the list above
- Be inclusive - select all potentially useful tools
- Focus on tools relevant to the specific optimization domain and candidate type
- Consider both direct and indirect relevance"""


@register_module("basic_analyzer", "0.8.0")
class BasicAnalyzer(AnalyzerModule):
    """
    General analyzer module that uses LLM to analyze optimization results.
    
    This module analyzes the current state of optimization and generates
    comprehensive reports for the planner module to make decisions.
    """
    
    def __init__(self, module_id: str, config: Dict[str, Any] = None, llm_config=None):
        """
        Initialize the general analyzer module.

        Args:
            module_id: Unique identifier for this module
            config: Configuration parameters
            llm_config: LLM configuration
        """
        super().__init__(module_id, config, llm_config)

        self.population_save_dir = self.config.get("population_save_dir", None)
        if self.population_save_dir:
            Path(self.population_save_dir).mkdir(parents=True, exist_ok=True)

        self.enable_human_feedback = self.config.get("enable_human_feedback", False)
        self.max_llm_retries = self.config.get("max_llm_retries", 3)

        # Refusal detection configuration
        self.enable_refusal_detection = self.config.get("enable_refusal_detection", True)
        self.refusal_detection_model = self.config.get("refusal_detection_model_name", "openai/gpt-4.1-nano-2025-04-14")

        # Initialize candidate analyzer
        self.enable_candidate_analysis = self.config.get("enable_candidate_analysis", True)
        if self.enable_candidate_analysis:
            self.candidate_analyzer = CandidateAnalyzer(self.config)
            logger.debug("Initialized CandidateAnalyzer subagent")
        else:
            self.candidate_analyzer = None
            logger.debug("CandidateAnalyzer disabled")

        # Domain tools configuration
        self.enable_domain_tools = self.config.get("candidate_analyzer_enable_domain_tools", True)
        self.tool_selection_model = self.config.get("candidate_analyzer_tool_selection_model", "anthropic/claude-sonnet-4-5-20250929")

        # Tool selection caching - track optimization parameters to detect changes
        self._last_tool_selection_params = None  # Will store (high_level_goal, context_information, serializer_name)
        self._cached_selected_tools = None  # Cached tool selection result

    async def _select_domain_tools(
        self,
        high_level_goal: str,
        context_information: str,
        serializer_name: str,
        current_population: Population,
    ) -> Optional[List[str]]:
        """
        Select relevant domain tools using LLM and actual ToolUniverse tool information.

        Uses caching to avoid re-selecting tools when optimization parameters haven't changed.

        Args:
            high_level_goal: The overall optimization goal
            context_information: Additional context about the task
            serializer_name: Name of serializer to get candidate type/description
            current_population: Current population to get example candidates

        Returns:
            List of selected tool names, or None if selection fails
        """
        if not self.enable_domain_tools:
            return None

        # Check if we can use cached tool selection
        current_params = (high_level_goal, context_information, serializer_name)
        if self._last_tool_selection_params == current_params and self._cached_selected_tools is not None:
            logger.debug("Using cached tool selection (optimization parameters unchanged)")
            return self._cached_selected_tools

        # Parameters have changed, need to re-select tools
        if self._last_tool_selection_params is not None:
            logger.info("Optimization parameters changed, re-selecting domain tools...")
        else:
            logger.debug("Selecting relevant domain tools from ToolUniverse catalog...")

        try:
            # Get candidate type and description from serializer
            serializer = get_serializer(serializer_name)
            candidate_type = serializer.sample_schema
            candidate_description = serializer.sample_description

            # Get example candidate from population
            example_candidate = ""
            if current_population and current_population.size > 0:
                first_candidate = current_population.candidates[0]
                example_candidate_repr = serializer.serialize(first_candidate)
                example_candidate = f"\n\n**Example candidate**:\n`{example_candidate_repr}`"

            # Load the ToolUniverse tool catalog
            tools_file = os.path.join(
                os.path.dirname(__file__),
                "tool_universe_tools.json"
            )

            if not os.path.exists(tools_file):
                logger.critical(f"ToolUniverse catalog not found at: {tools_file}")
                raise FileNotFoundError(f"ToolUniverse catalog not found at: {tools_file}")

            with open(tools_file, 'r') as f:
                all_tools = json.load(f)

            # Format tool information for the LLM (name + description only, to save tokens)
            tools_info = []
            for i, tool in enumerate(all_tools):
                tools_info.append(f"{i+1}. **{tool['name']}**: {tool['description']}")

            tools_text = "\n".join(tools_info)

            # Create the tool selection prompt
            prompt = ANALYZER_TOOL_SELECTION_PROMPT_TEMPLATE.format(
                high_level_goal=high_level_goal,
                context_information=context_information if context_information else "No additional context provided.",
                candidate_type=candidate_type,
                candidate_description=candidate_description,
                example_candidate=example_candidate,
                tools_text=tools_text
            )

            # Make the LLM call using self.call_llm_async
            logger.debug("Calling LLM for tool selection...")
            response = await self.call_llm_with_prompt_async(
                prompt,
                system_prompt=ANALYZER_TOOL_SELECTION_SYSTEM_PROMPT,
                model_name=self.tool_selection_model,
            )

            # Parse the response - extract JSON from <tool_selection> tags
            content = response['content']

            # Extract the tool selection JSON from tags
            match = re.search(r'<tool_selection>(.*?)</tool_selection>', content, re.DOTALL)
            if not match:
                raise ValueError("No <tool_selection>...</tool_selection> tags found in LLM response")

            json_str = match.group(1).strip()
            selection_result = json.loads(json_str)

            selected_tools = selection_result.get("selected_tools", [])
            reasoning = selection_result.get("reasoning", "No reasoning provided")

            # Validate that selected tools exist in the catalog
            valid_tool_names = {tool['name'] for tool in all_tools}
            validated_tools = [t for t in selected_tools if t in valid_tool_names]

            if len(validated_tools) < len(selected_tools):
                invalid_tools = set(selected_tools) - set(validated_tools)
                logger.warning(f"LLM selected {len(invalid_tools)} invalid tool names: {invalid_tools}")

            logger.debug(f"Tool selection reasoning: {reasoning}")
            logger.debug(f"Selected {len(validated_tools)} tools from ToolUniverse")
            logger.debug(f"Selected tools: {validated_tools}")

            # Check which API keys are required for the selected tools
            required_api_keys = set()
            for tool in all_tools:
                if tool['name'] in validated_tools:
                    if 'required_api_keys' in tool:
                        required_api_keys.update(tool['required_api_keys'])

            # Only warn if required API keys are actually missing from environment
            if required_api_keys:
                missing_api_keys = [key for key in required_api_keys if not os.getenv(key)]
                if missing_api_keys:
                    logger.warning(f"Selected tools require the following environment variables that are not set: {sorted(missing_api_keys)}")
                    logger.warning("Tools requiring these API keys will not be loaded. Set them in your environment if needed.")
                else:
                    logger.debug(f"All required API keys are set: {sorted(required_api_keys)}")
            else:
                logger.debug("Selected tools do not require any API keys")

            # Cache the results
            self._last_tool_selection_params = current_params
            self._cached_selected_tools = validated_tools

            return validated_tools

        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            # Don't cache failures
            return None

    async def analyze(
        self,
        iteration_number: int,
        high_level_goal: str,
        context_information: str,
        current_population: Population,
        current_objectives: List[Objective],
        serializer_name: str,
        historical_info: Optional[Dict[str, Any]] = None,
        random_replacement_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze the current state and generate a comprehensive report, plus a stop decision.

        Args:
            iteration_number: The current iteration number
            high_level_goal: The overall optimization goal
            context_information: Additional context information
            current_population: Current population of candidates
            current_objectives: List of current objectives used in this iteration
            serializer_name: Name of serializer to use for saving population
            historical_info: Optional historical information from previous iterations
            random_replacement_info: Optional information about random candidate replacement
                Dict with keys:
                  - occurred (bool): Whether replacement happened
                  - ratio (float): Replacement ratio
                  - num_replaced (int): Number of candidates replaced

        Returns:
            Dict with keys:
              - analysis_report (str)
              - should_stop (bool)
              - reasoning (str)
              - population_file (str, if population_save_dir is configured)
              - original_llm_analysis (dict, if human feedback is enabled)
              - candidate_analysis_workspace (str, if candidate analysis is enabled and ran)
              - candidate_analysis_usage (dict, if candidate analysis is enabled and ran)
        """
        historical_info = historical_info or {}
        random_replacement_info = random_replacement_info or {}

        last_population = historical_info.get("last_population", None)

        # Step 1: Save population to file (always, if save_dir configured)
        population_file = None
        if self.population_save_dir:
            population_file = self._save_population_to_file(
                current_population,
                iteration_number,
                serializer_name
            )

        # Step 2: Prepare formatted information for both analyzers
        optimization_status = self._format_optimization_status(iteration_number, current_population, last_population)
        objectives_info = self._format_objectives_info(current_objectives)
        results = await self._format_results(current_population, last_population, current_objectives)
        random_replacement_context = self._format_random_replacement_context(random_replacement_info)

        # Step 3: Select domain tools if enabled
        selected_domain_tools = None
        if self.enable_domain_tools and self.enable_candidate_analysis:
            selected_domain_tools = await self._select_domain_tools(
                high_level_goal,
                context_information,
                serializer_name,
                current_population
            )

        # Step 4: Run candidate analysis if enabled
        candidate_analysis_report = None
        candidate_analysis_usage = None

        if self.enable_candidate_analysis and self.candidate_analyzer and population_file:
            try:
                logger.debug("Running candidate analysis subagent...")
                candidate_analysis_result = await self.candidate_analyzer.analyze(
                    iteration_number=iteration_number,
                    high_level_goal=high_level_goal,
                    context_information=context_information,
                    current_population=current_population,
                    population_file=population_file,
                    serializer_name=serializer_name,
                    objectives_info=objectives_info,
                    performance_results=results,
                    selected_domain_tools=selected_domain_tools,
                )
                candidate_analysis_report = candidate_analysis_result.get("candidate_analysis_report")
                candidate_analysis_usage = candidate_analysis_result.get("usage_stats")
                # logger.info("Candidate analysis completed successfully")
            except Exception as e:
                logger.error(f"Candidate analysis failed: {str(e)}")
                # Continue without candidate analysis
                candidate_analysis_report = None
                raise e

        # Step 4: Get LLM analysis with retry logic (using pre-formatted information)
        llm_analysis = await self._get_llm_analysis(
            high_level_goal,
            context_information,
            optimization_status,
            objectives_info,
            results,
            random_replacement_context,
            candidate_analysis_report
        )

        llm_report_text = llm_analysis["analysis_report"]
        llm_should_stop_value = llm_analysis["should_stop"]
        llm_reasoning_text = llm_analysis["reasoning"]

        # Step 3: Get human feedback if enabled
        if self.enable_human_feedback:
            logger.info("Human feedback is enabled - requesting feedback on analysis")

            if not population_file:
                logger.warning("Population save directory not configured - human cannot review population file")
                population_file = "Not saved (population_save_dir not configured)"

            # Get human-revised analysis
            revised_analysis = await self._get_human_revised_analysis(
                llm_report_text,
                llm_should_stop_value,
                llm_reasoning_text,
                population_file
            )

            logger.info("Human feedback on analysis received and validated successfully")

            # Use human-revised version
            final_report = revised_analysis["analysis_report"]
            final_should_stop = revised_analysis["should_stop"]
            final_reasoning = revised_analysis["reasoning"]

            # Store original LLM analysis for record
            result = {
                "analysis_report": final_report,
                "should_stop": final_should_stop,
                "reasoning": final_reasoning,
                "candidate_analysis_report": candidate_analysis_report,
                "original_llm_analysis": {
                    "analysis_report": llm_report_text,
                    "should_stop": llm_should_stop_value,
                    "reasoning": llm_reasoning_text
                }
            }
        else:
            # No human feedback - use LLM analysis directly
            result = {
                "analysis_report": llm_report_text,
                "should_stop": llm_should_stop_value,
                "reasoning": llm_reasoning_text,
                "candidate_analysis_report": candidate_analysis_report,
            }

        # Add population file path to result if available
        if population_file:
            result["population_file"] = population_file

        # Merge candidate analysis usage stats to LLMClient stats
        if candidate_analysis_usage:
            if self.llm_client is None:
                self.llm_client = LLMClient(None)
            candidate_analyzer_model_name = candidate_analysis_usage.get("model_name") + '[candidate analyzer]'
            if candidate_analyzer_model_name not in self.llm_client.stats:
                self.llm_client.stats[candidate_analyzer_model_name] = {"call_count": 0, "total_tokens": 0, "input_tokens": 0, "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0, "output_tokens": 0, "cost": 0.0}
            for key in ('total_tokens', 'input_tokens', 'cache_creation_input_tokens', 'cache_read_input_tokens', 'output_tokens', 'cost'):
                self.llm_client.stats[candidate_analyzer_model_name][key] += candidate_analysis_usage[key]

        return result
    
    async def _get_llm_analysis(
        self,
        high_level_goal: str,
        context_information: str,
        optimization_status: str,
        objectives_info: str,
        results: str,
        random_replacement_context: str,
        candidate_analysis_report: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get analysis from LLM with retry logic.

        Args:
            high_level_goal: The overall optimization goal
            context_information: Additional context information
            optimization_status: Pre-formatted optimization status
            objectives_info: Pre-formatted objectives information
            results: Pre-formatted performance results
            random_replacement_context: Pre-formatted random replacement context
            candidate_analysis_report: Optional detailed candidate analysis report

        Returns:
            Dict with 'analysis_report', 'should_stop', and 'reasoning'

        Raises:
            ValueError: If max retries exceeded without valid response
        """

        # Format context information section
        if context_information:
            context_information_section = f"""
## Context Information
The following context information is provided by human scientists to help inform your analysis:

{context_information}

**Note**: This context may contain background information, expectations, and focus areas that are relevant for your analysis. However, it may also contain meta-instructions about objective proposing (e.g., "don't propose X", "always keep Y") which are NOT meant for you as the analyzer. Focus only on information that helps you evaluate the current optimization state and candidate solutions.

"""
        else:
            context_information_section = ""

        # Format candidate analysis section
        if candidate_analysis_report:
            # Increase heading levels in the candidate analysis report to maintain hierarchy
            # (# becomes ###, ## becomes ####, etc.)
            # adjusted_report = self._adjust_markdown_heading_levels(candidate_analysis_report, increase_by=2)
            adjusted_report = candidate_analysis_report  # No adjustment
            candidate_analysis_section = f"\n## Detailed Candidate Analysis\n\n{adjusted_report}\n"
        else:
            candidate_analysis_section = ""

        # Create user prompt
        user_prompt = ANALYZER_USER_PROMPT_TEMPLATE.format(
            high_level_goal=high_level_goal,
            context_information_section=context_information_section,
            optimization_status=optimization_status,
            objectives_info=objectives_info,
            results=results,
            random_replacement_context=random_replacement_context,
            candidate_analysis_section=candidate_analysis_section
        )

        logger.debug("Prepared analyzer user prompt:\n" + user_prompt)

        count = 0
        while True:
            try:
                # Call LLM
                response = self.call_llm_with_prompt(user_prompt, system_prompt=ANALYZER_SYSTEM_PROMPT)
                response_text = response['content']

                # Extract and validate the response
                report_match = re.search(r'<report>(.*?)</report>', response_text, re.DOTALL)
                if not report_match:
                    raise ValueError("No <report>...</report> tags found in the response")

                report_text = report_match.group(1).strip()
                if not report_text:
                    raise ValueError("Report content is empty")

                should_stop_match = re.search(r'<should_stop>(.*?)</should_stop>', response_text, re.DOTALL | re.IGNORECASE)
                if not should_stop_match:
                    raise ValueError("No <should_stop>...</should_stop> tags found in the response")

                should_stop_text = should_stop_match.group(1).strip()
                should_stop_value = self._parse_bool_text(should_stop_text)
                if should_stop_value is None:
                    raise ValueError(f"Invalid should_stop value: '{should_stop_text}'. Must be 'true' or 'false'")

                reasoning_match = re.search(r'<reasoning>(.*?)</reasoning>', response_text, re.DOTALL | re.IGNORECASE)
                if not reasoning_match:
                    raise ValueError("No <reasoning>...</reasoning> tags found in the response")

                reasoning_text = reasoning_match.group(1).strip()
                if not reasoning_text:
                    raise ValueError("Reasoning content is empty")
                
                # Check for refusal using a small LLM (if enabled)
                if self.enable_refusal_detection and self._detect_refusal(response_text):
                    raise ValueError("LLM refused to provide a proper response due to safety/compliance constraints (likely overrefusal)")

                # Successfully parsed valid response
                return {
                    "analysis_report": report_text,
                    "should_stop": should_stop_value,
                    "reasoning": reasoning_text
                }

            except ValueError as e:
                logger.error(f"LLM analysis response validation failed (attempt {count + 1})", {
                    "error": str(e),
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                })

                if count >= self.max_llm_retries:
                    logger.critical("Failed to get valid LLM analysis after maximum retries", {
                        "max_retries": self.max_llm_retries,
                        "final_error": str(e)
                    })
                    raise ValueError(f"Failed to get valid LLM analysis after {self.max_llm_retries} retries. Last error: {str(e)}")

                count += 1
                continue

    def _parse_bool_text(self, text: Optional[str]) -> Optional[bool]:
        """
        Parse boolean text.

        Args:
            text: Text to parse

        Returns:
            Boolean value or None if invalid
        """
        if text is None:
            return None
        normalized = text.strip().lower()
        if normalized in {"true", "yes", "y", "1"}:
            return True
        if normalized in {"false", "no", "n", "0"}:
            return False
        return None

    def _detect_refusal(self, response_text: str) -> bool:
        """
        Detect if the LLM response contains a refusal due to safety/compliance constraints.

        Uses a small LLM to analyze the response text and determine if it's a refusal.
        The model used for detection is configurable via the 'refusal_detection_model' config parameter.

        Args:
            response_text: The response text from the LLM to check

        Returns:
            True if refusal is detected, False otherwise
        """
        detection_prompt = f"""You are a refusal detector. Your task is to determine if the following LLM response is refusing to complete the requested task due to safety, compliance, or policy constraints.

A refusal typically includes phrases like:
- "I cannot", "I can't", "I'm unable to"
- "Due to safety and compliance constraints"
- References to safety concerns, policy violations, or ethical issues
- Apologetic language followed by declining the task
- Suggesting the request is inappropriate or harmful

The LLM response to analyze:
---
{response_text}
---

Think through whether this is a refusal, then output your judgment in the format:
<judgment>yes</judgment> or <judgment>no</judgment>"""

        try:
            # Use the configured model for detection (defaults to haiku for speed/cost)
            detection_response = self.call_llm_with_prompt(
                detection_prompt,
                system_prompt="You are a helpful assistant that detects refusals in LLM responses.",
                model=self.refusal_detection_model
            )

            response_content = detection_response['content']

            # Extract the judgment from XML tags
            judgment_match = re.search(r'<judgment>(.*?)</judgment>', response_content, re.DOTALL | re.IGNORECASE)

            if not judgment_match:
                logger.warning("No <judgment> tag found in refusal detection response, assuming no refusal")
                return False

            judgment = judgment_match.group(1).strip().lower()

            # Check if the judgment indicates a refusal
            if judgment == 'yes':
                logger.warning("Refusal detected in LLM response", {
                    "detection_model": self.refusal_detection_model
                })
                return True

            return False

        except Exception as e:
            logger.warning(f"Failed to detect refusal (assuming no refusal): {str(e)}")
            # If detection fails, assume no refusal to avoid false positives
            return False

    def _adjust_markdown_heading_levels(self, text: str, increase_by: int = 1) -> str:
        """
        Adjust markdown heading levels by adding additional # symbols.

        Args:
            text: Markdown text to adjust
            increase_by: Number of levels to increase (default: 1)

        Returns:
            Adjusted markdown text with increased heading levels

        Example:
            "# Title" with increase_by=2 becomes "### Title"
            "## Section" with increase_by=2 becomes "#### Section"
        """
        if increase_by <= 0:
            return text

        lines = text.split('\n')
        adjusted_lines = []

        for line in lines:
            # Check if line starts with markdown heading (one or more #)
            if line.strip().startswith('#'):
                # Count leading #'s
                stripped = line.lstrip()
                hash_count = 0
                for char in stripped:
                    if char == '#':
                        hash_count += 1
                    else:
                        break

                # Add additional #'s and keep the rest of the line
                if hash_count > 0 and hash_count < len(stripped):
                    # Ensure we don't exceed markdown's heading limit (h6 = ######)
                    new_hash_count = min(hash_count + increase_by, 6)
                    new_hashes = '#' * new_hash_count
                    rest_of_line = stripped[hash_count:]
                    # Preserve original leading whitespace
                    leading_space = line[:len(line) - len(line.lstrip())]
                    adjusted_lines.append(f"{leading_space}{new_hashes}{rest_of_line}")
                else:
                    adjusted_lines.append(line)
            else:
                adjusted_lines.append(line)

        return '\n'.join(adjusted_lines)

    def _format_random_replacement_context(self, random_replacement_info: Optional[Dict[str, Any]]) -> str:
        """
        Format random replacement context for the prompt.

        Args:
            random_replacement_info: Dict with 'occurred', 'ratio', 'num_replaced' keys

        Returns:
            Formatted context string (empty if no replacement occurred)
        """
        if not random_replacement_info or not random_replacement_info.get("occurred", False):
            return ""

        ratio = random_replacement_info.get("ratio", 0.0)
        num_replaced = random_replacement_info.get("num_replaced", 0)

        context = f"\n**Important Context - Random Candidate Replacement:**\n"
        context += f"Before optimization in this iteration, {num_replaced} candidates ({ratio*100:.1f}%) from the previous iteration's output were randomly replaced with newly generated random candidates. "
        context += f"This means the optimizer received a modified input population where {num_replaced} positions contained fresh random candidates instead of candidates from the previous iteration. "
        context += f"When analyzing performance changes compared to the previous iteration, consider that:\n"
        context += f"- Performance decreases may be partially attributed to the injection of random candidates, which typically start with lower quality\n"
        context += f"- Diversity increases may be partially due to the random injection introducing new variation, rather than solely from optimizer improvements\n"
        context += f"- The optimizer had to work with a partially randomized starting population in this iteration\n"

        return context

    def _format_optimization_status(self, iteration_number: int, current_population: Population, last_population: Optional[Population]) -> str:
        """Format optimization status for the prompt."""
        if iteration_number > 1 and last_population is None:
            raise ValueError("Last population is required for non-first iterations.")

        num_candidates = current_population.size
        num_last_candidates = last_population.size if last_population else None

        if num_last_candidates is None:
            status = f"During iteration {iteration_number}, the optimizer produced {num_candidates} candidates."
        else:
            status = f"During iteration {iteration_number}, the optimizer took in {num_last_candidates} candidates and produced {num_candidates} candidates."

        return status
    
    def _format_objectives_info(self, objectives: List[Objective]) -> str:
        """Format objectives information for the prompt."""
        if not objectives:
            raise ValueError("No objectives defined.")

        # Determine which types are present
        present_types = set(obj.type for obj in objectives)

        # Only show type explanations for types that are actually present
        info_lines = []
        type_descriptions = []
        if "candidate-wise" in present_types:
            type_descriptions.append("Candidate-wise objectives score every candidate")
        if "population-wise" in present_types:
            type_descriptions.append("Population-wise objectives score the entire population as a whole")
        if "filter" in present_types:
            type_descriptions.append("Filter objectives pass/fail each candidate based on criteria (True=pass, False=fail)")

        if type_descriptions:
            info_lines.append("Objective types:")
            for desc in type_descriptions:
                info_lines.append(f"- {desc}")
            info_lines.append("")

        # Sort objectives by type for better organization
        type_order = {"candidate-wise": 0, "population-wise": 1, "filter": 2}
        objectives = sorted(objectives, key=lambda x: type_order.get(x.type, 3))

        for i, obj in enumerate(objectives, 1):
            info_lines.append(f"{i}. {obj.name}")
            info_lines.append(f"   Description: {obj.description}")
            info_lines.append(f"   Type: {obj.type}")
            if obj.type != "filter":
                info_lines.append(f"   Direction: {obj.optimization_direction}")
            if obj.weight:
                info_lines.append(f"   Weight: {obj.weight}")

        return "\n".join(info_lines)
    
    async def _calculate_results(self, population: Population, objectives: List[Objective]) -> Dict[str, Any]:
        """Calculate the results for the population.

        Note: Filter objectives are not analyzed here because the current_population
        is the result after filters have been applied, so all candidates have already
        passed the filters. Filter objectives are only listed in the objectives section.
        """

        population = await population.evaluate(objectives)
        
        results = {
            "candidate_wise": {},  # objective_name -> {mean, std, best, worst}
            "population_wise": {},  # objective_name -> {value}
        }
        for obj in objectives:
            if obj.type == "population-wise":
                results["population_wise"][obj.name] = {'value': population.get_score(obj.name)}
            elif obj.type == "filter":
                # Skip filter objectives - they were already applied during optimization
                # The current_population only contains candidates that passed the filters
                pass
            else:  # candidate-wise
                mean, std, none_count = population.get_regular_score_mean_and_std(obj.name)
                best_candidate, best_score, best_type = population.find_best_candidate(obj)
                worst_candidate, worst_score, worst_type = population.find_worst_candidate(obj)
                results["candidate_wise"][obj.name] = {
                    'mean': mean,
                    'std': std,
                    'none_count': none_count,
                    'best_candidate': best_candidate,
                    'best_score': best_score,
                    'best_type': best_type,
                    'worst_candidate': worst_candidate,
                    'worst_score': worst_score,
                    'worst_type': worst_type,
                    'optimization_direction': obj.optimization_direction
                }

        return results
    
    async def _format_results(self, current_population: Population, last_population: Optional[Population], objectives: List[Objective]) -> str:
        """Analyze the current population."""
        
        current_population_results = await self._calculate_results(current_population, objectives)
        if last_population:
            last_population_results = await self._calculate_results(last_population, objectives)
            num_last_population = last_population.size
        else:
            last_population_results = None
            num_last_population = None
        
        analysis_lines = []
        analysis_lines.append(f"Population size: {current_population.size}")
        idx = 1
        if 'candidate_wise' in current_population_results and len(current_population_results['candidate_wise']) > 0:
            analysis_lines.append(f"\nCandidate-wise objectives:")
            candidate_wise_results = current_population_results['candidate_wise']
            for obj_name in candidate_wise_results:
                obj_results = candidate_wise_results[obj_name]
                analysis_lines.append(f"{idx}. {obj_name}")
                
                this_mean = obj_results['mean']
                this_std = obj_results['std']
                none_count = obj_results['none_count']
                total_candidates = current_population.size
                valid_candidates = total_candidates - none_count
                
                # Handle None scores in mean and std
                if this_mean is None or this_std is None:
                    mean_std_line = f"   Mean ± Std: Not available (evaluation could not be completed for any of the {total_candidates} candidates)"
                else:
                    mean_std_line = f"   Mean ± Std: {this_mean:.4f} ± {this_std:.4f}"
                    if none_count > 0:
                        mean_std_line += f" (based on {valid_candidates} out of {total_candidates} candidates, evaluation incomplete for {none_count} candidates)"
                
                # Handle comparison with last iteration
                if last_population_results:
                    last_obj_results = last_population_results['candidate_wise'][obj_name]
                    last_mean = last_obj_results['mean']
                    last_std = last_obj_results['std']
                    last_none_count = last_obj_results['none_count']
                    last_total_candidates = num_last_population
                    last_valid_candidates = last_total_candidates - last_none_count
                    
                    # Only compare if both current and last have valid means
                    if this_mean is not None and last_mean is not None:
                        mean_abs_diff = abs(this_mean - last_mean)
                        comparison_line = f"compared to the last iteration ({last_mean:.4f} ± {last_std:.4f} based on {last_valid_candidates} out of {last_total_candidates} candidates"
                        if last_none_count > 0:
                            comparison_line += f", evaluation incomplete for {last_none_count} candidates"
                        comparison_line += "), "

                        direction = obj_results['optimization_direction']
                        if this_mean == last_mean:
                            comparison_line += "no change"
                        else:
                            if direction == "maximize":
                                if this_mean > last_mean:
                                    comparison_line += f"got better by {mean_abs_diff:.4f}"
                                else:
                                    comparison_line += f"got worse by {mean_abs_diff:.4f}"
                            else:
                                if this_mean > last_mean:
                                    comparison_line += f"got worse by {mean_abs_diff:.4f}"
                                else:
                                    comparison_line += f"got better by {mean_abs_diff:.4f}"
                        mean_std_line += f"; {comparison_line}"
                    elif this_mean is None and last_mean is None:
                        mean_std_line += f"; compared to the last iteration (evaluation incomplete for all {last_total_candidates} candidates), no change"
                    elif this_mean is None:
                        mean_std_line += f"; compared to the last iteration ({last_mean:.4f} ± {last_std:.4f} based on {last_valid_candidates} out of {last_total_candidates} candidates), got worse (evaluation incomplete for all current candidates)"
                    else:  # last_mean is None
                        mean_std_line += f"; compared to the last iteration (evaluation incomplete for all {last_total_candidates} candidates), got better (current mean: {this_mean:.4f})"
                
                analysis_lines.append(mean_std_line)
                
                # Handle best and worst scores
                best_score = obj_results['best_score']
                best_type = obj_results['best_type']
                worst_score = obj_results['worst_score']
                worst_type = obj_results['worst_type']
                
                if best_score is None:
                    analysis_lines.append(f"   Best score: Not available (evaluation incomplete for all candidates)")
                else:
                    analysis_lines.append(f"   Best score: {best_score:.4f} ({best_type})")
                
                if worst_score is None:
                    analysis_lines.append(f"   Worst score: Not available (evaluation incomplete for all candidates)")
                else:
                    analysis_lines.append(f"   Worst score: {worst_score:.4f} ({worst_type})")

                idx += 1
        if 'population_wise' in current_population_results and len(current_population_results['population_wise']) > 0:
            analysis_lines.append(f"\nPopulation-wise objectives:")
            population_wise_results = current_population_results['population_wise']
            for obj_name in population_wise_results:
                analysis_lines.append(f"{idx}. {obj_name}")
                this_value = population_wise_results[obj_name]['value']
                value_line = f"   Value: {this_value:.4f}"
                if last_population_results:
                    last_value = last_population_results['population_wise'][obj_name]['value']
                    value_abs_diff = abs(this_value - last_value)
                    comparison_line = f"compared to the last iteration ({last_value:.4f} on {num_last_population} candidates), "
                    if this_value == last_value:
                        comparison_line += "no change"
                    else:
                        # For population-wise objectives, we need to get the direction from the objectives
                        # since population-wise objectives don't have direction in the results
                        direction = "maximize"  # default assumption
                        for obj in objectives:
                            if obj.name == obj_name:
                                direction = obj.optimization_direction
                                break
                        
                        if direction == "maximize":
                            if this_value > last_value:
                                comparison_line += f"got better by {value_abs_diff:.4f}"
                            else:
                                comparison_line += f"got worse by {value_abs_diff:.4f}"
                        else:
                            if this_value > last_value:
                                comparison_line += f"got worse by {value_abs_diff:.4f}"
                            else:
                                comparison_line += f"got better by {value_abs_diff:.4f}"
                    value_line += f"; ({comparison_line})"
                analysis_lines.append(value_line)
                idx += 1

        # Note: Filter objectives are not displayed in performance results because
        # the current_population only contains candidates that passed the filters.
        # Filter objectives are already listed in the "Objectives Configuration" section.

        return "\n".join(analysis_lines)

    def _save_population_to_file(
        self,
        population: Population,
        iteration_number: int,
        serializer_name: str
    ) -> str:
        """
        Save population to a JSON file.

        Args:
            population: Population to save
            iteration_number: Current iteration number
            serializer_name: Name of serializer to use for candidates

        Returns:
            Path to the saved file
        """
        if not self.population_save_dir:
            raise ValueError("population_save_dir not configured")

        # Get serializer
        serializer = get_serializer(serializer_name)
        if serializer is None:
            raise ValueError(f"Serializer '{serializer_name}' not found")

        # Create filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = f"population_iter_{iteration_number}_{timestamp}.json"
        filepath = Path(self.population_save_dir) / filename

        # Prepare data
        population_data = {
            "iteration": iteration_number,
            # "timestamp": timestamp,
            "size": population.size,
            "population_scores": population.scores,  # Population-level scores
            "candidates": []
        }

        # Save each candidate with serialized representation
        for candidate in population.candidates:
            candidate_data = {
                "id": candidate.id,
                "representation": serializer.serialize(candidate),
                "scores": candidate.scores,
                # "metadata": candidate.metadata
            }
            population_data["candidates"].append(candidate_data)

        # Write to file
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(population_data, f, indent=2, default=str)

        logger.info(f"Saved population to {filepath}")
        return str(filepath)

    def _display_analysis_for_feedback(
        self,
        analysis_report: str,
        should_stop: bool,
        reasoning: str,
        population_file: str
    ) -> None:
        """
        Display LLM's analysis in a human-friendly format.

        Args:
            analysis_report: The analysis report text
            should_stop: Whether to stop optimization
            reasoning: Reasoning for the decision
            population_file: Path to the saved population file
        """
        print("\n" + "="*80)
        print("LLM GENERATED ANALYSIS")
        print("="*80)

        print("\n" + "-"*80)
        print("ANALYSIS REPORT")
        print("-"*80)
        print(analysis_report)

        print("\n" + "-"*80)
        print("TERMINATION DECISION")
        print("-"*80)
        print(f"Should Stop: {should_stop}")
        print(f"Reasoning: {reasoning}")

        print("\n" + "-"*80)
        print("POPULATION DATA")
        print("-"*80)
        print(f"The current population has been saved to:")
        print(f"  {population_file}")
        print("\nYou can review the population data to inform your revisions.")

        print("\n" + "="*80)

    async def _get_human_revised_analysis(
        self,
        llm_analysis_report: str,
        llm_should_stop: bool,
        llm_reasoning: str,
        population_file: str
    ) -> Dict[str, Any]:
        """
        Get human-revised analysis with validation and retries.

        This method collects feedback separately for the analysis report and termination decision,
        giving users fine-grained control over each component.

        Args:
            llm_analysis_report: LLM's analysis report
            llm_should_stop: LLM's stop decision
            llm_reasoning: LLM's reasoning
            population_file: Path to saved population file

        Returns:
            Dict with validated 'analysis_report', 'should_stop', and 'reasoning'
        """
        # Display LLM's complete analysis
        self._display_analysis_for_feedback(
            llm_analysis_report,
            llm_should_stop,
            llm_reasoning,
            population_file
        )

        # ===== STAGE 1: Get feedback on analysis report =====
        final_analysis_report = None
        while final_analysis_report is None:
            print("\n" + "="*80)
            print("STAGE 1: ANALYSIS REPORT REVIEW")
            print("="*80)

            print("\n" + "-"*80)
            print("LLM's Analysis Report:")
            print("-"*80)
            print(llm_analysis_report)
            print("-"*80)

            accept_report = await confirm_input("\nAccept this analysis report?")

            if accept_report:
                print("\n✓ Using LLM's analysis report")
                logger.info("Human accepted LLM analysis report")
                final_analysis_report = llm_analysis_report
            else:
                # Get revised analysis report
                print("\n" + "-"*80)
                print("PROVIDE REVISED ANALYSIS REPORT")
                print("-"*80)
                print("\nSuggested sections (you have full freedom to structure as you see fit):")
                print("  1. Overview: Summarize current iteration and progress")
                print("  2. Performance Analysis: How population performs on objectives")
                print("  3. Issues and Concerns: Problems like stagnation or poor diversity")
                print("  4. Strategic Recommendations: Actionable suggestions for next iteration")

                analysis_report = await get_multiline_input("\nEnter your revised analysis report:", end_marker="<END>")

                if not analysis_report.strip():
                    print("\n❌ Analysis report cannot be empty. Please try again.")
                    continue

                # Show and confirm
                print("\n" + "-"*80)
                print("Your Revised Analysis Report:")
                print("-"*80)
                print(analysis_report.strip())
                print("-"*80)

                if await confirm_input("\nConfirm this analysis report?"):
                    final_analysis_report = analysis_report.strip()
                    logger.info("Human provided revised analysis report")
                else:
                    print("\n✗ Not confirmed. Please try again.")

        # ===== STAGE 2: Get feedback on termination decision =====
        final_should_stop = None
        final_reasoning = None
        while final_should_stop is None or final_reasoning is None:
            print("\n" + "="*80)
            print("STAGE 2: TERMINATION DECISION REVIEW")
            print("="*80)

            print("\n" + "-"*80)
            print("LLM's Termination Decision:")
            print("-"*80)
            print(f"Should Stop: {llm_should_stop}")
            print(f"Reasoning: {llm_reasoning}")
            print("-"*80)

            accept_decision = await confirm_input("\nAccept this termination decision?")

            if accept_decision:
                print("\n✓ Using LLM's termination decision")
                logger.info("Human accepted LLM termination decision")
                final_should_stop = llm_should_stop
                final_reasoning = llm_reasoning
            else:
                # Get revised termination decision
                print("\n" + "-"*80)
                print("PROVIDE REVISED TERMINATION DECISION")
                print("-"*80)

                should_stop = await confirm_input("\nShould the optimization stop after this iteration?")

                reasoning = await get_multiline_input("\nProvide your reasoning for this decision:", end_marker="<END>")

                if not reasoning.strip():
                    print("\n❌ Reasoning cannot be empty. Please try again.")
                    continue

                # Show and confirm
                print("\n" + "-"*80)
                print("Your Revised Termination Decision:")
                print("-"*80)
                print(f"Should Stop: {should_stop}")
                print(f"Reasoning: {reasoning.strip()}")
                print("-"*80)

                if await confirm_input("\nConfirm this termination decision?"):
                    final_should_stop = should_stop
                    final_reasoning = reasoning.strip()
                    logger.info("Human provided revised termination decision")
                else:
                    print("\n✗ Not confirmed. Please try again.")

        # ===== Return complete analysis =====
        complete_analysis = {
            "analysis_report": final_analysis_report,
            "should_stop": final_should_stop,
            "reasoning": final_reasoning
        }

        print("\n" + "="*80)
        print("FINAL ANALYSIS (AFTER YOUR REVIEW)")
        print("="*80)
        print("\nAnalysis Report:")
        print("-"*80)
        print(complete_analysis['analysis_report'])
        print("\n" + "-"*80)
        print(f"Should Stop: {complete_analysis['should_stop']}")
        print(f"Reasoning: {complete_analysis['reasoning']}")
        print("="*80)

        logger.info("Human feedback on analysis completed successfully")
        return complete_analysis

    def _validate_decision_dict(self, decision_dict: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate the structure of a decision dictionary (without analysis_report).

        Args:
            decision_dict: The decision dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ["should_stop", "reasoning"]
        for field in required_fields:
            if field not in decision_dict:
                return False, f"Missing required field: '{field}'"

        # Validate types
        if not isinstance(decision_dict["should_stop"], bool):
            return False, "'should_stop' must be a boolean (true or false)"

        if not isinstance(decision_dict["reasoning"], str):
            return False, "'reasoning' must be a string"

        if decision_dict["reasoning"].strip() == "":
            return False, "'reasoning' cannot be empty"

        return True, None

    def _validate_analysis_dict(self, analysis_dict: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate the structure of a complete analysis dictionary (with analysis_report).

        Args:
            analysis_dict: The analysis dictionary to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        required_fields = ["analysis_report", "should_stop", "reasoning"]
        for field in required_fields:
            if field not in analysis_dict:
                return False, f"Missing required field: '{field}'"

        # Validate types
        if not isinstance(analysis_dict["analysis_report"], str):
            return False, "'analysis_report' must be a string"

        if analysis_dict["analysis_report"].strip() == "":
            return False, "'analysis_report' cannot be empty"

        if not isinstance(analysis_dict["should_stop"], bool):
            return False, "'should_stop' must be a boolean (true or false)"

        if not isinstance(analysis_dict["reasoning"], str):
            return False, "'reasoning' must be a string"

        if analysis_dict["reasoning"].strip() == "":
            return False, "'reasoning' cannot be empty"

        return True, None
