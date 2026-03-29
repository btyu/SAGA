"""
General planner module for the SciLeo Agent framework.

This module provides a general-purpose planner that uses LLM to analyze
the current state and plan objectives for the next iteration.
"""

import json
import re
from typing import List, Optional, Dict, Any, Tuple

from scileo_agent.core.modules import PlannerModule
from scileo_agent.core.data_models.objective import Objective
from scileo_agent.core.registry.module_registry import register_module
from scileo_agent.core.config import LLMConfig
from scileo_agent.utils.logging import get_logger
from scileo_agent.utils.human_feedback import (
    get_human_feedback_on_objectives,
    display_objectives_for_feedback
)


logger = get_logger()


def clean_objective_name(name: str) -> str:
    """
    Clean objective name to ensure it only contains valid characters.
    Valid characters: English letters, numbers, and underscores (_).

    Args:
        name: The original objective name

    Returns:
        Cleaned objective name with only valid characters
    """
    # Replace invalid characters with underscores
    cleaned = re.sub(r'[^a-zA-Z0-9_]', '_', name)
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    # Collapse multiple consecutive underscores into one
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned


SYSTEM_PROMPT_WITH_PLANNING_TEMPLATE = """You are an expert optimization planner for an iterative optimization framework. Your role is to help design and execute a multi-iteration optimization process.

**Your responsibilities include:**
1. **Strategic Planning (when requested):** Analyze the optimization goal deeply and create a comprehensive high-level strategy, including:
   - Identifying potential computational objectives that could contribute to the goal
   - Designing a multi-iteration optimization strategy
   - Considering which objectives are most critical in early vs. later iterations
   - Explaining how to balance trade-offs between different objectives

2. **Objective Proposal (in JSON format):** Propose specific objectives for each iteration based on:
   - The optimization goal and what specific aspects need optimization
   - Previous iteration results and whether existing objectives were effective
   - Whether new or modified objectives could better address remaining challenges
   - The balance between different aspects of the optimization problem
   - Focus on a few critical objectives per iteration. Proposing too many objectives at once can overwhelm the optimizer and reduce effectiveness. Given the current optimization status, prioritize only the most important and needed objectives rather than listing all potential ones.

**Critical Requirements for Objectives:**
- Each objective must be **specific and measurable** (not general, high-level, or vague)
- Each objective must be **computationally implementable** via Python code, libraries, or existing tools and models
{objective_type_info}- Objectives should directly contribute to achieving the optimization goal

**Response Format:**
- For strategic planning: Respond in natural language with thorough analysis and reasoning
- For objective proposals: Respond with reasoning followed by a valid JSON string wrapped in <answer>...</answer> tags:

<answer>
{{
    "objectives": [
        {{
            "name": "objective_name",  # A clear, descriptive name for the objective. Must only contain English letters, numbers, dots (.), underscores (_), and dashes (-)
{type_field}            "description": "A detailed description including what this objective is, its value range, what it measures, how it is computed, and what a good or bad score means for the optimization goal. If there are any implementation instructions provided in the analysis report (such as the method of calculating the objective score or available resources for implementing the scorer), include that information here so it can be passed to the scorer creator agent.",
{optimization_direction_field}            {weight_field}"reasoning": "Why this objective is important for achieving progress in this iteration"
        }}
    ],
    "reasoning": "Overall justification for the complete set of selected objectives and how they work together"
}}
</answer>
"""


SYSTEM_PROMPT_TEMPLATE = """You are an expert optimization planner for an iterative optimization framework. Your role is to propose objectives for each iteration based on the optimization goal and any available context (optimization status analysis, initial objectives, etc.).

Consider the following factors:
1. The optimization goal and what specific aspects need optimization
2. Previous iteration results and whether existing objectives were effective
3. Whether new or modified objectives could better address remaining challenges
4. The balance between different aspects of the optimization problem
5. Focus on a few critical objectives per iteration. Proposing too many objectives at once can overwhelm the optimizer and reduce effectiveness. Given the current optimization status, prioritize only the most important and needed objectives rather than listing all potential ones.

**Critical Requirements for Objectives:**
- Each objective must be **specific and measurable** (not general, high-level, or vague)
- Each objective must be **computationally implementable** via Python code, libraries, or existing tools and models
{objective_type_info}- Objectives should directly contribute to achieving the optimization goal

You should reason step by step. Your final answer must contain a valid JSON string, wrapped in <answer>...</answer> tags:
<answer>
{{
    "objectives": [
        {{
            "name": "objective_name",  # A clear, descriptive name for the objective. Must only contain English letters, numbers, dots (.), underscores (_), and dashes (-)
{type_field}            "description": "A detailed description including what this objective is, its value range, what it measures, how it is computed, and what a good or bad score means for the optimization goal. If there are any implementation instructions provided in the analysis report (such as the method of calculating the objective score or available resources for implementing the scorer), include that information here so it can be passed to the scorer creator agent.",
{optimization_direction_field}            {weight_field}"reasoning": "Why this objective is important for achieving progress in this iteration"
        }}
    ],
    "reasoning": "Overall justification for the complete set of selected objectives and how they work together"
}}
</answer>
"""


HIGH_LEVEL_PLANNING_USER_PROMPT_TEMPLATE = """**Optimization Goal:** {high_level_goal}
{context_information_section}

**Task:** Before we begin the iterative optimization process, please create a comprehensive high-level optimization strategy. Analyze the goal deeply and provide:

1. **Potential Computational Objectives:** Identify and describe all types of objectives that could help achieve the optimization goal. Think broadly - consider objectives for quality, diversity, feasibility, novelty, constraints, etc. For each potential objective type, explain what it would measure and why it matters.

2. **Multi-Iteration Strategy:** Design a strategy that explains:
   - How to effectively use these objectives across multiple iterations
   - Which objectives are most critical in early iterations vs. later iterations
   - How to balance trade-offs between competing objectives
   - What milestones or indicators would show progress toward the goal
   - How objectives might need to evolve as optimization progresses

Please provide your analysis and strategy in natural language. Be thorough and consider the full optimization journey."""


FIRST_OBJECTIVE_USER_PROMPT_TEMPLATE = """**Optimization Goal:** {high_level_goal}
{context_information_section}

**Current Status:** The beginning of iteration {iteration_number}.

{initial_objective_section}

**Task:** {initial_objective_instruction}Your response must contain a valid JSON string wrapped in <answer>...</answer> tags, following the format specified in the system prompt."""


FIRST_OBJECTIVE_WITH_PLAN_USER_PROMPT_TEMPLATE = """Based on the high-level strategy you created above, now propose the specific objectives for iteration {iteration_number}.

{initial_objective_section}

**Task:** {initial_objective_instruction}According to your optimization strategy, select the objectives that are most appropriate for this first iteration. Your response must contain a valid JSON string wrapped in <answer>...</answer> tags, following the format specified in the system prompt."""


OBJECTIVE_USER_PROMPT_TEMPLATE = """**Optimization Goal:** {high_level_goal}
{context_information_section}

**Current Status:** The beginning of iteration {iteration_number}.

Here is the analysis report at the end of the last iteration:
{analysis_report}


**Task:** Based on the analysis from the previous iterations, propose a revised set of objectives {number_limit}for this new iteration. You may retain, modify, or remove previous objectives and introduce new ones based on what would be most effective for continued progress. You must ensure that all objectives are specific and measurable, computationally implementable, and directly contribute to achieving the optimization goal.

**Important:** If the analysis report contains any implementation instructions for objectives (such as specific methods for calculating scores, available resources like pre-trained models or tools, or implementation details), include those instructions in the objective descriptions. This implementation knowledge will be passed to the scorer creator agent to help implement the scorer accurately.

**Remember:** This is not about listing all potential objectives. Select only what is essential given the current status. Your response must contain a valid JSON string wrapped in <answer>...</answer> tags, following the format specified in the system prompt."""


OBJECTIVE_RETRY_USER_PROMPT_TEMPLATE = """Among the {number_of_proposed_objectives} objectives you proposed, {number_of_unmatched_objectives} cannot be implemented. 

**Unimplementable Objectives:**
{unmatched_objectives_info}

**Optimization Goal:** {high_level_goal}

**Task:** Please propose a revised set of objectives {number_limit}again for this iteration. You should remove or replace the unimplementable objectives listed above. Your response must contain a valid JSON string wrapped in <answer>...</answer> tags, following the format specified in the system prompt."""


def list_objectives(objectives: List[Objective], requires_objective_weights: bool, include_essential_fields_only: bool = False, show_type: bool = True) -> str:
    """
    List objectives in a human-readable format.

    Args:
        objectives: List of objectives to format
        requires_objective_weights: Whether to include weight information
        include_essential_fields_only: If True, only show name, type, and description
        show_type: Whether to show the type field (default True)

    Returns:
        Formatted string listing all objectives
    """
    objective_info = ""
    for objective in objectives:
        objective_info += f"- {objective.name}:\n"
        if show_type:
            objective_info += f"  - Type: {objective.type}\n"
        objective_info += f"  - Description: {objective.description}\n"
        if not include_essential_fields_only:
            # Filter objectives don't have optimization_direction
            if objective.type != "filter" and objective.optimization_direction:
                objective_info += f"  - Optimization direction: {objective.optimization_direction}\n"
            if requires_objective_weights:
                objective_info += f"  - Weight: {objective.weight}\n"
    return objective_info


@register_module("general_planner", "0.6.1")
class GeneralPlanner(PlannerModule):
    """
    General planner module that uses LLM to plan objectives.
    
    This module analyzes the optimization goal and optional analysis reports
    to decide which objectives to use for the next iteration.
    """
    
    def __init__(self, module_id: str, config: Optional[Dict[str, Any]] = None, llm_config: Optional[LLMConfig]=None):
        """
        Initialize the general planner module.

        Args:
            module_id: Unique identifier for this module
            config: Configuration parameters
            llm_config: LLM configuration
        """
        super().__init__(module_id, config, llm_config)

        # Load configurations
        self.max_llm_retries = config.get("max_llm_retries", 3)
        self.requires_objective_weights = config.get("requires_objective_weights", False)
        self.do_high_level_planning = config.get("do_high_level_planning", True)
        self.support_filter = config.get("support_filter", False)
        self.support_population_wise = config.get("support_population_wise", False)
        self.enable_human_feedback = config.get("enable_human_feedback", False)

        self.use_context_information = config.get("use_context_information", "first_iteration")
        if self.use_context_information not in ["first_iteration", "all_iterations"]:
            self.use_context_information = "disabled"

        self.max_objectives = config.get("max_objectives", None)

        # Build system prompt based on configuration
        weight_field = '"weight": 1.0,  # Used for multi-objective optimization or analysis\n            ' if self.requires_objective_weights else ""

        # Build objective type information section based on what types are supported
        objective_type_parts = []
        supported_types = ["candidate-wise"]  # Always supported

        if self.support_population_wise:
            supported_types.append("population-wise")
        if self.support_filter:
            supported_types.append("filter")

        # Build description based on supported types
        if len(supported_types) == 3:
            objective_type_parts.append("- All objectives take a population of candidates as input. Objectives can be of three types:\n")
            objective_type_parts.append('  1. **"candidate-wise"**: Evaluate single candidates independently, giving each candidate in the population a float score (e.g., molecular properties, docking scores)\n')
            objective_type_parts.append('  2. **"population-wise"**: Evaluate the entire population collectively, giving the whole population one single float score (e.g., diversity metrics across the entire population)\n')
            objective_type_parts.append('  3. **"filter"**: Filter objectives that give each candidate a binary judgment (True/False) to determine if it should be kept or removed. Used to filter out invalid or unwanted candidates rather than optimize their values (e.g., chemical validity checks, constraint satisfaction). Candidates that fail (False) are removed from the population.\n')
        elif "population-wise" in supported_types and "filter" not in supported_types:
            objective_type_parts.append("- All objectives take a population of candidates as input. Objectives can be of two types:\n")
            objective_type_parts.append('  1. **"candidate-wise"**: Evaluate single candidates independently, giving each candidate in the population a float score\n')
            objective_type_parts.append('  2. **"population-wise"**: Evaluate the entire population collectively, giving the whole population one single float score (e.g., diversity metrics)\n')
        elif "filter" in supported_types and "population-wise" not in supported_types:
            objective_type_parts.append("- All objectives take a population of candidates as input. Objectives can be of two types:\n")
            objective_type_parts.append('  1. **"candidate-wise"**: Regular objectives for optimization, giving each candidate in the population a float score\n')
            objective_type_parts.append('  2. **"filter"**: Filter objectives that give each candidate a binary judgment (True/False). Used to filter out invalid/unwanted candidates (they will be removed from population). Examples: chemical validity checks, constraint satisfaction.\n')
        else:
            objective_type_parts.append('- All objectives take a population of candidates as input and are of type "candidate-wise", evaluating individual candidates independently and giving each a float score\n')

        objective_type_info = "".join(objective_type_parts)

        # Build field specifications for JSON format - always use the "type" field
        if len(supported_types) > 1:
            # Multiple types supported - include type field with allowed values
            type_values = ' or '.join([f'"{t}"' for t in supported_types])
            type_field = f'            "type": {type_values},  # The type of objective: {", ".join([f"{t}" for t in supported_types])}\n'
        else:
            # Only candidate-wise supported - type field is optional or can be omitted
            type_field = ''

        # Build optimization_direction field - conditional for filter objectives
        if self.support_filter:
            optimization_direction_field = '            "optimization_direction": "maximize" or "minimize",  # Required for candidate-wise and population-wise; omit for filter objectives\n'
        else:
            optimization_direction_field = '            "optimization_direction": "maximize" or "minimize",  # Whether higher or lower scores are better\n'

        # Use different system prompt based on whether high-level planning is enabled
        if self.do_high_level_planning:
            system_prompt = SYSTEM_PROMPT_WITH_PLANNING_TEMPLATE.format(
                objective_type_info=objective_type_info,
                type_field=type_field,
                optimization_direction_field=optimization_direction_field,
                weight_field=weight_field
            )
        else:
            system_prompt = SYSTEM_PROMPT_TEMPLATE.format(
                objective_type_info=objective_type_info,
                type_field=type_field,
                optimization_direction_field=optimization_direction_field,
                weight_field=weight_field
            )

        self.message_history = [
            {
                "role": "system",
                "content": system_prompt,
            }
        ]
        self.objective_planning_response_dicts = {}
        self.high_level_plan = None  # Store high-level plan text if generated

        self.attached_scorers = dict()

    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the module with additional logging."""
        status = super().get_status()

        # Add GeneralPlanner specific information
        status.update({
            "objective_planning_responses_count": len(self.objective_planning_response_dicts),
            "message_history_length": len(self.message_history),
            "has_high_level_plan": self.high_level_plan is not None,
            "support_filter": self.support_filter,
            "support_population_wise": self.support_population_wise,
            "enable_human_feedback": self.enable_human_feedback
        })
        return status

    def create_high_level_plan(
        self,
        high_level_goal: str,
        context_information: Optional[str] = None
    ) -> str:
        """
        Create a high-level optimization plan before the first iteration.

        Args:
            high_level_goal: The overall optimization goal
            context_information: Optional context information for the task

        Returns:
            String containing the high-level planning response
        """
        logger.info("Creating high-level optimization plan", {
            "has_context_information": context_information is not None and context_information.strip() != ""
        })

        # Create high-level planning user prompt
        prompt = HIGH_LEVEL_PLANNING_USER_PROMPT_TEMPLATE.format(
            high_level_goal=high_level_goal,
            context_information_section="" if (context_information is None or context_information.strip() == "") else f"\n**Context Information:**\n{context_information}\n"
        )
        self.message_history.append({"role": "user", "content": prompt})

        # Get high-level plan from LLM (natural language response)
        response = self.call_llm(self.message_history)
        response_text = response['content']

        # Add assistant response to message history
        self.message_history.append({"role": "assistant", "content": response_text})

        # Store the high-level plan
        self.high_level_plan = response_text

        logger.info("High-level optimization plan created", {
            "plan_length": len(response_text)
        })

        return response_text
    
    async def plan_objectives(
        self,
        iteration_number: int,
        high_level_goal: str,
        context_information: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        analysis_report: Optional[str] = None
    ) -> Tuple[List[Objective], Dict[str, Any]]:
        """
        Plan objectives for the current iteration.
        
        Args:
            iteration_number: The current iteration number
            high_level_goal: The overall optimization goal
            context_information: Optional context information for the task
            initial_objectives: The initial list of objectives
            analysis_report: Optional analysis report from previous iteration
            
        Returns:
            Tuple of (List of objectives to use for this iteration, Dict of response)
        """

        if context_information is not None:
            context_information = context_information.strip()

        attached_scorers = dict()

        if initial_objectives:

            if len(initial_objectives) != len(set([objective.name for objective in initial_objectives])):
                raise ValueError("Initial objective names must be unique.")

            for objective in initial_objectives:
                if not objective.has_scorer():
                    continue

                attached_scorers[objective.name] = objective.scorer

        if self.max_objectives is not None:
            number_limit = f"(no more than {self.max_objectives}) "
        else:
            number_limit = ""

        if iteration_number == 1:
            initial_objective_section = ""
            if initial_objectives is None or len(initial_objectives) == 0:
                initial_objective_section = ""
                initial_objective_instruction = f"Please propose a set of objectives {number_limit}for this first iteration. "
            else:
                # Validate that initial objectives don't exceed the limit
                if self.max_objectives is not None and len(initial_objectives) > self.max_objectives:
                    logger.critical(
                        "Initial objectives exceed the maximum allowed objectives",
                        {
                            "initial_objectives_count": len(initial_objectives),
                            "max_objectives": self.max_objectives
                        }
                    )
                    raise ValueError(
                        f"Initial objectives count ({len(initial_objectives)}) exceeds the maximum "
                        f"allowed objectives ({self.max_objectives}). Please reduce the number of "
                        f"initial objectives or increase max_objectives."
                    )

                initial_objective_section = "Initial objective(s) provided by the user (these are the most important and must be included exactly as-is):\n"
                # Only show type field if multiple types are supported
                show_type = self.support_population_wise or self.support_filter
                objective_info = list_objectives(initial_objectives, self.requires_objective_weights, show_type=show_type)
                initial_objective_section += objective_info

                # Calculate how many additional objectives can be proposed
                num_initial = len(initial_objectives)
                if self.max_objectives is not None:
                    num_additional_allowed = self.max_objectives - num_initial
                    if num_additional_allowed > 0:
                        initial_objective_instruction = f"You must include ALL {num_initial} initial objectives listed above (unchanged) in your proposed objective set. You may also propose up to {num_additional_allowed} additional objectives if they would be beneficial for this iteration (total no more than {self.max_objectives}). "
                    else:
                        initial_objective_instruction = f"You must include ALL {num_initial} initial objectives listed above (unchanged) as the complete objective set for this iteration. Do not propose any additional objectives. "
                else:
                    initial_objective_instruction = f"You must include ALL {num_initial} initial objectives listed above (unchanged) in your proposed objective set. You may also propose additional objectives if they would be beneficial for this iteration. "

            # Use different prompt template depending on whether high-level plan exists
            if self.high_level_plan is not None:
                # High-level plan already created - reference it
                prompt = FIRST_OBJECTIVE_WITH_PLAN_USER_PROMPT_TEMPLATE.format(
                    iteration_number=iteration_number,
                    initial_objective_section=initial_objective_section,
                    initial_objective_instruction=initial_objective_instruction
                )
            else:
                # No high-level plan - use standard first iteration prompt
                prompt = FIRST_OBJECTIVE_USER_PROMPT_TEMPLATE.format(
                    iteration_number=iteration_number,
                    high_level_goal=high_level_goal,
                    initial_objective_section=initial_objective_section,
                    number_limit=number_limit,
                    initial_objective_instruction=initial_objective_instruction,
                    context_information_section="" if (context_information is None or self.use_context_information == "disabled") else f"\n**Context Information:**\n{context_information}\n"
                )
            self.message_history.append({"role": "user", "content": prompt})
        else:
            if analysis_report is None:
                logger.critical("Analysis report is required for non-first iterations", {
                    "iteration_number": iteration_number
                })
                raise ValueError("Analysis report is required for non-first iterations.")
            prompt = OBJECTIVE_USER_PROMPT_TEMPLATE.format(
                iteration_number=iteration_number,
                high_level_goal=high_level_goal,
                analysis_report=analysis_report,
                number_limit=number_limit,
                context_information_section="" if (context_information is None or self.use_context_information != "all_iterations") else f"\n**Context Information:**\n{context_information}\n"
            )
            self.message_history.append({"role": "user", "content": prompt})

        objectives, response_dict, response_text = await self._get_objectives_and_response_dict()
        for objective in objectives:
            if objective.name in attached_scorers:
                objective.scorer = attached_scorers[objective.name]
        
        self.message_history.append({
            "role": "assistant",
            "content": response_text
        })

        # logger.info("Planning objectives completed", {
        #     "iteration_number": iteration_number,
        #     "objectives": "Objectives:\n" + list_objectives(objectives, self.requires_objective_weights)
        # })

        return objectives, response_dict
    
    async def _process(
        self,
        iteration_number: int,
        high_level_goal: str,
        context_information: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        analysis_report: Optional[str] = None,
        mode: Optional[str] = "normal",
        additional_information: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Do the whole planning process."""
        
        if not isinstance(iteration_number, int) or iteration_number <= 0:
            logger.critical("Iteration number must be a positive integer.", {
                "iteration_number": iteration_number
            })
            raise ValueError("Iteration number must be a positive integer.")
        
        if mode == "normal":
            # Propose objectives for each iteration

            if iteration_number == 1:
                # If high-level planning is enabled, create the plan first
                if self.do_high_level_planning:
                    high_level_plan = self.create_high_level_plan(
                        high_level_goal=high_level_goal,
                        context_information=context_information if self.use_context_information != "disabled" else None
                    )
                else:
                    high_level_plan = None

                objectives, objective_planning_response_dict = await self.plan_objectives(
                    iteration_number,
                    high_level_goal,
                    context_information=context_information if self.use_context_information != "disabled" else None,
                    initial_objectives=initial_objectives,
                    analysis_report=None
                )
                self.objective_planning_response_dicts[iteration_number] = objective_planning_response_dict

                result = {
                    "objectives": objectives,
                    "objective_planning_response_dict": objective_planning_response_dict
                }

                # Include high-level plan in result if it was generated
                if high_level_plan is not None:
                    result["high_level_plan"] = high_level_plan
            
            else:
                objectives, objective_planning_response_dict = await self.plan_objectives(
                    iteration_number,
                    high_level_goal,
                    context_information=context_information if self.use_context_information == "all_iterations" else None,
                    initial_objectives=None,
                    analysis_report=analysis_report
                )
                self.objective_planning_response_dicts[iteration_number] = objective_planning_response_dict
                
                result = {
                    "objectives": objectives,
                    "objective_planning_response_dict": objective_planning_response_dict
                }

        elif mode == "retry":
            # Plan objectives again if some objectives are unimplementable

            objectives, objective_planning_response_dict = await self.plan_objectives_again(
                iteration_number=iteration_number,
                high_level_goal=high_level_goal,
                context_information=context_information,
                initial_objectives=initial_objectives,
                analysis_report=analysis_report,
                additional_information=additional_information
            )

            result = {
                "objectives": objectives,
                "objective_planning_response_dict": objective_planning_response_dict
            }

        else:
            logger.critical("Invalid mode.", {
                "mode": mode
            })
            raise ValueError(f"Invalid mode: {mode}")
        
        return result
            
    def _parse_response_dict(self, response_text: str) -> Dict[str, Any]:
        """Parse LLM response to extract objectives."""
        try:
            # Extract JSON content (wrapped in <answer>...</answer> tags) from response text
            json_match = re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                response_dict = json.loads(json_str)
            else:
                logger.error("No JSON content found in response", {
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                })
                raise ValueError("No JSON content found in the response.")
        except json.JSONDecodeError as e:
            logger.error("JSON decode error", {
                "error": str(e),
                "json_preview": json_str[:200] + "..." if 'json_str' in locals() and len(json_str) > 200 else json_str if 'json_str' in locals() else "N/A"
            })
            raise ValueError("Invalid JSON response from LLM: could not parse JSON string.")
        
        if not isinstance(response_dict, dict):
            logger.error("Response is not a dictionary", {
                "response_type": type(response_dict).__name__
            })
            raise ValueError("Invalid response from LLM: response_dict is not a dictionary.")
        
        return response_dict
    
    async def plan_objectives_again(
        self,
        iteration_number: int,
        high_level_goal: str,
        context_information: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        analysis_report: Optional[str] = None,
        additional_information: Optional[dict[str, Any]] = None
    ) -> Tuple[List[Objective], Dict[str, Any]]:
        """Plan objectives again."""

        if additional_information is None:
            logger.critical("Additional information is required for retry mode.", {
                "iteration_number": iteration_number
            })
            raise ValueError("Additional information is required for retry mode.")

        matched_objectives = additional_information["matched_objectives"]
        num_matched_objectives = len(matched_objectives)
        unmatched_objectives = additional_information["unmatched_objectives"]
        # available_objectives = additional_information["available_objectives"]

        num_proposed_objectives = len(matched_objectives) + len(unmatched_objectives)
        num_unmatched_objectives = len(unmatched_objectives)

        logger.info("Planning objectives again", {
            "iteration_number": iteration_number,
            "proposed_objectives_count": num_proposed_objectives,
            "unmatched_objectives_count": num_unmatched_objectives,
        })

        # Only show type field if multiple types are supported
        show_type = self.support_population_wise or self.support_filter
        unmatched_objectives_info = list_objectives(unmatched_objectives, self.requires_objective_weights, show_type=show_type)
        # available_objectives_info = list_objectives(available_objectives, self.requires_objective_weights, include_essential_fields_only=True, show_type=show_type)

        prompt = OBJECTIVE_RETRY_USER_PROMPT_TEMPLATE.format(
            high_level_goal=high_level_goal,
            number_of_proposed_objectives=num_proposed_objectives,
            number_of_unmatched_objectives=num_unmatched_objectives,
            unmatched_objectives_info=unmatched_objectives_info,
            # available_objectives_info=available_objectives_info,
            number_limit=f"(no more than {self.max_objectives}) " if self.max_objectives is not None else ""
        )

        self.message_history.append({"role": "user", "content": prompt})

        objectives, response_dict, response_text = await self._get_objectives_and_response_dict()

        self.message_history.append({
            "role": "assistant",
            "content": response_text
        })

        # For logging, always show type for clarity
        logger.info("Planning objectives again completed", {
            "iteration_number": iteration_number,
            "objectives": "Objectives:\n" + list_objectives(objectives, self.requires_objective_weights, show_type=True)
        })

        return objectives, response_dict
    
    async def _get_objectives_and_response_dict(self) -> Tuple[List[Objective], Dict[str, Any], str]:
        """Get objectives and response dict from LLM response."""
        # Step 1: Get valid LLM response that can be processed into objectives
        llm_response_dict = self._get_llm_response_dict()

        # Step 2: Get human feedback if enabled (this happens AFTER we have a valid LLM response)
        if self.enable_human_feedback:
            logger.info("Human feedback is enabled - requesting feedback on proposed objectives")

            # Get human feedback - this will keep retrying until valid input that can be processed is provided
            # No fallback to LLM version - we must get valid human input
            response_dict = await self._get_human_revised_response_dict(llm_response_dict)

            logger.info("Human feedback received and validated successfully")
        else:
            # No human feedback - use LLM response directly
            response_dict = llm_response_dict

        # Step 3: Process the response dict (either from LLM or human) into Objective objects
        # At this point, we know the response_dict can be successfully processed
        # The processing may update the dict (e.g., cleaning names), so use the returned updated dict
        objectives, updated_response_dict, response_text = self._process_response_dict_to_objectives(response_dict)

        # Step 4: Add original LLM response to updated_response_dict if human feedback was provided
        if self.enable_human_feedback:
            updated_response_dict["original_llm_response"] = llm_response_dict

        return objectives, updated_response_dict, response_text

    def _get_llm_response_dict(self) -> Dict[str, Any]:
        """
        Get a valid response dict from the LLM with retries.

        This method ensures the response can be successfully processed into objectives.
        It validates both the JSON structure AND that objectives can be created from it.
        """
        count = 0

        while True:
            response = self.call_llm(self.message_history)
            response_text = response['content']
            # logger.debug("LLM response:\n" + response_text.strip())

            try:
                response_dict = self._parse_response_dict(response_text)

                # Validate basic structure
                if not isinstance(response_dict["objectives"], list):
                    raise ValueError(f"Invalid response from LLM: objectives is not a list but a {type(response_dict['objectives'])}.")
                if len(response_dict["objectives"]) == 0:
                    raise ValueError("Invalid response from LLM: objectives list is empty.")
                if not isinstance(response_dict["reasoning"], str):
                    raise ValueError(f"Invalid response from LLM: reasoning is not a string but a {type(response_dict['reasoning'])}.")

                # Validate that we can actually process this into objectives
                # This catches issues like invalid types, missing directions, etc.
                try:
                    _, _, _ = self._process_response_dict_to_objectives(response_dict)
                except Exception as processing_error:
                    raise ValueError(f"LLM response cannot be processed into objectives: {str(processing_error)}")

                # Successfully got valid response that can be processed
                return response_dict

            except ValueError as e:
                logger.error(f"LLM response validation failed (attempt {count + 1})", {
                    "error": str(e),
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                })
                if count >= self.max_llm_retries:
                    logger.critical("Failed to get valid LLM response after maximum retries", {
                        "max_retries": self.max_llm_retries,
                        "final_error": str(e)
                    })
                    raise e
                count += 1
                continue
            except KeyError as e:
                logger.error(f"Missing key in LLM response (attempt {count + 1})", {
                    "missing_key": str(e),
                    "response_preview": response_text[:200] + "..." if len(response_text) > 200 else response_text
                })
                if count >= self.max_llm_retries:
                    logger.critical("Failed to get valid LLM response after maximum retries", {
                        "max_retries": self.max_llm_retries,
                        "final_error": f"Missing key: {e}"
                    })
                    raise ValueError(f"Invalid response from LLM: {e}")
                count += 1
                continue

    async def _get_human_revised_response_dict(self, llm_response_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get human-revised response dict with validation that it can be processed.

        This method keeps asking for human input until valid input that can be
        successfully processed into objectives is provided.

        Args:
            llm_response_dict: The LLM's proposed objectives to show to the human

        Returns:
            A validated response dict that can be successfully processed into objectives
        """
        # Keep trying until we get valid input that can be processed
        while True:
            # Get human feedback on the objectives (with unlimited internal retries for JSON validation)
            revised_response_dict = await get_human_feedback_on_objectives(
                llm_proposed_objectives=llm_response_dict,
                end_marker="<END>",
                max_attempts=100,  # Allow many attempts for JSON validation
                support_population_wise=self.support_population_wise,
                support_filter=self.support_filter,
                requires_weights=self.requires_objective_weights
            )

            # Validate that the revised response can be processed into objectives
            # Also get the updated dict (with cleaned names, etc.)
            try:
                _, updated_revised_dict, _ = self._process_response_dict_to_objectives(revised_response_dict)
                # Success! Return the validated and updated response
                return updated_revised_dict

            except Exception as processing_error:
                error_msg = f"Your revised objectives cannot be processed: {str(processing_error)}"
                logger.warning(f"Human-revised response processing failed: {error_msg}")

                print("\n" + "="*80)
                print("❌ PROCESSING ERROR")
                print("="*80)
                print(f"\n{error_msg}\n")
                print("This usually means:")
                print("  - Invalid objective type (must be 'candidate-wise', 'population-wise', or 'filter')")
                print("  - Missing 'optimization_direction' for non-filter objectives")
                print("  - Invalid 'optimization_direction' (must be 'maximize' or 'minimize')")
                print("  - Filter objective has 'optimization_direction' (filters shouldn't have this)")
                print("  - Missing or invalid 'weight' field (if weights are required)")
                print("="*80)

                print("\nPlease try again.")

                # Show the LLM proposal again for reference
                print("\n" + "="*80)
                print("REMINDER: LLM's Original Proposal")
                print("="*80)
                display_objectives_for_feedback(llm_response_dict)

                # Loop back to get new input
                continue

    def _process_response_dict_to_objectives(self, response_dict: Dict[str, Any]) -> Tuple[List[Objective], Dict[str, Any], str]:
        """
        Process a response dict (from LLM or human) into Objective objects.

        This method may modify the response_dict (e.g., cleaning objective names),
        so it returns the updated version.

        Args:
            response_dict: Dictionary containing objectives and reasoning

        Returns:
            Tuple of (list of Objective objects, updated response dict, response text with <answer> tags)
        """
        objectives = []
        any_name_cleaned = False

        for objective in response_dict["objectives"]:
            # Make a copy to avoid modifying the original
            obj_copy = objective.copy()

            reasoning = obj_copy.pop("reasoning", "No reasoning provided")

            if self.requires_objective_weights:
                weight = obj_copy.pop("weight", None)
                if weight is not None and not isinstance(weight, (float, int)):
                    raise ValueError(f"Invalid weight for objective '{obj_copy['name']}': weight is not a number but a {type(weight)}.")
            else:
                weight = None

            # Clean the objective name to ensure it only contains valid characters
            original_name = obj_copy["name"]
            cleaned_name = clean_objective_name(original_name)
            if cleaned_name != original_name:
                logger.warning("Objective name was cleaned", {
                    "original_name": original_name,
                    "cleaned_name": cleaned_name
                })
                obj_copy["name"] = cleaned_name
                any_name_cleaned = True

            # Get objective type from response
            # If type field is present, use it directly
            # If not present (only for backward compatibility), default to "candidate-wise"
            obj_type = obj_copy.get("type", "candidate-wise")

            # Validate the type value
            valid_types = ["candidate-wise"]
            if self.support_population_wise:
                valid_types.append("population-wise")
            if self.support_filter:
                valid_types.append("filter")

            if obj_type not in valid_types:
                raise ValueError(
                    f"Invalid objective type '{obj_type}' for objective '{cleaned_name}'. "
                    f"Valid types based on configuration: {', '.join(valid_types)}"
                )

            # For filter objectives, optimization_direction should not be provided
            # For other types, it's required
            optimization_direction = obj_copy.get("optimization_direction")
            if obj_type == "filter":
                if optimization_direction is not None:
                    raise ValueError(
                        f"Filter objective '{cleaned_name}' should not have optimization_direction. "
                        "Filter objectives only return pass/fail (True/False)."
                    )
            else:
                if optimization_direction is None:
                    raise ValueError(
                        f"Objective '{cleaned_name}' (type: {obj_type}) must have optimization_direction."
                    )

            objectives.append(
                Objective(
                    name=cleaned_name,
                    description=obj_copy["description"],
                    type=obj_type,
                    optimization_direction=optimization_direction,
                    weight=weight,
                    metadata={"reasoning": reasoning},
                )
            )

        # Generate response text with <answer> tags
        # Reconstruct the JSON with cleaned names if any were cleaned
        updated_response_dict = {
            "objectives": [],
            "reasoning": response_dict["reasoning"]
        }
        for obj in objectives:
            obj_dict = {
                "name": obj.name,
                "description": obj.description,
            }
            # Only include type field if multiple types are supported
            if self.support_population_wise or self.support_filter:
                obj_dict["type"] = obj.type
            # Only include optimization_direction for non-filter objectives
            if obj.type != "filter":
                obj_dict["optimization_direction"] = obj.optimization_direction
            if self.requires_objective_weights:
                obj_dict["weight"] = obj.weight
            obj_dict["reasoning"] = obj.metadata["reasoning"]
            updated_response_dict["objectives"].append(obj_dict)

        # Generate JSON
        updated_json = json.dumps(updated_response_dict, indent=4)

        # Create response text with <answer> tags
        response_text = f"<answer>\n{updated_json}\n</answer>"

        if any_name_cleaned:
            logger.debug("Updated response text with cleaned objective names")

        return objectives, updated_response_dict, response_text
