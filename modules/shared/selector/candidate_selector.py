"""
Candidate Selector for selecting the best candidates across optimization iterations.

This module provides a specialized selector that uses ClaudeAgent to perform
intelligent selection of the best candidates from multiple iterations of optimization.

The selector supports domain-specific tools via ToolUniverse MCP server, similar to
the CandidateAnalyzer. Tool selection is performed automatically using an LLM to identify
relevant tools for the specific optimization domain.

Configuration Example:
    config = {
        # Basic configuration
        'candidate_selector_workspace': 'candidate_selection_workspace',
        'candidate_selector_model_name': 'claude_code/claude-sonnet-4-5-20250929',
        'candidate_selector_models_file': 'llm_configs/claude_code.yaml',
        'candidate_selector_credentials_file': 'llm_configs/credentials.yaml',
        'candidate_selector_run_in_docker': True,
        'candidate_selector_max_thinking_tokens': 4096,

        # Domain tools configuration
        'candidate_selector_enable_domain_tools': True,  # Enable ToolUniverse MCP server
        'candidate_selector_tooluniverse_path': '/opt/tooluniverse-env',  # Docker: /opt/tooluniverse-env, Native: ./tooluniverse-env

        # Tool selection LLM configuration (automatically initialized if domain tools enabled)
        'candidate_selector_tool_selection_model': 'anthropic/claude-sonnet-4-5-20250929',
        'candidate_selector_tool_selection_models_file': 'llm_configs/models.yaml',
        'candidate_selector_tool_selection_credentials_file': 'llm_configs/credentials.yaml',
    }

    selector = CandidateSelector(config)

Tool Selection Architecture:
    1. LLM client is automatically initialized from config when domain tools are enabled
    2. Load ToolUniverse catalog from tool_universe_tools.json
    3. Use LLM client to select relevant tools based on optimization goal and candidate type
    4. Configure MCP server with only the selected tools
    5. ClaudeAgent runs with access to the filtered tool set
"""

import os
import time
import uuid
import shutil
import json
import re
from typing import Dict, Any, List, Optional, Set
from pathlib import Path

from scileo_agent.utils.logging import get_logger
from scileo_agent.core.data_models import Population, Candidate, Objective
from scileo_agent.core.registry.serializer_registry import get_serializer
from scileo_agent.utils import LLMClient

# Import ClaudeAgent from the local module
from .claude_agent import ClaudeAgent


logger = get_logger()


SYSTEM_PROMPT = """You are an expert optimization analyst specializing in candidate selection across multiple optimization iterations. Your task is to examine candidates from all iterations and select the best ones based on the optimization goal, objective scores, and your analysis.

You have access to file reading, code execution, web search, and domain-specific tools to thoroughly analyze the candidates. Use these tools to:
1. Understand the candidate structure, properties, and objective scores
2. Identify patterns and trends across iterations
3. Compare candidates across different iterations
4. Apply domain knowledge to assess candidate quality beyond just scores
5. Select the best candidates that best satisfy the optimization goal

Your selection MUST be grounded in verifiable evidence from one or more of the following sources:
- Direct observations and measurements from examining actual candidate data
- Computational results from executing code to analyze candidates
- Domain-specific tools (if any) applied to evaluate candidate properties
- Authoritative information retrieved from online sources via web search

CRITICAL: Base ALL selection decisions strictly on factual evidence. NEVER fabricate data, invent measurements, or hallucinate patterns that are not demonstrably present in the actual candidates or verifiable through the tools available to you.
"""


USER_PROMPT_TEMPLATE = """# Candidate Selection Task

## Optimization Context

**Optimization goal**: {high_level_goal}
{context_information_section}
### Optimization Process

The optimization system iteratively improves a population of candidates to achieve the optimization goal. In each iteration, the optimizer takes the population from the previous iteration (or a subset of it, depending on configuration) and produces an improved population through optimization algorithms. Each iteration may use a different set of objectives to guide the optimization, allowing the system to adapt its focus as optimization progresses.

### Objective Types

There are three possible types of objectives used in an optimization:

- **Candidate-wise objectives**: Score each individual candidate with a numerical value. These evaluate the quality of each candidate independently.
- **Population-wise objectives**: Score the entire population as a whole with a single numerical value. These evaluate collective properties like diversity or coverage.
- **Filter objectives**: Pass/fail each candidate based on criteria (True=pass, False=fail). These act as hard constraints to eliminate infeasible candidates.

### Objectives Used in This Optimization

{objectives_info}

## Population Information Across Iterations

You have candidates generated from {num_iterations} iteration(s) of optimization, with a total of {total_candidates} candidates across all iterations.

All candidates have been saved to the file `{candidates_file}`. Use this file to analyze and select the best candidates. It is a JSON file containing candidates from all iterations with their representations and objective scores.

### File Structure

```json
{{
  "optimization_goal": "{high_level_goal}",
  "num_iterations": {num_iterations},
  "total_candidates": {total_candidates},
  "iterations": [
    {{
      "iteration": 1,
      "population_size": 100,
      "population_scores": {{  // population-wise objective scores (if any)
        "objective_name": float_value,
        ...
      }},
      "candidates": [
        {{
          "id": "...",  // unique candidate ID
          "representation": "...",  // the candidate representation
          "scores": {{  // candidate-wise objective scores or filter results (if any)
            "objective_name": float_value or boolean_value,
            ...
          }}
        }},
        ...
      ]
    }},
    ...
  ]
}}
```

Each candidate representation ({candidate_type}) is {candidate_description}

## Your Task

Your primary goal is to **select the best candidates from across all iterations** that best satisfy the optimization goal. The number of candidates to select will be specified below.

### Selection Criteria

When selecting candidates, consider:

1. **Objective Scores**: How well do candidates perform on the defined objectives?
   - For candidate-wise objectives, look at individual scores
   - For population-wise objectives, consider the population context
   - Balance multiple objectives if applicable

2. **Iteration Progress**: How have candidates improved over iterations?
   - Later iterations may have better candidates
   - But earlier iterations might have unique valuable candidates

3. **Domain-Specific Quality**: Beyond scores, what makes a candidate genuinely good?
   - Use domain knowledge and available tools to assess quality
   - Consider practical feasibility and real-world applicability
   - Identify potential issues not captured by objectives

4. **Diversity**: Should you select diverse candidates or focus on the absolute best?
   - Consider whether diversity is valuable for the use case
   - Balance exploitation (best performers) vs exploration (diverse alternatives)

### Recommended Workflow

1. **Load and Explore the Data**:
   - Read the candidates file
   - Understand the distribution of scores across iterations
   - Identify trends and patterns

2. **Define Selection Strategy**:
   - Based on the optimization goal, decide what "best" means
   - Consider how to weight different objectives
   - Decide on diversity vs optimization trade-offs

3. **Quantitative Analysis**:
   - Use Python code to compute rankings, statistics, and filtering
   - Apply domain-specific tools to evaluate candidates
   - Identify top candidates based on your strategy

4. **Qualitative Verification**:
   - Examine the top candidates in detail
   - Verify they meet domain-specific quality criteria
   - Check for any issues or concerns

5. **Final Selection**:
   - Select exactly {num_to_select} candidates
   - Provide the candidate IDs in your output

### Available Python Environment

The environment includes extensive scientific computing, machine learning, and domain-specific packages:

**Core Scientific Computing**: numpy, scipy, pandas, matplotlib, plotly, seaborn, scikit-learn, jupyter

**Machine Learning & Deep Learning**: pytorch, transformers, datasets, pytorch-lightning, dgl, optuna, wandb, tensorboard

**Chemistry & Materials Science**: rdkit, pymatgen, ase, openmm, openbabel, xtb-python, alignn, unidock, descriptastorus, drug-likeness

**Bioinformatics**: biopython, grelu

**Other Tools**: ray, xgboost, umap-learn, pydantic, litellm, and more

If you need additional packages or tools, install them as needed with conda, mamba, pip, or apt.

### Output Format

After completing your analysis and selection, write your results to a file called `selection_results.json` in the working directory with the following structure:

```json
{{
  "selected_candidate_ids": [
    "candidate_id_1",
    "candidate_id_2",
    ...
  ],
  "selection_reasoning": "Brief explanation of why these candidates were selected and the strategy used.",
  "selection_metadata": {{
    "selection_strategy": "description of the strategy (e.g., 'top scorers on objective X', 'diverse set balancing objectives Y and Z')",
    "key_criteria": ["criterion 1", "criterion 2", ...],
    "iterations_represented": [1, 2, 3, ...]  // which iterations the selected candidates come from
  }}
}}
```

**IMPORTANT**:
- You MUST select exactly {num_to_select} candidates
- The `selected_candidate_ids` list must contain exactly {num_to_select} unique candidate IDs
- All candidate IDs must exist in the input candidates file
- Write the output to `selection_results.json` before completing your task
"""


TOOL_SELECTION_SYSTEM_PROMPT = "You are a helpful assistant that selects relevant scientific tools based on optimization goals."


TOOL_SELECTION_PROMPT_TEMPLATE = """You are a tool selection expert for scientific optimization tasks. You will be given:
1. An optimization goal
2. Context information about the task
3. Information about the candidates being analyzed
4. A complete list of available scientific tools from ToolUniverse

Your task is to select ALL tools that could be potentially useful for analyzing and selecting candidates in this optimization task. Be inclusive rather than restrictive - it's better to include a tool that might be useful than to exclude one that could help.

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
- Any other analysis that could provide insights into candidate selection

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


class CandidateSelector:
    """
    Selector for intelligent candidate selection across optimization iterations.

    This class manages workspace preparation, prompt formation, and execution
    of candidate selection using a ClaudeAgent with full tool access.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the candidate selector.

        Args:
            config: Configuration dictionary with optional keys:
                - candidate_selector_workspace: Base path for selection workspaces (default: 'candidate_selection_workspace')
                - candidate_selector_model_name: Model name for selector (default: 'claude_code/claude-sonnet-4-5-20250929')
                - candidate_selector_models_file: Path to models config (default: 'llm_configs/claude_code.yaml')
                - candidate_selector_credentials_file: Path to credentials (default: 'llm_configs/credentials.yaml')
                - candidate_selector_run_in_docker: Whether to run in Docker (default: True)
                - candidate_selector_max_thinking_tokens: Max thinking tokens (default: 4096)
                - candidate_selector_enable_domain_tools: Enable ToolUniverse MCP server (default: True)
                - candidate_selector_tooluniverse_path: Path to ToolUniverse installation (default: '/opt/tooluniverse-env' for Docker, './tooluniverse-env' for native)
                - candidate_selector_tool_selection_model: Model for tool selection (default: 'anthropic/claude-sonnet-4-5-20250929')
                - candidate_selector_tool_selection_models_file: Path to models config for tool selection (default: 'llm_configs/models.yaml')
                - candidate_selector_tool_selection_credentials_file: Path to credentials for tool selection (default: 'llm_configs/credentials.yaml')
        """
        if config is None:
            config = {}

        self.workspace_base = config.get('candidate_selector_workspace', 'candidate_selection_workspace')

        # Candidate selector agent configuration
        model_name = config.get('candidate_selector_model_name', 'claude_code/claude-sonnet-4-5-20250929')
        models_file = config.get('candidate_selector_models_file', 'llm_configs/claude_code.yaml')
        credentials_file = config.get('candidate_selector_credentials_file', 'llm_configs/credentials.yaml')
        self.run_in_docker = config.get('candidate_selector_run_in_docker', True)

        # Domain tools configuration
        self.enable_domain_tools = config.get('candidate_selector_enable_domain_tools', True)
        self.tooluniverse_path = config.get(
            'candidate_selector_tooluniverse_path',
            '/opt/tooluniverse-env' if self.run_in_docker else './tooluniverse-env'
        )

        self.max_thinking_tokens = config.get('candidate_selector_max_thinking_tokens', 4096)

        # Store config for later use
        self.models_file = models_file
        self.credentials_file = credentials_file
        self.model_name = model_name

        # Tool selection configuration
        self.tool_selection_model = config.get('candidate_selector_tool_selection_model', 'anthropic/claude-sonnet-4-5-20250929')
        self.tool_selection_models_file = config.get('candidate_selector_tool_selection_models_file', 'llm_configs/models.yaml')
        self.tool_selection_credentials_file = config.get('candidate_selector_tool_selection_credentials_file', 'llm_configs/credentials.yaml')

        # Initialize LLM client for tool selection if domain tools are enabled
        self.llm_client = None
        if self.enable_domain_tools:
            try:
                from scileo_agent.utils.llm import create_client
                self.llm_client = create_client(
                    model_name=self.tool_selection_model,
                    models_file=self.tool_selection_models_file,
                    credentials_file=self.tool_selection_credentials_file,
                )
                logger.debug(f"Initialized LLM client for tool selection with model: {self.tool_selection_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize LLM client for tool selection: {e}")
                self.llm_client = None

        # Initialize candidate selector agent (will be configured with MCP if needed)
        self.claude_agent = None  # Will be initialized in select() after tool selection
        self._last_selected_tools = None  # Track last selected tools to detect changes

        logger.debug(f"Initialized CandidateSelector with model: {model_name}")
        if self.enable_domain_tools:
            logger.debug(f"Domain tools enabled: tooluniverse_path={self.tooluniverse_path}")

    def _prepare_workspace(self, run_id: str) -> str:
        """
        Prepare a workspace for candidate selection.

        Args:
            run_id: Unique identifier for this selection run

        Returns:
            Path to the prepared workspace
        """
        # Create unique workspace with timestamp and run_id
        timestamp = time.strftime('%Y%m%d%H%M%S')
        workspace_name = f"selection_{timestamp}_{run_id}"
        workspace_path = os.path.join(self.workspace_base, workspace_name)

        # Create workspace directory
        os.makedirs(workspace_path, exist_ok=True)
        logger.debug(f"Created candidate selection workspace: {workspace_path}")

        return workspace_path

    def _save_candidates_to_file(
        self,
        populations: Dict[int, Population],
        workspace_path: str,
        serializer_name: str,
        high_level_goal: str
    ) -> str:
        """
        Save all candidates from multiple iterations to a single JSON file.

        Args:
            populations: Dict mapping iteration number to Population instance
            workspace_path: Path to the workspace directory
            serializer_name: Name of serializer to use for candidates
            high_level_goal: The optimization goal

        Returns:
            Path to the saved candidates file
        """
        # Get serializer
        serializer = get_serializer(serializer_name)
        if serializer is None:
            raise ValueError(f"Serializer '{serializer_name}' not found")

        # Prepare data structure
        total_candidates = sum(pop.size for pop in populations.values())
        candidates_data = {
            "optimization_goal": high_level_goal,
            "num_iterations": len(populations),
            "total_candidates": total_candidates,
            "iterations": []
        }

        # Process each population (iteration) - sort by iteration number
        for iter_num in sorted(populations.keys()):
            population = populations[iter_num]
            iteration_data = {
                "iteration": iter_num,
                "population_size": population.size,
                "population_scores": population.scores,  # Population-level scores
                "candidates": []
            }

            # Save each candidate
            for candidate in population.candidates:
                candidate_data = {
                    "id": candidate.id,
                    "representation": serializer.serialize(candidate),
                    "scores": candidate.scores,
                }
                iteration_data["candidates"].append(candidate_data)

            candidates_data["iterations"].append(iteration_data)

        # Write to file
        filename = "candidates_all_iterations.json"
        filepath = os.path.join(workspace_path, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(candidates_data, f, indent=2, default=str)

        logger.debug(f"Saved {total_candidates} candidates from {len(populations)} iterations to {filepath}")
        return filepath

    def _save_selection_results(
        self,
        result: Dict[str, Any],
        save_path: str,
        serializer_name: str,
        high_level_goal: str,
        num_to_select: int
    ) -> None:
        """
        Save selection results to a JSON file.

        Args:
            result: The result dictionary from select()
            save_path: Path to save the JSON file
            serializer_name: Name of serializer to use for candidates
            high_level_goal: The optimization goal
            num_to_select: Number of candidates that were requested
        """
        # Get serializer
        serializer = get_serializer(serializer_name)
        if serializer is None:
            raise ValueError(f"Serializer '{serializer_name}' not found")

        # Prepare data structure similar to candidates_all_iterations.json format
        selected_candidates = result["selected_candidates"]
        save_data = {
            "optimization_goal": high_level_goal,
            "num_requested": num_to_select,
            "num_selected": len(selected_candidates),
            "selection_reasoning": result["selection_reasoning"],
            "selection_metadata": result["selection_metadata"],
            "candidates": []
        }

        # Serialize each selected candidate
        for candidate in selected_candidates:
            candidate_data = {
                "id": candidate.id,
                "representation": serializer.serialize(candidate),
                "scores": candidate.scores,
            }
            save_data["candidates"].append(candidate_data)

        # Include usage stats if available
        if result["usage_stats"]:
            save_data["usage_stats"] = result["usage_stats"]

        # Include tool selection usage stats if available
        if result.get("tool_selection_usage_stats"):
            save_data["tool_selection_usage_stats"] = result["tool_selection_usage_stats"]

        # Write to file
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(save_data, f, indent=2, default=str)
            logger.info(f"Saved selection results to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save selection results to {save_path}: {e}")
            raise

    def _format_objectives_info(self, objectives_info: Dict[int, List[Objective]]) -> str:
        """
        Format objectives information for the prompt.

        Args:
            objectives_info: Dict mapping iteration number to list of Objective instances

        Returns:
            Formatted string describing the objectives used across iterations
        """
        if not objectives_info:
            raise ValueError("No objectives information available.")

        # Collect all unique objectives across all iterations
        all_objectives = {}  # objective_name -> Objective
        iteration_objectives = {}  # iteration -> list of objective names

        for iter_num in sorted(objectives_info.keys()):
            objectives = objectives_info[iter_num]
            iteration_objectives[iter_num] = []
            for obj in objectives:
                all_objectives[obj.name] = obj
                iteration_objectives[iter_num].append(obj.name)

        # Format output
        lines = []

        # Section 1: All unique objectives
        lines.append("**All Objectives Used Across Iterations:**\n")

        # Sort objectives by type
        type_order = {"candidate-wise": 0, "population-wise": 1, "filter": 2}
        sorted_objectives = sorted(all_objectives.values(), key=lambda x: type_order.get(x.type, 3))

        for i, obj in enumerate(sorted_objectives, 1):
            lines.append(f"{i}. **{obj.name}**")
            lines.append(f"   - Description: {obj.description}")
            lines.append(f"   - Type: {obj.type}")
            lines.append("")

        # Section 2: Objectives per iteration (with direction and weight)
        lines.append("\n**Objectives Used Per Iteration:**\n")
        for iter_num in sorted(iteration_objectives.keys()):
            objectives = objectives_info[iter_num]
            if iter_num == 0:
                continue  # Skip iteration 0 (initial population)
            else:
                lines.append(f"\n**Iteration {iter_num}**:")

            for obj in objectives:
                obj_info = f"  - {obj.name} ({obj.type})"
                if obj.type != "filter":
                    obj_info += f", direction: {obj.optimization_direction}"
                if obj.weight:
                    obj_info += f", weight: {obj.weight}"
                lines.append(obj_info)

        return "\n".join(lines)

    async def _select_domain_tools(
        self,
        high_level_goal: str,
        context_information: str,
        serializer_name: str,
        example_population: Population,
    ) -> Optional[Set[str]]:
        """
        Select relevant domain tools using LLM and ToolUniverse catalog.

        Args:
            high_level_goal: The overall optimization goal
            context_information: Additional context about the task
            serializer_name: Name of serializer to get candidate type/description
            example_population: A sample population to get example candidates

        Returns:
            Set of selected tool names, or None if selection fails
        """
        if not self.enable_domain_tools:
            return None

        if self.llm_client is None:
            logger.warning("No LLM client available for tool selection, skipping domain tools")
            return None

        logger.debug("Selecting relevant domain tools from ToolUniverse catalog...")

        try:
            # Get candidate type and description from serializer
            serializer = get_serializer(serializer_name)
            candidate_type = serializer.sample_schema
            candidate_description = serializer.sample_description

            # Get example candidate from population
            example_candidate = ""
            if example_population and example_population.size > 0:
                first_candidate = example_population.candidates[0]
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
            prompt = TOOL_SELECTION_PROMPT_TEMPLATE.format(
                high_level_goal=high_level_goal,
                context_information=context_information if context_information else "No additional context provided.",
                candidate_type=candidate_type,
                candidate_description=candidate_description,
                example_candidate=example_candidate,
                tools_text=tools_text
            )

            # Make the LLM call
            logger.debug("Calling LLM for tool selection...")
            response = await self.llm_client.call_async(
                [
                    {"role": "system", "content": TOOL_SELECTION_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
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

            return set(validated_tools)

        except Exception as e:
            logger.error(f"Tool selection failed: {e}")
            return None

    async def select(
        self,
        populations: Dict[int, Population],
        high_level_goal: str,
        context_information: str,
        serializer_name: str,
        num_to_select: int,
        objectives_info: Dict[int, List[Objective]],
        selected_domain_tools: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Select the best candidates from multiple iterations using ClaudeAgent.

        Args:
            populations: Dict mapping iteration number to Population instance.
                        Iteration 0 is the initial population, 1+ are optimization iterations.
            high_level_goal: The overall optimization goal
            context_information: Additional context information
            serializer_name: Name of serializer for getting candidate schema
            num_to_select: Number of candidates to select
            objectives_info: Dict mapping iteration number to list of Objective instances used in that iteration
            selected_domain_tools: Optional list of selected ToolUniverse tool names (if domain tools enabled)
            save_path: Optional path to save the selection results as JSON file

        Returns:
            Dict with keys:
                - selected_candidates (List[Candidate]): List of selected Candidate objects
                - selection_reasoning (str): Reasoning for the selection
                - selection_metadata (dict): Metadata about the selection
                - workspace_path (str): Path to selection workspace
                - usage_stats (dict): Claude agent usage statistics
                - tool_selection_usage_stats (dict): LLM client usage statistics for tool selection (None if no tool selection)
        """
        logger.debug(f"Starting candidate selection from {len(populations)} iterations")

        # Validate num_to_select
        total_candidates = sum(pop.size for pop in populations.values())
        if num_to_select >= total_candidates:
            raise ValueError(
                f"num_to_select ({num_to_select}) must be less than the total number of candidates ({total_candidates}). "
                f"Cannot select all or more candidates than available."
            )

        # Generate unique run ID
        run_id = str(uuid.uuid4())[:8]

        if selected_domain_tools is not None:
            selected_domain_tools = set(selected_domain_tools)
        
        # Step 1: Select domain tools if not provided and domain tools are enabled
        if selected_domain_tools is None and self.enable_domain_tools and self.llm_client:
            logger.debug("Domain tools enabled but not provided, selecting tools automatically...")
            # Get any population as example (prefer iteration 0 if available)
            example_population = populations.get(0) or (list(populations.values())[0] if populations else None)
            selected_domain_tools = await self._select_domain_tools(
                high_level_goal=high_level_goal,
                context_information=context_information,
                serializer_name=serializer_name,
                example_population=example_population,
            )

        # Step 2: Check if selected tools have changed and reinitialize agent if needed
        tools_changed = False
        if selected_domain_tools != self._last_selected_tools:
            tools_changed = True
            if self._last_selected_tools is not None:
                logger.info("Selected domain tools changed, reinitializing ClaudeAgent...")
            self._last_selected_tools = selected_domain_tools

        # Step 3: Configure MCP server with selected tools (if provided)
        mcp_servers = {}

        if self.enable_domain_tools and selected_domain_tools:
            # Configure ToolUniverse MCP server with --include-tools parameter
            mcp_servers["tooluniverse"] = {
                "command": "uv",
                "args": [
                    "--directory", self.tooluniverse_path,
                    "run", "tooluniverse-smcp-stdio",
                    "--include-tools"
                ] + list(selected_domain_tools)
            }

            logger.debug(f"ToolUniverse MCP server configured at: {self.tooluniverse_path}")
            logger.debug(f"Loading {len(selected_domain_tools)} pre-selected domain-specific tools")
            logger.debug(f"Selected tools: {selected_domain_tools}")

        elif self.enable_domain_tools:
            logger.warning("Tool selection failed or returned no tools; proceeding without domain tools")

        # Step 4: Initialize or reinitialize Claude agent if needed
        if self.claude_agent is None or tools_changed:
            if self.claude_agent is not None and tools_changed:
                # Clean up old agent if it exists
                logger.debug("Cleaning up previous ClaudeAgent instance...")
                del self.claude_agent

            logger.debug("Initializing ClaudeAgent with MCP configuration...")
            self.claude_agent = ClaudeAgent(
                model_name=self.model_name,
                models_file=self.models_file,
                credentials_file=self.credentials_file,
                run_in_docker=self.run_in_docker,
                system_prompt=SYSTEM_PROMPT,
                mcp_servers=mcp_servers if mcp_servers else None
            )
        else:
            logger.debug("Reusing existing ClaudeAgent instance (tools unchanged)")

        # Step 5: Prepare workspace
        workspace_path = self._prepare_workspace(run_id)

        # Step 6: Save all candidates to file
        candidates_file = self._save_candidates_to_file(
            populations,
            workspace_path,
            serializer_name,
            high_level_goal
        )
        candidates_file_basename = os.path.basename(candidates_file)

        # Step 7: Calculate statistics and format objectives for prompt
        total_candidates = sum(pop.size for pop in populations.values())
        num_iterations = len(populations)

        # Format objectives information
        objectives_info_str = self._format_objectives_info(objectives_info)

        # Format context information section
        if context_information:
            context_information_section = f"""
**Context information**: The following context information is provided by human scientists to help inform your selection:

{context_information}

**Note**: This context may contain background information, expectations, and focus areas that are relevant for your selection. However, it may also contain meta-instructions about objective proposing (e.g., "don't propose X", "always keep Y") which are NOT meant for you as the candidate selector. Focus only on information that helps you select the best candidates based on their quality, characteristics, and how well they satisfy the optimization goal.

"""
        else:
            context_information_section = ""

        # Step 8: Create user prompt
        serializer = get_serializer(serializer_name)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            high_level_goal=high_level_goal,
            context_information_section=context_information_section,
            num_iterations=num_iterations,
            total_candidates=total_candidates,
            candidates_file=candidates_file_basename,
            candidate_type=serializer.sample_schema,
            candidate_description=serializer.sample_description,
            objectives_info=objectives_info_str,
            num_to_select=num_to_select,
        )

        logger.debug("Formatted user prompt for candidate selection:\n" + user_prompt)

        # Step 9: Run candidate selector agent
        try:
            logger.info(f"Running candidate selector agent (workspace: {workspace_path})")
            usage_stats = await self.claude_agent.run(
                user_prompt=user_prompt,
                cwd=os.path.abspath(workspace_path),
                add_dirs=[],
                max_thinking_tokens=self.max_thinking_tokens
            )

            # Step 10: Read the selection results
            selection_results = self._extract_selection_results(workspace_path)

            # Step 11: Validate and extract selected candidates
            selected_candidate_ids = selection_results.get("selected_candidate_ids", [])

            if len(selected_candidate_ids) != num_to_select:
                logger.warning(f"Expected {num_to_select} candidates, but got {len(selected_candidate_ids)}")

            # Build a mapping of candidate ID to Candidate object
            candidate_map = {}
            for population in populations.values():
                for candidate in population.candidates:
                    candidate_map[candidate.id] = candidate

            # Extract the actual Candidate objects
            selected_candidates = []
            for cand_id in selected_candidate_ids:
                if cand_id in candidate_map:
                    selected_candidates.append(candidate_map[cand_id])
                else:
                    logger.warning(f"Selected candidate ID '{cand_id}' not found in populations")

            logger.debug(f"Successfully selected {len(selected_candidates)} candidates")

            # Prepare return data
            result = {
                "selected_candidates": selected_candidates,
                "selection_reasoning": selection_results.get("selection_reasoning", "No reasoning provided"),
                "selection_metadata": selection_results.get("selection_metadata", {}),
                "workspace_path": workspace_path,
                "usage_stats": usage_stats,
                "tool_selection_usage_stats": self.llm_client.stats if self.llm_client else None
            }

            # Save results to file if save_path is provided
            if save_path:
                self._save_selection_results(
                    result,
                    save_path,
                    serializer_name,
                    high_level_goal,
                    num_to_select
                )

            return result

        except Exception as e:
            logger.error(f"Error during candidate selection: {str(e)}")
            return {
                "selected_candidates": [],
                "selection_reasoning": f"Candidate selection failed: {str(e)}",
                "selection_metadata": {},
                "workspace_path": workspace_path,
                "usage_stats": None,
                "tool_selection_usage_stats": self.llm_client.stats if self.llm_client else None
            }

    def _extract_selection_results(self, workspace_path: str) -> Dict[str, Any]:
        """
        Extract the selection results from the workspace.

        Looks for selection_results.json file.

        Args:
            workspace_path: Path to the selection workspace

        Returns:
            Dictionary with selection results

        Raises:
            FileNotFoundError: If selection results file not found
            json.JSONDecodeError: If file cannot be parsed
        """
        results_file = os.path.join(workspace_path, 'selection_results.json')

        if not os.path.exists(results_file):
            logger.critical(f"No selection results file found in workspace `{workspace_path}`")
            raise FileNotFoundError(f"No selection results file found in workspace `{workspace_path}`.")

        try:
            with open(results_file, 'r', encoding='utf-8') as f:
                results = json.load(f)
                return results
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse selection results JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Failed to read selection results: {e}")
            raise
