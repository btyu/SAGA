"""
Candidate Analyzer for analyzing specific candidates in the population.

This module provides a specialized analyzer that uses ClaudeAgent to perform
detailed analysis of individual candidates in the optimization population.

The analyzer supports domain-specific tools via ToolUniverse MCP server, which
can be enabled through the config dictionary. Tool selection is handled by the
parent BasicAnalyzer, which passes the selected tools to this analyzer.

Configuration Example:
    config = {
        # Basic configuration
        'candidate_analyzer_workspace': 'candidate_analysis_workspace',
        'candidate_analyzer_model_name': 'claude_code/claude-sonnet-4-5-20250929',
        'candidate_analyzer_models_file': 'llm_configs/claude_code.yaml',
        'candidate_analyzer_credentials_file': 'llm_configs/credentials.yaml',
        'candidate_analyzer_run_in_docker': True,
        'candidate_analyzer_max_thinking_tokens': 4096,

        # Domain tools configuration
        'candidate_analyzer_enable_domain_tools': True,  # Enable ToolUniverse MCP server
        'candidate_analyzer_tooluniverse_path': '/opt/tooluniverse-env',  # Docker: /opt/tooluniverse-env, Native: ./tooluniverse-env
        'candidate_analyzer_tool_selection_model': 'anthropic/claude-sonnet-4-5-20250929',  # Model for tool selection (in BasicAnalyzer)
    }

    analyzer = CandidateAnalyzer(config)

Tool Selection Architecture:
    1. BasicAnalyzer loads ToolUniverse catalog from tool_universe_tools.json
    2. BasicAnalyzer uses its LLMClient to select relevant tools based on optimization goal
    3. Selected tools are passed to CandidateAnalyzer.analyze()
    4. CandidateAnalyzer configures MCP server with only the selected tools
    5. ClaudeAgent runs with access to the filtered tool set
"""

import os
import time
import uuid
import shutil
import json
import tempfile
from typing import Dict, Any, List, Optional
from pathlib import Path

from scileo_agent.utils.logging import get_logger
from scileo_agent.core.data_models import Population
from scileo_agent.core.registry.serializer_registry import get_serializer

from .claude_agent import ClaudeAgent


logger = get_logger()


SYSTEM_PROMPT = """You are an expert optimization analyst specializing in candidate-level analysis. Your task is to examine specific candidates from an optimization population and provide detailed insights about their characteristics, quality, and potential.

You have access to file reading, code execution, web search, and other tools to thoroughly analyze the candidates. Use these tools to:
1. Understand the candidate structure and properties
2. Identify patterns, strengths, and weaknesses
3. Compare candidates to understand diversity and quality distribution
4. Provide actionable insights for optimization strategy

Your analysis MUST be grounded in verifiable evidence from one or more of the following sources:
- Direct observations and measurements from examining actual candidate data
- Computational results from executing code to analyze candidates
- Domain-specific tools (if any) applied to evaluate candidate properties
- Authoritative information retrieved from online sources via web search

CRITICAL: Base ALL findings, conclusions, and insights strictly on factual evidence. NEVER fabricate data, invent measurements, or hallucinate patterns that are not demonstrably present in the actual candidates or verifiable through the tools available to you. If you cannot verify a hypothesis or observation, explicitly state this limitation rather than speculating.
"""


USER_PROMPT_TEMPLATE = """# Candidate Analysis Task

## Optimization Context
**Optimization goal**: {high_level_goal}
{context_information_section}
## Population Information
The population to analyze is the output of iteration {iteration_number}, containing {population_size} candidates.

### Objectives
In this iteration, candidates are optimized using the following objectives:

{objectives_info}

### Performance Summary

{performance_summary}

### Specific Candidates
The specific candidates of this population have been saved to the population file `{population_file}`. Use this file to analyze the candidates. It is a JSON file containing all candidates with their representations and objective scores. Its structure is as follows:

```JSON
{{
  "iteration": {iteration_number},
  "size": 156,
  "population_scores": {{  # population-wise objective scores
    "objective_name": float_value,
    ...
  }},
  "candidates": [  # specific candidates
    {{
      "id": "...",  # unique candidate ID
      "representation": "...",  # the candidate representation
      "scores": {{  # candidate-wise objective scores of this candidate
        "objective_name": float_value,
        ...
      }}
    }},
    ...
  ]
}}
```

Each candidate representation ({candidate_type}) is {candidate_description}


## Your Task
Your primary goal is to **perform a comprehensive, fact-based analysis of the candidates** to understand their quality, characteristics, and patterns. This analysis should provide deep insights into the candidate population that go beyond what the objective scores alone reveal.

### Recommended Analysis Workflow

While you have flexibility to adapt your approach as needed, here is a suggested workflow:

1. **Conceptualize Quality Criteria**: Based on the optimization goal, reflect on what properties, features, or characteristics make a candidate "good" or "bad". Consider domain-specific aspects that matter for this optimization problem.

2. **Quantitative Exploration**: Write Python code to systematically explore the candidates:
   - Analyze distributions of scores and other measurable properties
   - Compute domain-specific metrics or features using appropriate packages
   - Identify patterns, correlations, clusters, or outliers

3. **Qualitative Examination**: Take a closer look at specific candidates:
   - Sample diverse candidates (e.g., best/worst performers, outliers, representative examples)
   - Examine them using domain knowledge and domain-specific tools
   - Use code or specialized tools to validate observations
   - Identify concrete examples that illustrate important patterns

4. **Synthesize Findings**: Write a comprehensive analysis report summarizing:
   - Key characteristics and quality patterns observed
   - Distribution of important properties across the population
   - Specific strengths and weaknesses found in candidates
   - Any quality aspects or failure modes NOT adequately captured by current objectives
   - Concrete examples supporting your findings

### Key Focus Areas

- **What makes candidates good or bad** relative to the optimization goal
- **Patterns and diversity** in candidate properties and structures
- **Domain-specific quality aspects** that may not be fully measured by existing objectives
- **Failure modes or limitations** that objective scores might miss
- **Concrete, verifiable observations** backed by code analysis or domain tools

### Important Guidelines

- **Use tools extensively**: Rely on Python code, domain-specific packages, and scientific tools (provided as MCP tools if any) to extract accurate information rather than manual inspection alone. This helps avoid hallucination and ensures reproducibility.
- **Leverage available resources**: The environment has extensive Python packages pre-installed (see below). If you need additional tools, you can install them via conda, pip, mamba, or apt.
- **Ground all findings in evidence**: Every observation should be backed by actual data, code output, or tool-based analysis.
- **Balance breadth and depth**: Combine population-level statistics with detailed examination of specific candidates.
- **Focus on actionable insights**: Provide findings that help understand candidate quality and guide optimization strategy.

### Available Python Environment

The environment includes extensive scientific computing, machine learning, and domain-specific packages:

**Core Scientific Computing**: numpy, scipy, pandas, matplotlib, plotly, seaborn, scikit-learn, jupyter

**Machine Learning & Deep Learning**: pytorch (with CUDA), transformers, datasets, pytorch-lightning, dgl, optuna, wandb, tensorboard

**Chemistry & Materials Science**: rdkit, pymatgen, ase, openmm, openbabel, xtb-python, alignn, unidock, descriptastorus, drug-likeness

**Bioinformatics**: biopython, grelu

**Other Tools**: ray, xgboost, umap-learn, pydantic, litellm, and more

If you need additional packages or tools, install them as needed.

### Output Format
Write your analysis as a comprehensive report in markdown format to a file called `candidate_analysis.md` in the working directory. Since it will be inserted into the main analyzer's context as a subsection, start directly with level 3 headings (###) or lower for your content sections.
"""


class CandidateAnalyzer:
    """
    Analyzer for detailed candidate-level analysis using ClaudeAgent.

    This class manages workspace preparation, prompt formation, and execution
    of candidate analysis using a ClaudeAgent with full tool access.
    """

    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the candidate analyzer.

        Args:
            config: Configuration dictionary with optional keys:
                - candidate_analyzer_workspace: Base path for analysis workspaces (default: 'candidate_analysis_workspace')
                - candidate_analyzer_model_name: Model name for candidate analyzer (default: 'claude_code/claude-sonnet-4-5-20250929')
                - candidate_analyzer_models_file: Path to models config (default: 'llm_configs/claude_code.yaml')
                - candidate_analyzer_credentials_file: Path to credentials (default: 'llm_configs/credentials.yaml')
                - candidate_analyzer_run_in_docker: Whether to run in Docker (default: True)
                - candidate_analyzer_max_thinking_tokens: Max thinking tokens (default: 4096)
                - candidate_analyzer_enable_domain_tools: Enable ToolUniverse MCP server (default: False)
                - candidate_analyzer_tooluniverse_path: Path to ToolUniverse installation (default: '/opt/tooluniverse-env' for Docker, './tooluniverse-env' for native)
        """
        if config is None:
            config = {}

        self.workspace_base = config.get('candidate_analyzer_workspace', 'candidate_analysis_workspace')

        # Candidate analyzer agent configuration
        model_name = config.get('candidate_analyzer_model_name', 'claude_code/claude-sonnet-4-5-20250929')
        models_file = config.get('candidate_analyzer_models_file', 'llm_configs/claude_code.yaml')
        credentials_file = config.get('candidate_analyzer_credentials_file', 'llm_configs/credentials.yaml')
        self.run_in_docker = config.get('candidate_analyzer_run_in_docker', True)

        # Domain tools configuration
        self.enable_domain_tools = config.get('candidate_analyzer_enable_domain_tools', True)
        self.tooluniverse_path = config.get(
            'candidate_analyzer_tooluniverse_path',
            '/opt/tooluniverse-env' if self.run_in_docker else './tooluniverse-env'
        )

        self.max_thinking_tokens = config.get('candidate_analyzer_max_thinking_tokens', 4096)

        # Store config for later use
        self.models_file = models_file
        self.credentials_file = credentials_file
        self.model_name = model_name

        # Initialize candidate analyzer agent (will be configured with MCP if needed)
        self.claude_agent = None  # Will be initialized in analyze() after tool selection
        self._last_selected_tools = None  # Track last selected tools to detect changes

        logger.debug(f"Initialized CandidateAnalyzer with model: {model_name}")
        if self.enable_domain_tools:
            logger.debug(f"Domain tools enabled: tooluniverse_path={self.tooluniverse_path}")

    def _prepare_workspace(self, iteration_number: int, population_file: str) -> str:
        """
        Prepare a workspace for candidate analysis.

        Args:
            iteration_number: Current iteration number
            population_file: Path to the saved population file

        Returns:
            Path to the prepared workspace
        """
        # Create unique workspace with timestamp and UUID
        timestamp = time.strftime('%Y%m%d%H%M%S')
        workspace_id = str(uuid.uuid4())[:8]
        workspace_name = f"iter_{iteration_number}_{timestamp}_{workspace_id}"
        workspace_path = os.path.join(self.workspace_base, workspace_name)

        # Create workspace directory
        os.makedirs(workspace_path, exist_ok=True)
        logger.debug(f"Created candidate analysis workspace: {workspace_path}")

        # Copy population file to workspace
        if population_file and os.path.exists(population_file):
            dest_file = os.path.join(workspace_path, os.path.basename(population_file))
            shutil.copy(population_file, dest_file)
            logger.debug(f"Copied population file to workspace: {dest_file}")
        else:
            logger.warning(f"Population file not found or not provided: {population_file}")

        return workspace_path

    async def analyze(
        self,
        iteration_number: int,
        high_level_goal: str,
        context_information: str,
        current_population: Population,
        population_file: str,
        serializer_name: str,
        objectives_info: str,
        performance_results: str,
        selected_domain_tools: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Perform detailed candidate analysis using ClaudeAgent.

        Args:
            iteration_number: Current iteration number
            high_level_goal: The overall optimization goal
            context_information: Additional context information
            current_population: Current population of candidates
            population_file: Path to the saved population JSON file
            serializer_name: Name of serializer for getting candidate schema
            objectives_info: Pre-formatted objectives information
            performance_results: Pre-formatted performance results
            selected_domain_tools: Optional list of selected ToolUniverse tool names (if domain tools enabled)

        Returns:
            Dict with keys:
                - candidate_analysis_report (str): Detailed analysis report
                - workspace_path (str): Path to analysis workspace
                - usage_stats (dict): Claude agent usage statistics
        """
        logger.debug(f"Starting candidate analysis for iteration {iteration_number}")

        # Step 1: Check if selected tools have changed and reinitialize agent if needed
        tools_changed = False
        if selected_domain_tools != self._last_selected_tools:
            tools_changed = True
            if self._last_selected_tools is not None:
                logger.info("Selected domain tools changed, reinitializing ClaudeAgent...")
            self._last_selected_tools = selected_domain_tools

        # Step 2: Configure MCP server with selected tools (if provided)
        mcp_servers = {}

        if self.enable_domain_tools and selected_domain_tools:
            # Configure ToolUniverse MCP server with --include-tools parameter
            # Format: --include-tools "tool_1" "tool_2" "tool_3"
            # This loads only the selected tools, making initialization faster and more efficient
            mcp_servers["tooluniverse"] = {
                "command": "uv",
                "args": [
                    "--directory", self.tooluniverse_path,
                    "run", "tooluniverse-smcp-stdio",
                    "--include-tools"
                ] + selected_domain_tools  # Add tool names as separate arguments (will be quoted by subprocess)
            }

            logger.debug(f"ToolUniverse MCP server configured at: {self.tooluniverse_path}")
            logger.debug(f"Loading {len(selected_domain_tools)} pre-selected domain-specific tools")
            logger.debug(f"Selected tools: {selected_domain_tools}")

        elif self.enable_domain_tools:
            logger.warning("Tool selection failed or returned no tools; proceeding without domain tools")

        # Step 3: Initialize or reinitialize Claude agent if needed
        if self.claude_agent is None or tools_changed:
            if self.claude_agent is not None and tools_changed:
                # Clean up old agent if it exists (though ClaudeAgent handles cleanup internally)
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

        # Prepare workspace
        workspace_path = self._prepare_workspace(iteration_number, population_file)

        # Get relative path to population file from workspace
        population_file_basename = os.path.basename(population_file) if population_file else "population.json"

        # Format context information section
        if context_information:
            context_information_section = f"""
**Context information**: The following context information is provided by human scientists to help inform your analysis:

{context_information}

**Note**: This context may contain background information, expectations, and focus areas that are relevant for your analysis. However, it may also contain meta-instructions about objective proposing (e.g., "don't propose X", "always keep Y") which are NOT meant for you as the candidate analyzer. Focus only on information that helps you evaluate candidate quality, characteristics, and patterns in the population.

"""
        else:
            context_information_section = ""

        # Create user prompt using pre-formatted information
        serializer = get_serializer(serializer_name)
        user_prompt = USER_PROMPT_TEMPLATE.format(
            high_level_goal=high_level_goal,
            context_information_section=context_information_section,
            iteration_number=iteration_number,
            population_size=current_population.size,
            population_file=population_file_basename,
            objectives_info=objectives_info,
            performance_summary=performance_results,
            candidate_type=serializer.sample_schema,
            candidate_description=serializer.sample_description,
        )
        
        logger.debug("Formatted user prompt for candidate analysis:\n" + user_prompt)

        # Run candidate analyzer agent
        try:
            logger.debug(f"Running candidate analyzer agent (workspace: {workspace_path})")
            usage_stats = await self.claude_agent.run(
                user_prompt=user_prompt,
                cwd=os.path.abspath(workspace_path),
                add_dirs=[],
                max_thinking_tokens=self.max_thinking_tokens
            )

            # Read the analysis output
            # The agent should write its analysis to a file or we can extract it from logs
            # For now, we'll look for a common output file pattern
            analysis_report = self._extract_analysis_report(workspace_path)

            logger.debug("Candidate analysis report:\n" + analysis_report)

            return {
                "candidate_analysis_report": analysis_report,
                "workspace_path": workspace_path,
                "usage_stats": usage_stats
            }

        except Exception as e:
            logger.error(f"Error during candidate analysis: {str(e)}")
            return {
                "candidate_analysis_report": f"Candidate analysis failed: {str(e)}",
                "workspace_path": workspace_path,
                "usage_stats": None
            }

    def _extract_analysis_report(self, workspace_path: str) -> str:
        """
        Extract the analysis report from the workspace.

        Looks for common output files, prioritizing candidate_analysis.md

        Args:
            workspace_path: Path to the analysis workspace

        Returns:
            The analysis report content, or a default message if not found
        """
        # Common filenames for analysis output (in priority order)
        candidate_files = [
            'candidate_analysis.md',  # Explicitly requested filename
            # 'analysis.md',
            # 'report.md',
            # 'README.md',
            # 'analysis.txt',
            # 'report.txt'
        ]

        for filename in candidate_files:
            filepath = os.path.join(workspace_path, filename)
            if os.path.exists(filepath):
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        if content:
                            # logger.debug(f"Found analysis report in: {filename}")
                            return content
                except Exception as e:
                    logger.warning(f"Failed to read {filename}: {e}")

        # If no report file found, provide a note
        logger.critical("No analysis report file found in workspace `{workspace_path}`")
        raise FileNotFoundError(f"No analysis report file found in workspace `{workspace_path}`.")
