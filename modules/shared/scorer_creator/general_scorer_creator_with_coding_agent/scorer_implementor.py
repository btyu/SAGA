import time
import os
import uuid
import shutil
import ast
import asyncio
from typing import Any, Optional, Dict, Tuple, List
import sys
import importlib.util
import json

from scileo_agent.utils.logging import get_logger
from scileo_agent.core.config import DEV_DEFAULT
from scileo_agent.core.registry.mcp_scorer_registry import load_module_scorers, McpScorerManager
from scileo_agent.core.registry.scorer_registry import ScorerManager
from scileo_agent.core.registry.serializer_registry import get_serializer

from .claude_code_agent import ClaudeAgent
from claude_agent_sdk import AgentDefinition
from scileo_agent.core.data_models.candidate import Candidate


logger = get_logger()


# Subagent definition for research phase
SCORER_RESEARCHER_AGENT = AgentDefinition(
    description="Use this subagent at the beginning to conduct thorough research on computational methods, algorithms, data sources, and implementation strategies for the scorer objective before starting implementation.",
    prompt="""You are a scientific research specialist with deep expertise in computational methods, bioinformatics, cheminformatics, machine learning, and scientific software development.

Your task is to conduct THOROUGH research to inform the implementation of a scientific scorer (evaluator function). You will receive an objective specification and must research the best computational approaches.

## Research Objectives

1. **Scientific Background & Methods**
   - Research the scientific principles and theoretical foundations
   - Identify established computational methods and algorithms
   - Find peer-reviewed papers, authoritative sources, and benchmarks
   - Determine if there are standard/gold-standard approaches

2. **Implementation Strategies**
   - Research existing implementations (GitHub, scientific libraries, etc.)
   - Identify appropriate Python packages and libraries
   - Find pre-trained models if applicable (with sources/URLs)
   - Identify APIs or databases that can be used programmatically

3. **Data Sources & Resources**
   - **CRITICAL**: For objectives requiring reference data (similarity comparisons, database lookups, statistical baselines), you MUST identify real, production-quality data sources
   - Research authoritative databases (e.g., ChEMBL, PubChem, UniProt, JASPAR, KEGG)
   - Find downloadable datasets or APIs for accessing data programmatically
   - Provide specific URLs, package names, or API endpoints
   - **NEVER** suggest using small hardcoded lists or toy data

4. **Technical Requirements**
   - Identify required computational dependencies
   - Determine if GPU acceleration is beneficial
   - Research typical computational complexity and performance considerations
   - Identify potential edge cases and error conditions

## Research Process

**Use WebSearch and WebFetch extensively** to find:
- Recent papers and reviews (prefer last 5 years)
- GitHub repositories with implementations
- Documentation for relevant packages
- Database documentation and access methods
- Benchmarking studies and comparisons

**Output Format**

Create a detailed research report (write to `research_findings.md` in the workspace) with:

### 1. Scientific Background
- Brief explanation of the objective
- Key scientific principles
- Standard approaches in the field

### 2. Recommended Computational Approach
- **Primary method**: Describe the algorithm/method to use
- **Justification**: Why this approach (cite sources)
- **Alternatives**: Other viable approaches (if any)

### 3. Implementation Details
- **Required packages**: List with installation commands
- **Pre-trained models**: Specific models to use (with sources/URLs)
- **Databases/APIs**: Exact databases/APIs with access methods and URLs
- **Code examples**: Links to reference implementations

### 4. Technical Specifications
- **Input processing**: How to handle input data
- **Computation steps**: High-level algorithmic steps
- **Output format**: Expected output structure
- **Performance**: Expected computational requirements

### 5. Edge Cases & Error Handling
- Invalid inputs to check for
- Potential failure modes
- Recommended validation strategies

### 6. Resources Summary
- URLs for datasets/databases
- GitHub repos for reference
- Package documentation links
- Relevant papers (with DOIs/links)

## Quality Standards

- **Thorough web research**: Use multiple searches with different queries
- **Authoritative sources**: Prioritize peer-reviewed papers, official docs, established libraries
- **Specific details**: Provide exact package names, URLs, model identifiers
- **Production-ready**: All recommendations must be suitable for automated deployment
- **No placeholders**: Never say "find a database" - actually find and specify it

## Important Notes

- This is for PRODUCTION use - no toy implementations or shortcuts
- All resources must be programmatically accessible (no manual downloads unless absolutely necessary)
- If real data sources cannot be found, clearly state this in your report
- Be thorough - this research directly impacts implementation success
""",
    tools=["WebSearch", "WebFetch", "Read", "Write", "Grep", "Glob"],
    model="sonnet"
)


USER_PROMPT_TEMPLATE = """You are an expert in scientific coding. Your task is to implement a robust and reliable Python-based scorer (evaluator) for a given optimization objective.

**IMPORTANT: Before starting implementation, you MUST use the 'scorer-researcher' subagent to conduct thorough research on computational methods, algorithms, and data sources. The research findings will guide your implementation.**

## Objective Specifications
- **Name**: {objective_name}
- **Type**: {type_description}
- **Description**: {objective_description}
- **Input Format**: {input_description}
- **Expected Output**: {output_description}

## Implementation Workflow

### Step 1: Research & Planning (MANDATORY)
**Use the scorer-researcher subagent to conduct thorough research:**

Invoke the 'scorer-researcher' subagent with the objective specifications. The subagent will:
- Research scientific background and established computational methods
- Identify appropriate packages, models, and algorithms
- Find real data sources and APIs (NO toy data or hardcoded lists)
- Create a detailed research report (`research_findings.md`)

**After research is complete:**
- Read and carefully study `research_findings.md`
- Use the recommended approaches, packages, and data sources
- Follow the implementation strategy outlined in the research
{reference_strategy}

### Step 2: Implementation Based on Research
You'll work with a pre-configured scorer template in the {objective_name} folder:

#### Primary Implementation (`{objective_name}/base.py`)
This is the main file you must implement. Specifically, implement the following methods in the predefined `Scorer` class:
- A `{objective_name}(samples)` method that takes a list of samples and returns {output_description}
- An optional `__init__()` method for loading models/data
Feel free to implement other helper methods in this file or other files as needed.

#### Dependencies (`{objective_name}/setup.sh`)
The runtime environment includes common scientific packages (pandas, numpy, scipy, scikit-learn, rdkit, torch, transformers, etc.) specified in the file. Only edit this file if you need additional packages not already installed.

#### Required Resources (`{objective_name}/scorer_data/`)
If your implementation requires pre-trained models, datasets, or other required files that cannot be downloaded programmatically, place them in this folder and ensure your code references them with correct relative paths.

#### Implementation Status (`{objective_name}/__implementation__.py`)
**CRITICAL**: You can ONLY mark the implementation as successful (`implementation_success = True`) when ALL of the following conditions are met:
1. The scorer is fully implemented and functional
2. All required packages can be installed automatically via `setup.sh`
3. All tests pass successfully in the Docker environment
4. No manual intervention or setup is required

If you cannot meet ALL these criteria (e.g., incomputable objective, missing required resources, packages that cannot be automatically installed, tests that fail, or any issues requiring manual intervention), you MUST put `implementation_success = False` in this file.

This is part of an AUTOMATED workflow with NO human involvement. Do NOT ask humans to manually install packages or perform setup steps. If automatic installation fails, try alternative packages or implementation methods, or mark the implementation as unsuccessful.

### Step 3: Testing & Validation
You should create a test file `test.py` in the workspace root that:
- Demonstrates that the implementation works correctly
- Verifies the output format and data types
- Tests the scorer with diverse, realistic samples
- Validates edge cases (invalid inputs, boundary conditions)

{test_examples_section}

Then, use the provided Docker testing system to test your implementation:
```bash
python docker_run.py --scorer_name {objective_name} python test.py
```
This command will start a Ubuntu 22.04 docker container with CUDA 12.4 support, mount the workspace to `/workspace` in the container, run `{objective_name}/setup.sh` to install dependencies, and execute your test file in the container.

You MUST test in the docker environment, as it mirrors the production environment exactly.

## Quality Standards

### Accuracy & Reliability
- Implement the exact objective as specified -- no approximations or shortcuts
- Use established, peer-reviewed methods when available
- Validate against known test cases or benchmarks if possible
- **ABSOLUTELY NO TOY IMPLEMENTATIONS**: If the scorer requires reference data (e.g., similarity comparisons, database lookups, statistical baselines), you MUST use real, production-quality data sources. Hardcoded lists of a few examples are UNACCEPTABLE and will cause the scorer to fail in production. Either:
  1. Download and use complete authoritative databases (preferred)
  2. Use APIs to access real-time data from established sources
  3. If real data is unavailable and you cannot implement a proper solution, mark `implementation_success = False`

### Robustness & Error Handling
- Always validate inputs before processing
- Wrap complex operations in try/except blocks to contain failures
- Return None for invalid inputs or failed computations
- Ensure that failures in one sample don't affect others in batch processing

### Logging Best Practices
- Use `loguru` exclusively (no `print` statements)
- Don't modify global logging configuration

### GPU Compatibility
- Always check GPU availability before attempting to use it
- Implement fallback CPU computation when GPU is not available

### Documentation Standards
- Clear docstrings for all `Scorer.{objective_name}`
- Inline comments for complex or domain-specific logic

## Success Criteria
**Your implementation succeeds when:**
- The scorer calculates the exact objective
- Output format exactly matches specifications
- Edge cases are handled gracefully
- Code is well-documented and readable
- Performance is acceptable for production use
- ALL dependencies can be installed automatically via `setup.sh` with NO manual intervention
- ALL tests pass successfully in the Docker environment with NO errors

**Common Pitfalls to Avoid:**
- **CRITICAL**: Using hardcoded toy data or creating small mock reference lists/databases instead of accessing real, authoritative data sources (e.g., hardcoding 10 molecules for similarity comparison instead of downloading from ChEMBL)
- Using naive approximations when established algorithms exist
- Using unreliable or deprecated models/methods
- Assuming GPU availability without checking
- Ignoring edge cases or error conditions
- Testing only outside the Docker environment
- **Marking implementation as successful when tests don't pass or require manual setup**
- **Asking humans to manually install packages or perform configuration steps**

{reference_section}

## Final Notes
This is production code that will be used in real scientific applications. Prioritize correctness, reliability, and robustness over speed or elegance. Take the time to research the best approaches and implement them properly. When in doubt, err on the side of being more thorough rather than cutting corners.
"""


CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class ScorerImplementor:
    def __init__(self, config: dict = None):
        if config is None:
            config = {}

        self.runtime_path = config.get('coding_workspace_path', 'coding_workspace')
        self.template_path = config.get('template_path', os.path.join(CURRENT_FILE_DIR, 'template'))
        self.generated_scorer_library_path = config.get("generated_scorer_library_path", "generated_scorers")
        self.scorer_library_subfolder = config.get("scorer_library_subfolder", None)
        self.dev = config.get('dev', DEV_DEFAULT)

        self.coding_agent_model = config.get('coding_agent_model', 'anthropic/claude-sonnet-4-20250514')
        coding_agent_models_file = config.get('coding_agent_models_file', os.path.join('llm_configs', 'claude_code.yaml'))
        coding_agent_credentials_file = config.get('coding_agent_credentials_file', os.path.join('llm_configs', 'credentials.yaml'))

        run_in_docker = config.get('coding_agent_run_in_docker', True)
        self.run_in_docker = run_in_docker

        # Define subagents for the coding agent
        subagents = {
            'scorer-researcher': SCORER_RESEARCHER_AGENT
        }

        # Initialize a single shared Claude Code Agent with subagents
        # Now safe for parallel execution thanks to per-run container tracking
        self.claude_code_agent = ClaudeAgent(
            model_name=self.coding_agent_model,
            models_file=coding_agent_models_file,
            credentials_file=coding_agent_credentials_file,
            run_in_docker=run_in_docker,
            agents=subagents
        )

    def _prepare_workspace(self, name: str, description: str, type: str, sample_schema: str, sample_description: str, reference_module_paths: list[str], serializer_name: str = None, test_candidates: Optional[List[Candidate]] = None):
        # Create a unique workspace path for the runtime
        workspace_path = os.path.join(self.runtime_path, time.strftime('%Y%m%d%H%M%S') + '_' + str(uuid.uuid4()))
        if os.path.isdir(workspace_path):
            shutil.rmtree(workspace_path)
        os.makedirs(workspace_path, exist_ok=True)

        logger.debug(f"Created workspace: {workspace_path}")
        
        # Prepare the scorer template
        name = name.strip().replace(' ', '_')
        scorer_path = os.path.join(workspace_path, f'{name}')
        os.makedirs(scorer_path, exist_ok=True)
        
        with open(os.path.join(scorer_path, '__implementation__.py'), 'w', encoding='utf-8') as f:
            pass
        
        with open(os.path.join(self.template_path, 'base.py.txt'), 'r') as f:
            base_template = f.read()
        
        name_text = name
        description_text = description
        sample_schema_text = sample_schema
        input_description_text = f"List of input samples, where each sample is {sample_description}"

        # Set output schema and description based on type
        if type == "population-wise":
            output_schema_text = "Optional[float]"
            output_description_text = "A single float score calculated based on the given samples; score can be None for invalid samples or invalid computations"
        elif type == "filter":
            # Filter type returns boolean values for pass/fail
            output_schema_text = "List[Optional[bool]]"
            output_description_text = "List of boolean values, each indicating whether the sample passes (True) or fails (False) the filter; can be None for invalid samples or invalid computations"
        else:  # candidate-wise
            output_schema_text = "List[Optional[float]]"
            output_description_text = "List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations"

        base_template = base_template.replace('<name>', name_text)
        base_template = base_template.replace('<type>', type)
        base_template = base_template.replace('<description>', description_text)
        base_template = base_template.replace('<sample_schema>', sample_schema_text)
        base_template = base_template.replace('<output_schema>', output_schema_text)
        base_template = base_template.replace('<input_description>', input_description_text)
        base_template = base_template.replace('<output_description>', output_description_text)

        with open(os.path.join(scorer_path, 'base.py'), 'w', encoding='utf-8') as f:
            f.write(base_template)

        shutil.copy(os.path.join(self.template_path, 'setup.sh.txt'), os.path.join(scorer_path, 'setup.sh'))
        shutil.copy(os.path.join(self.template_path, 'scorer_utils.py.txt'), os.path.join(scorer_path, 'scorer_utils.py'))
        
        # Copy Docker SDK version of docker_run.py
        docker_run_src = os.path.join(self.template_path, 'docker_run.py.txt')
        if os.path.exists(docker_run_src):
            shutil.copy(docker_run_src, os.path.join(workspace_path, 'docker_run.py'))
            logger.debug("Copied Docker SDK docker_run.py")
        else:
            logger.error(f"Docker SDK template not found: {docker_run_src}")
            raise FileNotFoundError(f"Docker SDK template not found: {docker_run_src}")

        # Copy reference modules
        exclude_files = {'__init__.py', '__main__.py', 'docker_run.py', 'mcp_server.py', 'test.sh'}
        for reference_module_path in reference_module_paths:
            reference_module_name = os.path.basename(reference_module_path)
            workspace_reference_module_path = os.path.join(workspace_path, 'references', reference_module_name)
            os.makedirs(workspace_reference_module_path, exist_ok=True)

            # Copy all files and directories except excluded ones
            for item in os.listdir(reference_module_path):
                if item not in exclude_files:
                    src_path = os.path.join(reference_module_path, item)
                    dst_path = os.path.join(workspace_reference_module_path, item)
                    if os.path.isdir(src_path):
                        shutil.copytree(src_path, dst_path, dirs_exist_ok=True)
                    else:
                        shutil.copy(src_path, dst_path)

            with open(os.path.join(workspace_reference_module_path, '__implementation__.py'), 'w', encoding='utf-8') as f:
                f.write('implementation_success = True')
            logger.debug(f"Added reference module: {reference_module_name}")

        # Serialize and save test candidates if provided
        if test_candidates and serializer_name:
            try:
                serializer = get_serializer(serializer_name)
                if serializer is None:
                    raise ValueError(f"Serializer '{serializer_name}' not found.")
                else:
                    # Serialize each candidate
                    serialized_samples = []
                    for candidate in test_candidates:
                        try:
                            serialized = serializer.serialize(candidate)
                            serialized_samples.append(serialized)
                        except Exception as e:
                            raise RuntimeError(f"Failed to serialize test candidate: {e}")

                    if serialized_samples:
                        # Save to test_examples.json in workspace
                        test_examples_path = os.path.join(workspace_path, 'test_examples.json')
                        with open(test_examples_path, 'w', encoding='utf-8') as f:
                            json.dump(serialized_samples, f, indent=2, default=str)
                        logger.debug(f"Saved {len(serialized_samples)} test example(s) to test_examples.json")
            except Exception as e:
                logger.warning(f"Failed to save test candidates: {e}")

        return workspace_path

    def _extract_scorers_from_base(self, base_py_path: str):
        """
        Parse the generated base.py without importing it, and extract scorer metadata
        from @scorer-decorated methods on class Scorer.

        Returns a dict mapping scorer_name -> metadata (excluding callable).
        """
        with open(base_py_path, 'r', encoding='utf-8') as f:
            source = f.read()

        module = ast.parse(source)
        scorers_meta = {}

        def _last_attr_name(node):
            # Return the rightmost name of a decorator like pkg.mod.scorer -> 'scorer'
            if isinstance(node, ast.Name):
                return node.id
            if isinstance(node, ast.Attribute):
                return _last_attr_name(node.attr) if isinstance(node.attr, ast.AST) else node.attr
            return None

        for node in module.body:
            if isinstance(node, ast.ClassDef) and node.name == 'Scorer':
                for sub in node.body:
                    if isinstance(sub, ast.FunctionDef):
                        for dec in sub.decorator_list:
                            if isinstance(dec, ast.Call) and _last_attr_name(dec.func) == 'scorer':
                                # Extract keyword args
                                kwargs = {kw.arg: kw.value for kw in dec.keywords if kw.arg}

                                def _const(value_node):
                                    if isinstance(value_node, ast.Constant):
                                        return value_node.value
                                    # Handle older AST node types for compatibility
                                    if hasattr(value_node, 'value'):
                                        return value_node.value
                                    # Fallback: best-effort string
                                    try:
                                        return ast.literal_eval(value_node)
                                    except Exception:
                                        return None

                                scorer_name = _const(kwargs.get('name'))
                                scorer_type = _const(kwargs.get('type'))
                                # Also check for legacy population_wise for backward compatibility
                                population_wise = _const(kwargs.get('population_wise'))
                                description = _const(kwargs.get('description'))
                                tool_description = ast.get_docstring(sub)

                                # Determine type: prefer explicit type, fallback to population_wise
                                if scorer_type is None and population_wise is not None:
                                    scorer_type = "population-wise" if population_wise else "candidate-wise"

                                if isinstance(scorer_name, str) and isinstance(scorer_type, str) and isinstance(description, str):
                                    scorers_meta[scorer_name] = {
                                        'function_name': sub.name,
                                        'type': scorer_type,
                                        'description': description,
                                        'tool_description': tool_description or '',
                                    }
        return scorers_meta

    def _write_scorers_to_init(self, module_dir: str, scorers_meta: dict):
        """
        Write a top-level `scorers` dict in module __init__.py so that
        metadata is accessible without importing the Scorer class.
        """
        init_path = os.path.join(module_dir, '__init__.py')

        # Build dictionary literal
        lines = ['# --- auto-generated scorers start ---', 'scorers: dict = {']
        for key, meta in scorers_meta.items():
            lines.append(f"    {repr(key)}: {{")
            lines.append(f"        'function_name': {repr(meta.get('function_name', ''))},")
            lines.append(f"        'type': {repr(meta.get('type', 'candidate-wise'))},")
            lines.append(f"        'description': {repr(meta.get('description', ''))},")
            lines.append(f"        'tool_description': {repr(meta.get('tool_description', ''))},")
            lines.append("    },")
        lines.append('}')
        lines.append('# --- auto-generated scorers end ---')
        
        # Add import for implementation_success
        lines.insert(0, 'try:')
        lines.insert(1, '    from .__implementation__ import implementation_success')
        lines.insert(2, 'except ImportError:')
        lines.insert(3, '    implementation_success = True')
        lines.insert(4, '')
        
        content = '\n'.join(lines) + '\n'
        
        with open(init_path, 'w', encoding='utf-8') as f:
            f.write(content)

    async def _write_scorer_async(self, workspace_path: str, name: str, description: str, type: str, sample_description: str, reference_modules_info: dict[str, dict[str, Any]], potential_matched_scorer_module: Optional[Dict[str, str]] = None):
        """Async version of _write_scorer for parallel execution."""
        if type == "population-wise":
            type_description = 'This is a POPULATION-WISE scorer, which means it calculates a single score for the entire list of samples (not per-sample).'
            output_description = 'a single float score for the given samples'
        elif type == "filter":
            type_description = 'This is a FILTER scorer, which means it evaluates each sample and returns True for valid/passing samples and False for invalid/failing samples.'
            output_description = 'a list of boolean values for each sample (True for valid/passing, False for invalid/failing)'
        else:  # candidate-wise
            type_description = 'This is a CANDIDATE-WISE scorer, which means it calculates a score for each sample independently.'
            output_description = 'a list of float scores for each sample'
        input_description = f'a list of input samples, where each sample is {sample_description}'

        if reference_modules_info:
            reference_section = '## Reference Modules\nThe `references/` folder contains implemented scorers that you can use as examples. Study the code (mainly `base.py`) of these scorers, especially ones similar to your objective, as they may provide helpful insights on implementation methods, models, packages, and algorithms to use.\n\n**IMPORTANT**: If you need to reuse checkpoints or data files from reference scorers, you MUST copy them to your module folder (e.g., `' + name + '/scorer_data/`) instead of creating symlinks. The reference folder will not be available in deployment, so all required resources must be physically present in your module directory.\n\n'
            for module_name, module_scorers in reference_modules_info.items():
                reference_section += f'Scorers in `references/{module_name}`:\n'
                for scorer_name, scorer_info in module_scorers.items():
                    # Get type from scorer info, with fallback to population_wise
                    scorer_type = scorer_info.get("type")
                    if scorer_type is None:
                        # Fallback: infer from population_wise for legacy scorers
                        is_population_wise = scorer_info.get("population_wise", False)
                        scorer_type = "population-wise" if is_population_wise else "candidate-wise"
                    reference_section += f'- {scorer_name}\n'
                    reference_section += f'  - Type: {scorer_type}\n'
                    reference_section += f'  - Description: {scorer_info["description"]}\n'
                reference_section += '\n'
            reference_strategy = '- Check the implemented scorers in the `references/` folder to get insights, as specified later\n'
        else:
            reference_section = ''
            reference_strategy = ''

        # Add section for potential matched scorers
        if potential_matched_scorer_module:
            # Validate that matched scorers are in reference modules
            for scorer_name, module_name in potential_matched_scorer_module.items():
                if module_name not in reference_modules_info:
                    raise ValueError(f"Potential matched scorer '{scorer_name}' references module '{module_name}' which is not in reference_module_paths")
                if scorer_name not in reference_modules_info[module_name]:
                    raise ValueError(f"Potential matched scorer '{scorer_name}' not found in reference module '{module_name}'")

            # Add note about particularly relevant scorers
            scorer_list = ', '.join([f'`{scorer_name}`' for scorer_name in potential_matched_scorer_module.keys()])
            reference_section += f'Among the reference scorers above, {scorer_list} may be particularly relevant to your implementation objective, so pay special attention to their implementations.\n\n'

        # Check if test_examples.json exists in workspace
        test_examples_path = os.path.join(workspace_path, 'test_examples.json')
        if os.path.exists(test_examples_path):
            test_examples_section = """**Available Test Examples**
The file `test_examples.json` in the workspace root contains a list of sample inputs that you can use for initial testing and debugging. These examples are provided as a reference but may be limited in scope or randomly selected. **You MUST still design comprehensive test cases** covering:
- Diverse, realistic samples beyond the provided examples
- Edge cases (invalid inputs, boundary conditions, extreme values)
- Various input patterns

"""
        else:
            test_examples_section = ''

        user_prompt = USER_PROMPT_TEMPLATE.format(
            objective_name=name,
            type_description=type_description,
            test_examples_section=test_examples_section,
            objective_description=description,
            input_description=input_description,
            output_description=output_description,
            reference_section=reference_section,
            reference_strategy=reference_strategy
        )

        # Use shared agent - now safe for parallel execution
        usage_stats = await self.claude_code_agent.run(
            user_prompt=user_prompt,
            cwd=os.path.abspath(workspace_path),
            add_dirs=[]
        )

        return usage_stats

    def _complete_mcp_server(self, workspace_path: str, name: str):
        module_path = os.path.join(workspace_path, name)
        shutil.copy(os.path.join(self.template_path, 'mcp_server.py.txt'), os.path.join(module_path, 'mcp_server.py'))

        # Copy Docker SDK version of __main__.py
        main_src = os.path.join(self.template_path, '__main__.py.txt')
        if os.path.exists(main_src):
            shutil.copy(main_src, os.path.join(module_path, '__main__.py'))
            logger.debug("Completed MCP server setup")
        else:
            logger.error(f"MCP template not found: {main_src}")
            raise FileNotFoundError(f"Docker SDK template not found: {main_src}")

    async def _verify_scorer_deployment(self, module_path: str, name: str, serializer_name: str) -> bool:
        """
        Verify that the scorer can be deployed and started successfully.

        Args:
            module_path: Path to the scorer module
            name: Name of the scorer

        Returns:
            True if scorer can be deployed and started successfully, False otherwise
        """

        scorer_manager = ScorerManager(run_in_docker=self.run_in_docker)
        mcp_manager = McpScorerManager(run_in_docker=self.run_in_docker)
        module_name = name  # Module name is the same as scorer name
        registered_scorer_names = []

        try:
            # Register the module with ScorerManager
            logger.debug(f"Attempting to register scorer module for deployment verification: {module_path}")
            num_registered = scorer_manager.register_mcp_module(module_path, serializer_name=serializer_name)

            if num_registered == 0:
                logger.error("No scorers were registered from the module")
                return False

            # Get the list of scorers registered from this module
            registered_scorer_names = [s for s, m in scorer_manager.mcp_scorer_to_module.items() if m == module_name]
            logger.debug(f"Successfully registered {num_registered} scorer(s) from module '{name}': {registered_scorer_names}")

            # Actually start the MCP server by calling _start_mcp_server
            logger.debug(f"Starting MCP server for module '{module_name}'...")
            server_started = mcp_manager._start_mcp_server(module_name)

            if not server_started:
                logger.error(f"Failed to start MCP server for module '{module_name}'")
                return False

            logger.debug(f"✓ MCP server started successfully for module '{module_name}'")
            return True

        except Exception as e:
            logger.error(f"Error during scorer deployment verification: {str(e)}")
            return False

        finally:
            # Clean up: unregister all scorers (which will also stop the server and clean up MCP manager)
            for scorer_name in registered_scorer_names:
                try:
                    scorer_manager.unregister_scorer(scorer_name)
                    logger.debug(f"Unregistered scorer '{scorer_name}' after verification")
                except Exception as e:
                    logger.warning(f"Error unregistering scorer '{scorer_name}': {str(e)}")

    def _copy_scorer_to_generated_library(self, workspace_path: str, name: str):
        if self.scorer_library_subfolder:
            generated_library_path = os.path.join(self.generated_scorer_library_path, self.scorer_library_subfolder)
        else:
            generated_library_path = self.generated_scorer_library_path

        module_workspace_path = os.path.join(workspace_path, name)
        module_library_path = os.path.join(generated_library_path, name)

        def ignore_cache_dirs(src, names):
            return {name for name in names if name == '__pycache__' or name.endswith('.pyc')}

        shutil.copytree(module_workspace_path, module_library_path, dirs_exist_ok=True, ignore=ignore_cache_dirs)
        logger.debug(f"Copied scorer to library: {module_library_path}")
        return generated_library_path

    def _get_reference_modules_info(self, reference_module_paths: list[str]) -> dict[str, dict[str, Any]]:
        reference_modules_info = {}
        scorer_names = set()
        for reference_module_path in reference_module_paths:
            module_name = os.path.basename(reference_module_path)
            if module_name in reference_modules_info:
                logger.error(f"Duplicate reference module: {module_name}")
                raise ValueError(f"Reference module `{module_name}` already exists")
            try:
                module_scorers = load_module_scorers(reference_module_path, return_raw=True)
            except Exception as e:
                logger.error(f"Failed to load reference module {module_name}: {str(e)}")
                raise ValueError(f"Error loading module scorer info from reference module {reference_module_path}: {e}") from e
            for scorer_name in module_scorers:
                if scorer_name in scorer_names:
                    logger.error(f"Duplicate scorer in references: {scorer_name}")
                    raise ValueError(f"Scorer `{scorer_name}` already exists`")
                scorer_names.add(scorer_name)
            reference_modules_info[module_name] = module_scorers
            logger.debug(f"Loaded reference module {module_name} with {len(module_scorers)} scorer(s)")
        return reference_modules_info

    async def process(
        self, name: str,
        description: str,
        type: str,
        serializer_name: str,
        sample_schema: str,
        sample_description: str,
        reference_module_paths: list[str] = [],
        potential_matched_scorer_module: Optional[Dict[str, str]] = None,
        test_candidates: Optional[List[Candidate]] = None,
    ) -> Tuple[Optional[str], str, str, bool, Optional[Dict[str, Any]]]:
        """
        Async version of process() for parallel scorer implementation.

        Args:
            name: Name of the scorer
            description: Description of what the scorer measures
            type: Type of scorer - "candidate-wise", "population-wise", or "filter"
            serializer_name: Name of the serializer to use
            sample_schema: Schema of the input samples
            sample_description: Description of the input samples
            reference_module_paths: Paths to reference modules for examples
            potential_matched_scorer_module: Dict mapping scorer names to module names
            test_candidates: Optional list of example candidates for testing. These will be serialized
                           and saved as test_examples.json for reference during implementation.

        Returns:
            Tuple of (generated_library_path, workspace_path, name, implementation_success, coding_agent_usage_stats)
        """
        name = name.strip().replace(' ', '_')

        logger.info(f"Implementing scorer (async): {name}\nDescription: {description}\nType: {type}\nSample schema: {sample_schema}\nSample description: {sample_description}")

        # Load reference modules (sync, but fast)
        reference_modules_info = self._get_reference_modules_info(reference_module_paths)
        if reference_modules_info:
            logger.info(f"Loaded {len(reference_modules_info)} reference module(s)")

        # Prepare workspace (sync, but fast and isolated per scorer)
        workspace_path = self._prepare_workspace(name, description, type, sample_schema, sample_description, reference_module_paths, serializer_name, test_candidates)

        # Implement scorer with coding agent (async - main time consumer)
        coding_agent_usage_stats = None
        try:
            logger.info(f"Starting coding agent to implement scorer: {name}")
            coding_agent_usage_stats = await self._write_scorer_async(workspace_path, name, description, type, sample_description, reference_modules_info, potential_matched_scorer_module)
        except Exception as e:
            logger.error(f"Scorer implementation failed: {str(e)}")
            if self.dev:
                raise e
            else:
                return None, workspace_path, name, False, coding_agent_usage_stats

        # Check if implementation was successful (sync)
        # logger.debug("Checking implementation status")
        module_dir = os.path.join(workspace_path, name)
        init_file_path = os.path.join(module_dir, '__init__.py')

        implementation_success = True  # Default to True
        if os.path.exists(init_file_path):
            try:
                # Import the module dynamically to check implementation_success
                spec = importlib.util.spec_from_file_location(f"temp_scorer_{name}", init_file_path)
                if spec is not None and spec.loader is not None:
                    temp_module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(temp_module)

                    # Check if implementation_success attribute exists and is False
                    if hasattr(temp_module, 'implementation_success'):
                        implementation_success = temp_module.implementation_success

                    # Clean up the temporary module
                    if f"temp_scorer_{name}" in sys.modules:
                        del sys.modules[f"temp_scorer_{name}"]
            except Exception as e:
                logger.debug(f"Could not import module for status check: {str(e)}")
                # If we can't import the module, fall back to string checking
                with open(init_file_path, 'r') as f:
                    init_content = f.read()
                    if 'implementation_success = False' in init_content:
                        implementation_success = False

        if implementation_success:
            logger.debug("✓ Scorer implementation successful")

            # Extract and expose scorers metadata in __init__.py
            try:
                base_py = os.path.join(module_dir, 'base.py')
                scorers_meta = self._extract_scorers_from_base(base_py)
                if scorers_meta:
                    self._write_scorers_to_init(module_dir, scorers_meta)
                    logger.debug(f"Extracted {len(scorers_meta)} scorer(s) from implementation")
            except Exception as e:
                logger.warning(f"Could not extract scorer metadata: {str(e)}")

            self._complete_mcp_server(workspace_path, name)

            # Verify deployment before marking as successful
            logger.debug("Verifying scorer deployment...")
            deployment_successful = await self._verify_scorer_deployment(module_dir, name, serializer_name)

            if deployment_successful:
                logger.debug("✓ Scorer deployment verification successful")
                generated_library_path = self._copy_scorer_to_generated_library(workspace_path, name)
                logger.debug(f"Scorer installed to: {generated_library_path}")
            else:
                logger.error("✗ Scorer deployment verification failed - marking implementation as unsuccessful")
                implementation_success = False
                generated_library_path = None

                # Update the __implementation__.py file to reflect deployment failure
                impl_file_path = os.path.join(module_dir, '__implementation__.py')
                with open(impl_file_path, 'w', encoding='utf-8') as f:
                    f.write('implementation_success = False\n')
                    f.write('# Deployment verification failed: scorer could not be registered and run successfully\n')
        else:
            logger.warning("✗ Scorer implementation marked as unsuccessful")
            generated_library_path = None

        return generated_library_path, workspace_path, name, implementation_success, coding_agent_usage_stats



if __name__ == "__main__":
    scorer_implementor = ScorerImplementor({})
    generated_library_path, workspace_path, name, success, usage_stats = asyncio.run(
        scorer_implementor.process_async(
            name="logp_score",
            description="Lipophilicity (LogP) druglikeness score (0-1). LogP measures the partition coefficient between octanol and water, indicating lipophilicity/hydrophobicity. Optimal LogP balance is crucial for drug absorption, distribution, and membrane permeability. Values <2 (score=1) indicate good water solubility and permeability, >4 (score=0) suggest poor ADMET properties due to excessive lipophilicity.",
            type="candidate-wise",
            sample_schema="str",
            sample_description="a SMILES string of a molecule")
    )
    print(f"Library path: {generated_library_path}, Workspace: {workspace_path}, Success: {success}")

