#!/usr/bin/env python3
"""
Run SAGA K. pneumoniae Antibiotics Design (Levels 0-3)

This script provides experiments for K. pneumoniae antibiotic optimization with comprehensive configurations.
Supports levels 0-3 via command-line flag.

Usage:
    python exp_kp.py --level 0    # Level 0: 1 iteration, both feedbacks enabled
    python exp_kp.py --level 1    # Level 1: Multiple iterations, both feedbacks enabled
    python exp_kp.py --level 2    # Level 2: Multiple iterations, analyzer feedback only
    python exp_kp.py --level 3    # Level 3: Multiple iterations, fully automated

NOTE ON HUMAN FEEDBACK:
- When `enable_human_feedback` is True (for planner and/or analyzer), the script will pause
  and prompt you for input via the terminal when human feedback is needed.
- For planner feedback: You'll see proposed objectives and can accept them (option 1) or
  provide revised objectives in JSON format (option 2).
- For analyzer feedback: You'll see analysis reports and can accept or revise them.
- The interactive prompts will appear automatically when needed - just follow the instructions.
"""


################################################################################
# # Run SAGA Level 0 - K. pneumoniae Antibiotics Design
#
# This notebook provides a Level 0 experiment for K. pneumoniae antibiotic optimization with comprehensive configurations.
################################################################################


################################################################################
# **Note**: This script supports multiple experiment levels:
# - Level 0: 1 iteration, both feedbacks enabled
# - Level 1: Multiple iterations, both feedbacks enabled
# - Level 2: Multiple iterations, analyzer feedback only
# - Level 3: Multiple iterations, fully automated (no feedback)
# Use --level flag to specify which level to run.
################################################################################


################################################################################
# ## Device Requirements
#
# - Must be able to run docker, because the coding agent will need it to debug and test scorers that it creates.
# - GPUs are optional. If your scorers rely on running some deep learning models, having GPUs would be much quicker.
# - Better be a Linux machine. Not sure if you'd meet any issue on other OSs.
################################################################################



################################################################################
# ## Part 1: Set up Configurations for an Experiment
################################################################################


################################################################################
# ### Step 1.1: Import Everything
################################################################################


# Cell 12
import os
import sys
import json
import argparse
import asyncio
import subprocess
from pprint import pprint
from datetime import datetime
from pathlib import Path


# Ensure local MCP servers bypass any configured HTTP(S) proxies
def _ensure_localhost_no_proxy():
    def _inject(var_name: str):
        current = os.environ.get(var_name, "")
        tokens = [token.strip() for token in current.split(",") if token.strip()]
        for host in ("localhost", "127.0.0.1", "::1"):
            if host not in tokens:
                tokens.append(host)
        os.environ[var_name] = ",".join(tokens)

    _inject("NO_PROXY")
    _inject("no_proxy")


_ensure_localhost_no_proxy()

# Optimize MCP batch sizes for better performance
import os
# Increase batch size for MCP scorers (default is 1000, increase for better throughput)
os.environ.setdefault("SCILEO_MCP_MAX_BATCH_SIZE", "5000")
# Increase SSE timeout for large batches
os.environ.setdefault("SCILEO_MCP_SSE_READ_TIMEOUT", "3600")


# Cell 13
# This changes the working directory to the root directory of the project.
current_dir = os.getcwd()
split_path = current_dir.split("/")
project_root_index = split_path.index("SAGA")
project_root_dir = "/".join(split_path[: project_root_index + 1])
os.chdir(project_root_dir)
if project_root_dir not in sys.path:
    sys.path.insert(0, os.getcwd())
print("Current working directory:", os.getcwd())


# Cell 14
# Import the core modules
from scileo_agent.core.config import create_config
from scileo_agent.core.orchestrator import OptimizationOrchestrator
from scileo_agent.core.registry import (
    list_registered_modules,
    get_scorer,
    list_scorers,
    register_mcp_module,
    get_serializer,
    list_serializers,
    reset_scorer_manager,
    get_scorer_metadata,
    ScorerManager,
)
from scileo_agent.core.data_models import Candidate, Population, Objective
from scileo_agent.utils.logging import get_logger, setup_logging


# Cell 15
# Import the logging utils
logger = setup_logging(level="DEBUG")  # Setup the logger


# Cell 16
# Import the shared modules (e.g., the planner, the scorer creator, the analyzer)
# 🔴 Please make sure that your serializer is imported here.
from modules.shared import (
    planner,
    scorer_creator,
    analyzer,
    knowledge_manager,
    serializer,
)


# Cell 17
# Check the available scorers
# At this point, there should not be any scorers registered
scorer_names = list_scorers()
print(scorer_names)
if len(scorer_names) > 0:
    raise RuntimeError(
        "There should not be any scorers registered at this point. Check the above imports and make sure no scorer is imported along with them."
    )


# Cell 18
# Get the serializer
# 🔴 Needs your customization

serializer_name = "smiles_serializer"

serializer = get_serializer(serializer_name)
if serializer is None:
    raise RuntimeError(
        f"Serializer '{serializer_name}' not found. Registered serializers: {list_serializers()}. If your serializer is not listed, make sure it's imported in `modules/shared/serializer/__init__.py`."
    )


# Cell 19
# Import your optimizer
# 🔴 Needs your customization

from modules.small_molecule_drug_design.llm_sbdd_optimizer import LLMSBDDOptimizer


# Cell 20
# Check the available modules
# 🔴 Needs your check: Make sure all modules to use are imported, especially your custom optimizer

pprint(list_registered_modules())


# Cell 21
# Import the MCP scorers
# 🔴 Needs your customization

# Put ONLY the initial objectives' MCP scorer module paths for your task here
# They will be registered in the scorer library
mcp_scorer_paths = [
    "modules/small_molecule_drug_design/scorer_mcp/minimol_scorer_mcp",
    "modules/small_molecule_drug_design/scorer_mcp/antibiotics_scorer_mcp",
    "modules/small_molecule_drug_design/scorer_mcp/chemprop_scorers_mcp",
    "modules/small_molecule_drug_design/scorer_mcp/arthor_similarity_scorer_mcp",
    "modules/small_molecule_drug_design/scorer_mcp/local_similarity_scorer_mcp",
]

# You don't have to change anything below

module_names = set()
for scorer_path in mcp_scorer_paths:
    module_name = Path(scorer_path).name
    module_names.add(module_name)

if len(module_names) != len(mcp_scorer_paths):
    raise RuntimeError(
        "Duplicate module names found in mcp_scorer_paths. Please make sure each module has a unique name."
    )

for scorer_path in mcp_scorer_paths:
    register_mcp_module(scorer_path, serializer_name=serializer_name)


################################################################################
# ### Step 1.2: Set up Configurations
#
# 🔴 Please carefully check the description of each argument and set its value. The existing values are recommended but feel free to change them as required.
################################################################################


# Cell 23
# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run SAGA K. pneumoniae Antibiotics Design"
)
parser.add_argument(
    "--level",
    "-l",
    type=int,
    choices=[0, 1, 2, 3],
    default=0,
    help="Experiment level: 0 (1 iteration, both feedbacks), 1 (multiple iterations, both feedbacks), "
    "2 (multiple iterations, analyzer feedback only), 3 (multiple iterations, fully automated)",
)
parser.add_argument(
    "--enable-analog-mapping",
    action="store_true",
    help="Enable mapping to synthesizable analogs using Synformer",
)
parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed for reproducibility (default: 42)",
)
parser.add_argument(
    "--no-barebone-prompts",
    action="store_true",
    help="Disable barebone prompts (use detailed multi-objective prompts instead). Default: use barebone prompts (True)",
)
parser.add_argument(
    "--run-name",
    type=str,
    default=None,
    help="Custom run name for output folder. If not provided, auto-generated from other flags.",
)
args = parser.parse_args()

# Run ID
exp_level = args.level
enable_analog_mapping = args.enable_analog_mapping
seed = args.seed
use_barebone_prompts = not args.no_barebone_prompts
custom_run_name = args.run_name

# 🔴 Configs that you must change for your task
if custom_run_name:
    run_name = custom_run_name
else:
    run_name = "kp_antibiotics_001" + f"_level{exp_level}"
    if not enable_analog_mapping:
        run_name += "_no_analog"  # Could be your task name
    run_name += f"_seed{seed}"  # Include seed in run name

# Other configs
run_id = f"{run_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"

print(f"Running at Level {exp_level}")
print(f"Seed: {seed}")
print(f"run_id: {run_id}")

# Centralized location for human-readable GA logs
human_logger_dir = f"runs/{run_id}/human_logs"
Path(human_logger_dir).mkdir(parents=True, exist_ok=True)

DEFAULT_INIT_GROUP = "@modules/small_molecule_drug_design/data/molecules/Enamine_screening_collection_202510.smi"


# Cell 24
# Set up the level-specific configurations

if exp_level == 0:
    analyzer_enable_human_feedback = True
    planner_enable_human_feedback = True
elif exp_level == 1:
    analyzer_enable_human_feedback = True
    planner_enable_human_feedback = True
elif exp_level == 2:
    analyzer_enable_human_feedback = True
    planner_enable_human_feedback = (
        True  # Enable for first iteration only (handled below)
    )
    planner_enable_human_feedback_first_iteration_only = True
elif exp_level == 3:
    analyzer_enable_human_feedback = False
    planner_enable_human_feedback = False
else:
    raise ValueError("exp_level must be 0, 1, 2, or 3.")

# Initialize first iteration only flag for other levels
if exp_level != 2:
    planner_enable_human_feedback_first_iteration_only = False



# Cell 25
# Loop configs

# 🔴 Configs that you may want to change for your task
# Set max_iterations based on level
if exp_level == 0:
    max_iterations = 1  # Level 0: Single iteration only
else:
    max_iterations = 5  # Levels 1-3: 5 iterations
random_candidate_ratio = 1.0  # The ratio of candidates to be replaced before being fed into the optimizer every iteration (except iteration 1)

# Other configs
max_objective_planning_retries = 3  # The maximum number of retries for objective planning, if any planned objective cannot be implemented
return_all_candidates = True  # Whether to return and evaluate all candidates across all iterations in result.all_candidates_population
run_scorers_in_docker = True  # Whether to run scorer modules in Docker containers


# Cell 26
# Planner configs

# 🔴 Configs that you may want to change for your task
planner_model_name = "openai/gpt-5-2025-08-07"  # The LLM for the planner. Don't change the model, but you can change the provider
requires_objective_weights = (
    False  # Whether the planner needs to provide weights for each objective
)
support_filter = True  # Whether the planner should propose filter objectives
support_population_wise = (
    False  # Whether the planner should propose population-wise objectives
)
max_objectives = None  # The maximum number of objectives to propose for each iteration. None means no limit. Set it if your optimizer cannot handle too many objectives
do_high_level_planning = False  # Whether to do high-level planning before proposing objectives in the first iteration

# Other configs
planner_max_llm_retries = 3  # The maximum number of retries for LLM calls if one fails to get a valid objective plan
use_context_information = "first_iteration"  # The value can be "first_iteration", "all_iterations", "disabled". This controls whether to provide context information to the planner, and in which iterations. You can also set it to "all_iterations" if you want to provide the context information over and over again to the planner in all iterations


# Cell 27
# Scorer creator configs

# 🔴 Configs that you may want to change for your task
scorer_creator_model_name = "openai/gpt-5-2025-08-07"  # The LLM for the scorer creator for matching existing scorers, not for the coding agent. Don't change the model, but you can change the provider
enable_llm_scorer_creation = True  # Whether to enable the coding agent to create new scorers when no existing scorers can match the proposed objectives
coding_agent_model_name = "anthropic/claude-sonnet-4-5-20250929"  # Don't change the model, but you can change the provider to "anthropic" (Claude API), "bedrock" (AWS) or "claude_code" (your local Claude Code client, which can use your Claude Pro/Max account). Recommending not using "claude_code" to avoid unexpected limit exceeded issues.
reference_module_paths = mcp_scorer_paths  # The reference module paths to provide to the coding agent as examples. We provide all your registered MCP scorer modules here by default
use_potential_matched_scorers_as_references = True  # Whether to automatically use the potential matched existing scorers (judged by an LLM) as references even if they are not in the reference_module_paths (but included in the registered scorers mcp_scorer_paths)

# Other configs
coding_workspace_path = (
    f"runs/{run_id}/coding_workspace"  # The workspace path for the coding agent
)
generated_scorer_library_path = (
    f"runs/{run_id}/generated_scorers"  # The path to save the generated scorer modules
)
scorer_library_subfolder = None  # The subfolder under the generated_scorer_library_path to save the generated scorer modules. If None, the modules will be saved directly under the generated_scorer_library_path
scorer_creator_dev = False  # Whether to enable the developer mode for the scorer creator. If enabled, it would be more strict on exceptions and may raise exceptions instead of recovering from them
coding_agent_run_in_docker = True  # Whether to run the coding agent in a Docker container. By default, do it for better safety, but not do it if using your local Claude Code client because it doesn't support
scorer_creator_max_llm_retries = 3  # The maximum number of retries for LLM calls if one fails to get a valid response
coding_agent_max_parallel_scorer_creation = (
    1  # The maximum number of scorers that the coding agent can implement in parallel.
)
max_parallel_llm_matching = 2  # The maximum number of concurrent LLM calls for scorer matching (to avoid rate limits)
enable_name_matching = True  # Whether to enable the name-based matching
enable_llm_matching = (
    True  # Whether to enable the LLM-based matching by scorer descriptions
)


# Cell 28
# print available optimizers

pprint(list_registered_modules(module_type="optimizer")["optimizer"])


# Cell 29
# Optimizer configs

# 🔴 Configs that you must set for your task
optimizer_module_name = "llm_sbdd_optimizer"  # The name of the optimizer module to use. It has to be one of the registered optimizers (check the above printed list)
optimizer_module_version = "1.0.0"  # The version of the optimizer module to use. It has to be one of the registered versions of the selected optimizer (check the above printed list)
optimizer_config = {  # Put your optimizer-specific configs here
    # GA parameters (defaults shown in comments)
    # "population_size": 120,
    # "offspring_size": 70,
    # "mutation_size": 7,
    # "oracle_budget": 10000,
    # "tournament_size": 3,
    # Survival selection
    "survival_selection_method": "diverse_top",  # options: "fitness", "diverse_top", "butina_cluster"
    "elitism_fraction": 0.05,  # default: 0.025
    "elitism_fields": ["klebsiella_pneumoniae_minimol"],
    # Mutation mode
    "mutation_mode": "llm",  # "llm" or "non_llm" (GB-GA)
    "non_llm_mutation_rate": 0,
    # Initialization
    "seed": seed,
    "init_group": "enamine",
    # Logging
    "human_logger_output_dir": human_logger_dir,
    "human_logger_max_examples": 3,
    # Prompt style
    "use_barebone_prompts": use_barebone_prompts,  # Default: True (use short prompts)
}
optimizer_model_name = "openai/gpt-5-mini-2025-08-07"  # The LLM for the optimizer. Set None if your optimizer doesn't use any LLM.


# Cell 30
# Analyzer configs

# 🔴 Configs that you may want to change for your task
analyzer_model_name = "openai/gpt-5-2025-08-07"  # The LLM for the analyzer. Don't change the model, but you can change the provider
refusal_detection_model_name = "openai/gpt-4.1-nano-2025-04-14"  # The LLM for refusal detection. Don't change the model, but you can change the provider
candidate_analyzer_model_name = "anthropic/claude-sonnet-4-5-20250929"  # Don't change the model, but you can change the provider: claude_code, anthropic, bedrock
candidate_analyzer_run_in_docker = True  # Recommend to set True, because the docker environment installs many dependencies for analysis
candidate_analyzer_enable_domain_tools = (
    True  # Enable domain-specific tools for candidate analysis
)
candidate_analyzer_tool_selection_model = "anthropic/claude-sonnet-4-5-20250929" # Model for tool selection. Don't change the model, but you can change the provider.

# Other configs
population_save_dir = f"runs/{run_id}/populations_for_analysis"  # The directory to save the population data for each iteration
analyzer_max_llm_retries = 3  # The maximum number of retries for LLM calls if one fails to get a valid analysis
enable_candidate_analysis = True  # Whether to enable candidate-level analysis
candidate_analyzer_workspace = f"runs/{run_id}/candidate_analyzer_workspace"  # The workspace path for the candidate analyzer
enable_refusal_detection = True  # Whether to enable refusal detection in the analyzer
candidate_analyzer_tooluniverse_path = (
    "/opt/tooluniverse-env"
    if candidate_analyzer_run_in_docker
    else "./tooluniverse-env"
)  # Don't need to change


# Cell 31
# Create the configurations
# You don't need to change anything here. It just assembles all the configs above into a single config object.
config = create_config(
    run_id=run_id,
    run_name=run_name,
    loop_config={
        "max_iterations": max_iterations,
        "max_objective_planning_retries": max_objective_planning_retries,
        "random_candidate_ratio": random_candidate_ratio,
        "return_all_candidates": return_all_candidates,
        "run_scorers_in_docker": run_scorers_in_docker,
    },
    module_configs={
        "planner": {
            "config": {
                "requires_objective_weights": requires_objective_weights,
                "support_filter": support_filter,
                "support_population_wise": support_population_wise,
                "max_objectives": max_objectives,
                "do_high_level_planning": do_high_level_planning,
                "max_llm_retries": planner_max_llm_retries,
                "use_context_information": use_context_information,
                "enable_human_feedback": planner_enable_human_feedback,
                "enable_human_feedback_first_iteration_only": planner_enable_human_feedback_first_iteration_only,
            },
            "llm_config": {"model_name": planner_model_name},
        },
        "scorer_creator": {
            "config": {
                "enable_llm_scorer_creation": enable_llm_scorer_creation,
                "coding_agent_model_name": coding_agent_model_name,
                "reference_module_paths": reference_module_paths,
                "use_potential_matched_scorers_as_references": use_potential_matched_scorers_as_references,
                "coding_workspace_path": coding_workspace_path,
                "generated_scorer_library_path": generated_scorer_library_path,
                "scorer_library_subfolder": scorer_library_subfolder,
                "dev": scorer_creator_dev,
                "coding_agent_run_in_docker": coding_agent_run_in_docker,
                "max_llm_retries": scorer_creator_max_llm_retries,
                "max_parallel_llm_matching": max_parallel_llm_matching,
                "coding_agent_max_parallel_scorer_creation": coding_agent_max_parallel_scorer_creation,
                "enable_name_matching": enable_name_matching,
                "enable_llm_matching": enable_llm_matching,
            },
            "llm_config": {"model_name": scorer_creator_model_name},
        },
        "optimizer": {
            "module_name": optimizer_module_name,
            "module_version": optimizer_module_version,
            "config": optimizer_config,
            "llm_config": (
                {"model_name": optimizer_model_name}
                if optimizer_model_name is not None
                else None
            ),
        },
        "analyzer": {
            "config": {
                "analyzer_model_name": analyzer_model_name,
                "refusal_detection_model_name": refusal_detection_model_name,
                "candidate_analyzer_workspace": candidate_analyzer_workspace,
                "candidate_analyzer_model_name": candidate_analyzer_model_name,
                "candidate_analyzer_run_in_docker": candidate_analyzer_run_in_docker,
                "candidate_analyzer_enable_domain_tools": candidate_analyzer_enable_domain_tools,
                "candidate_analyzer_tool_selection_model": candidate_analyzer_tool_selection_model,
                "population_save_dir": population_save_dir,
                "analyzer_max_llm_retries": analyzer_max_llm_retries,
                "enable_candidate_analysis": enable_candidate_analysis,
                "enable_refusal_detection": enable_refusal_detection,
                "candidate_analyzer_tooluniverse_path": candidate_analyzer_tooluniverse_path,
                "enable_human_feedback": analyzer_enable_human_feedback,
            },
            "llm_config": {"model_name": analyzer_model_name},
        },
    },
)


# Cell 32
# Print all the configurations and let's check if everything is correct
print(config)


# Cell 33
# Create the orchestrator
orchestrator = OptimizationOrchestrator(
    config,
    run_name=run_name,
    run_id=run_id,
)



################################################################################
# ## Part 2: Run SAGA
################################################################################


################################################################################
# ### Step 2.1: Prepare Your Inputs
################################################################################


# Cell 36
# 🔴 Prepare your inputs

# Clearly describe the optimization goal
optimization_goal = "Design novel antibiotic small molecules that are highly effective against Klebsiella pneumoniae bacteria while maintaining good safety profiles and drug-like properties."

# Provide additional context information to help the planner better understand your task and propose better objectives
# Such as the task background, the constraints, and your initial ideas about the objectives
# No need to mention that you'll provide an initial set of objectives, because the planner will always get them from the user in the first iteration
# You don't have to necessarily follow the below example. You can provide any information that you think is helpful for the planner to understand your task and propose better objectives
context_information = """For this task, we want to design novel antibiotics targeting K. pneumoniae bacteria.
The molecules should:
1. Show high predicted activity against K. pneumoniae.
2. Maintain low toxicity to human cells
3. Avoid problematic substructures for medicinal chemistry
4. Show structural novelty compared to existing antibiotics
5. Have good drug-like properties and molecular weight for small molecule drug design
6. Be purchasable from Enamine Real Space (we want to purchase molecules from Enamine Real Space)

The optimizer will automatically enforce SMILES validity and length constraints, so do not propose objectives related to these.

IMPORTANT SCORER REQUIREMENTS:
- For candidate-wise objectives: Scores must be normalized to [0, 1] range, where higher values are better (maximization direction).
- For filter objectives: Scores must return 1.0 for pass and 0.0 for fail. Filters do not need normalization or inversion when multiplied into aggregated scores.
"""



# Cell 38
# Print out the registered objectives

print("Registered scorers:")
for objective_name in list_scorers():
    metadata = get_scorer_metadata(objective_name)
    print(f"- {objective_name}: {metadata['description']}")


# Cell 39
# Specify initial objectives

# 🔴 Put the names of the initial objectives and their optimization directions here
# They have to be among the registered scorers printed above
# For filter objectives (if any), the optimization direction should always be None
initial_objective_names_and_optimization_directions = {
    "klebsiella_pneumoniae_minimol": "maximize",
    "antibiotics_novelty": "maximize",
    "toxicity_safety_chemprop": "maximize",
    "antibiotics_motifs_filter": None,  # Filter objective - optimization direction is None
    "local_similarity": "maximize",
}

# You don't need to change anything below. It checks the validity of the initial objectives and creates Objective instances for them.

# Make sure that the corresponding MCP modules do not contain other objectives than the initial ones
scorer_manager = ScorerManager()
module_set = set()
for objective_name in initial_objective_names_and_optimization_directions:
    module_name = scorer_manager.mcp_scorer_to_module[objective_name]
    module_set.add(module_name)
module_scorers_set = set()
for module_name in module_set:
    module_scorers = scorer_manager.mcp_module_to_scorers[module_name]
    for scorer in module_scorers:
        module_scorers_set.add(scorer)
        print(scorer)


# Check the initial objectives' types
type_set = set()
for objective_name in initial_objective_names_and_optimization_directions:
    metadata = get_scorer_metadata(objective_name)
    type_set.add(metadata["type"])

if not support_filter and "filter" in type_set:
    raise RuntimeError(
        "One or more initial objectives are of type 'filter', but the planner's `support_filter` is set to `False`."
    )
if not support_population_wise and "population_wise" in type_set:
    raise RuntimeError(
        "One or more initial objectives are of type 'population_wise', but the planner's `support_population_wise` is set to `False`."
    )

# Create Objective instances for the initial objectives

initial_objectives = []
for (
    initial_objective_name,
    optimization_direction,
) in initial_objective_names_and_optimization_directions.items():
    scorer = get_scorer(initial_objective_name)
    if scorer is None:
        raise RuntimeError(
            f"Initial objective scorer '{initial_objective_name}' not found. Registered scorers: {list_scorers()}."
        )
    metadata = get_scorer_metadata(initial_objective_name)
    objective = Objective(
        name=initial_objective_name,
        description=metadata["description"],
        type=metadata["type"],
        scorer=scorer,
        optimization_direction=optimization_direction,
    )
    initial_objectives.append(objective)


# Cell 40
# 🔴 Create initial population in some way
# Change it to your own way


def process_enamine_neighbors(run_id: str):
    """
    Process all iteration_*/per_run directories with the Enamine neighbor finding script.
    Runs after all iterations complete successfully.
    """
    run_dir = Path(f"runs/{run_id}/logs")
    if not run_dir.exists():
        logger.warning(f"Run directory not found: {run_dir}")
        return

    # Find all iteration_*/per_run directories
    per_run_dirs = sorted(run_dir.glob("iteration_*/per_run"))
    if not per_run_dirs:
        logger.info(f"No iteration_*/per_run directories found in {run_dir}")
        return

    logger.info(f"Found {len(per_run_dirs)} iteration per_run directories to process")
    for per_run_dir in per_run_dirs:
        logger.info(f"  - {per_run_dir}")

    # Build command to run the script
    script_path = (
        Path(__file__).parent.parent.parent / "scripts" / "process_per_run_enamine.py"
    )
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        return

    # Prepare command with all per_run directories (quote paths to handle spaces)
    per_run_paths_str = " ".join(f'"{str(d)}"' for d in per_run_dirs)
    cmd = [
        "bash",
        "-lc",
        f'eval "$(conda shell.bash hook)" && conda activate genesis && python "{script_path}" {per_run_paths_str}',
    ]

    logger.info(
        f"Running Enamine neighbor processing script on {len(per_run_dirs)} directories..."
    )
    try:
        result = subprocess.run(
            cmd,
            cwd=Path(__file__).parent.parent.parent,
            capture_output=True,
            text=True,
            check=False,  # Don't raise on error, we'll handle it
        )

        if result.returncode == 0:
            logger.info("✓ Enamine neighbor processing completed successfully")
            if result.stdout:
                logger.debug(f"Script output:\n{result.stdout}")
        else:
            logger.warning(
                f"Enamine neighbor processing completed with errors (return code: {result.returncode})"
            )
            if result.stderr:
                logger.warning(f"Script errors:\n{result.stderr}")
    except Exception as e:
        logger.error(
            f"Failed to run Enamine neighbor processing script: {e}", exc_info=True
        )


async def main():
    optimizer = orchestrator.optimizer
    initial_population = Population(
        candidates=await optimizer.create_random_candidates(
            120, init_group=DEFAULT_INIT_GROUP
        )
    )

    # Cell 41
    # Boom! Run!
    try:
        result = await orchestrator.run(
            optimization_goal,
            context_information,
            serializer_name=serializer_name,
            initial_objectives=initial_objectives,
            initial_population=initial_population,
        )

        # Process Enamine neighbors after all iterations complete successfully
        if result.is_successful:
            logger.info("=" * 80)
            logger.info("Post-processing: Finding Enamine neighbors for all iterations")
            logger.info("=" * 80)
            process_enamine_neighbors(run_id)
        else:
            logger.info(
                f"Skipping Enamine neighbor processing (experiment status: {result.status})"
            )
        logger.info(
            "Human-friendly GA logs (CSV, Markdown, JSON) are available at %s",
            human_logger_dir,
        )
    finally:
        # Make sure MCP scorers' docker containers are properly cleaned up
        reset_scorer_manager()
        print_iteration_stats(orchestrator)

    return result


def print_iteration_stats(orch):
    """Print per-iteration statistics in a human-friendly format.

    Accepts the orchestrator directly so it can be called from a finally block
    even if the run was interrupted before `result` was returned.
    """
    result = orch.result
    if result is None:
        print("\n(No result available — optimization did not start.)")
        return

    print(f"\n{'=' * 70}")
    print("OPTIMIZATION RESULTS — Per-Iteration Summary")
    print(f"{'=' * 70}")
    print(f"  Run ID    : {result.run_id}")
    print(f"  Status    : {result.status}")
    print(f"  Iterations: {result.total_generations}")
    if result.termination_reason:
        print(f"  Termination: {result.termination_reason}")
    print()

    for i in range(0, result.total_generations + 1):
        population = orch.knowledge_manager.get_population(i)
        if population is None or population.is_empty:
            continue

        label = "Iteration 0 (Initial Population)" if i == 0 else f"Iteration {i}"
        print(f"{'─' * 70}")
        print(f"  {label}")
        print(f"{'─' * 70}")
        print(f"  Candidates : {population.size}")

        objective_names = sorted({
            key
            for candidate in population.candidates
            for key in candidate.scores.keys()
        })

        if objective_names:
            print("  Scores:")
            col_w = max(len(n) for n in objective_names)
            for obj_name in objective_names:
                mean, std, none_count = population.get_regular_score_mean_and_std(obj_name)
                mean_str = f"{mean:>10.4f}" if mean is not None else "       N/A"
                std_str  = f"{std:>9.4f}"   if std  is not None else "      N/A"
                valid    = population.size - none_count
                print(f"    {obj_name:<{col_w}}  mean={mean_str}  std={std_str}  "
                      f"valid={valid}/{population.size}  missing={none_count}")
        else:
            print("  (no scores available)")
        print()

    print(f"{'=' * 70}")
    if result.final_population:
        print(f"  Final population size : {result.final_population.size}")
    if result.all_candidates_population:
        print(f"  All candidates (all iterations) : {result.all_candidates_population.size}")
    print(f"{'=' * 70}\n")


if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"Starting SAGA Optimization - Level {exp_level}")
    print(f"{'='*80}\n")
    result = asyncio.run(main())
