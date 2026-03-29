#!/usr/bin/env python3
"""
Simple script for running molecular optimization with UniDock scoring functions.
"""

import argparse
import logging
import sys
import warnings
import os
from typing import List
from datetime import datetime

import os
import pandas as pd
module_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(module_dir)  # modules/
grand_dir = os.path.dirname(base_dir)# SciLeiAgent
print(grand_dir)
import sys
sys.path.insert(0, grand_dir)
sys.path.insert(0, grand_dir+"/scileo_agent")

from scileo_agent.core.config import LLMConfig
from scileo_agent.core.data_models import Objective
from scileo_agent.core.registry import get_scorer
from modules.small_molecule_drug_design.llm_sbdd_optimizer import LLMSBDDOptimizer
from modules.small_molecule_drug_design.scorer.unidock_scorer import UniDockScorers
from modules.small_molecule_drug_design.scorer.druglikeness_scorer import DruglikenessScorers
from modules.small_molecule_drug_design.scorer.ra_scorer import RAScorers
# Import module to trigger scorer registration on import
import modules.small_molecule_drug_design.scorer.antibiotics_scorer  # noqa: F401
from modules.small_molecule_drug_design.ga_logging import GALogger, ChemistLogger
from modules.small_molecule_drug_design.utils.rdkit_utils import calculate_population_diversity
from modules.small_molecule_drug_design.postprocessing.aggregate_selection import (
    aggregate_and_select_folder,
)
from scileo_agent.core.data_models import Population, Objective, Candidate, ObjectiveIndex

def configure_logging():
    """Configure logging to reduce noisy INFO messages and suppress warnings."""
    # Set root logger to WARNING
    logging.getLogger().setLevel(logging.WARNING)

    # Specifically quiet these noisy loggers
    noisy_loggers = [
        "LiteLLM", "litellm", "openai", "httpx", "httpcore", "MDAnalysis",
        "MDAnalysis.coordinates", "MDAnalysis.topology", "MDAnalysis.universe",
        "MDAnalysis.topology.base"
    ]

    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Suppress specific MDAnalysis warnings
    warnings.filterwarnings("ignore",
                            message="Unit cell dimensions not found.*")
    warnings.filterwarnings("ignore",
                            message="Found no information for attr.*")
    warnings.filterwarnings("ignore", message=".*CRYST1 record.*")


# Available scoring functions
AVAILABLE_TARGETS = {
    'ampcclean':'ampcclean_unidock',
    'muopioidclean':'muopioidclean_unidock',
    'mpro': 'mpro_unidock',
    'mars1': 'mars1_unidock',
    'qed': 'qed',
    'logp': 'logp_score',
    'deepdl': 'deepdl_druglikeness',
    'toxicity': 'toxicity_safety_chemprop',
    'staph_aureus': 'staph_aureus_chemprop',
    'pains': 'pains_filter',
    'brenk': 'brenk_filter',
    'ra': 'ra_score_xgb',
    'antibiotics_novelty': 'antibiotics_novelty',
}


def create_optimizer(config: dict = None,init_group=None) -> LLMSBDDOptimizer:
    """Create and configure the optimizer."""
    default_config = {
        "population_size": 120,
        "offspring_size": 70,
        "mutation_size": 30,
        "oracle_budget": 10000,
        "seed": 42
    }

    if config:
        default_config.update(config)

    # LLM configuration
    llm_config = LLMConfig(provider="openai", model="gpt-4o-mini")
    return LLMSBDDOptimizer(module_id="llm_sbdd_optimizer",
                            config=default_config,
                            llm_config=llm_config,init_group=init_group)


def create_objective(target: str) -> Objective:
    """Create optimization objective for the specified target."""
    scorer = get_scorer(AVAILABLE_TARGETS[target])

    objective = Objective(
        name=scorer._scorer_name,
        description=scorer._scorer_metadata['description'],
        optimization_direction="maximize",
        population_wise=scorer._scorer_metadata['population_wise'],
        scorer=scorer)

    return objective


def main():
    # Configure logging early to reduce noise
    configure_logging()

    parser = argparse.ArgumentParser(
        description="Run molecular optimization with UniDock scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available targets:
  {', '.join(AVAILABLE_TARGETS.keys())}

Example usage:
  python run_optimization.py --target drd2 --budget 200
  python run_optimization.py --target brd4 --population-size 50 --budget 500
  python run_optimization.py --targets drd2 qed logp --budget 500  # Multi-objective
  python run_optimization.py --target drd2 --debug  # Quick test with small parameters
  python run_optimization.py --target drd2 --output-dir results/  # Custom output directory
        """)

    # Single target (backward compatible)
    parser.add_argument(
        "--target",
        choices=list(AVAILABLE_TARGETS.keys()),
        help=
        "Single target for optimization (use --targets for multi-objective)")
    # Multi-objective: space-separated list of targets
    parser.add_argument(
        "--targets",
        nargs='+',
        choices=list(AVAILABLE_TARGETS.keys()),
        help="Space-separated targets for multi-objective optimization")

    parser.add_argument(
        "--budget",
        type=int,
        default=10000,
        help="Oracle budget (number of evaluations) (default: 10000)")

    parser.add_argument("--population-size",
                        type=int,
                        default=120,
                        help="Population size (default: 120)")

    parser.add_argument(
        "--offspring-size",
        type=int,
        default=70,
        help="Number of offspring per generation (default: 70)")

    parser.add_argument(
        "--mutation-size",
        type=int,
        default=30,
        help="Number of mutations per generation (default: 30)")

    parser.add_argument("--seed",
                        type=int,
                        default=42,
                        help="Random seed (default: 42)")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with smaller population/budget for testing")

    parser.add_argument(
        "--output-dir",
        type=str,
        default="logs",
        help="Output directory for logs and plots (default: logs)")
    
    parser.add_argument(
        "--name",
        dest="experiment_name",
        type=str,
        help="Optional custom experiment name for the subfolder; date (YYYYMMDD_HHMMSS) will be prepended"
    )

    # Optional weights for multi-objective runs
    parser.add_argument(
        "--weights",
        type=str,
        help=
        "Comma-separated weights corresponding to the order of --targets (optional)"
    )

    parser.add_argument(
        "--survival-selection-method",
        type=str,
        choices=["fitness", "diverse_top"],
        default="fitness",
        help="Survivor selection method: fitness (default) or diverse_top",
    )

    # Elitism controls
    elitism_group = parser.add_mutually_exclusive_group()
    elitism_group.add_argument(
        "--elitism-fraction",
        type=float,
        help="Fraction of current population carried over as elites (0-1). Defaults to 0.10 in the optimizer",
    )
    elitism_group.add_argument(
        "--no-elitism",
        action="store_true",
        help="Disable elitism (equivalent to --elitism-fraction 0.0)",
    )

    # Objective combiner selection
    parser.add_argument(
        "--combiner",
        type=str,
        default="simple_product",
        choices=["simple_sum", "simple_product", "weighted_sum", "antibiotic_geomean", "covid_simple"],
        help="Objective combiner for multi-objective aggregation (default: simple_product - geometric mean)",
    )
    
    parser.add_argument(
        "--init_group",
        type=str,
        default="diverse_10k",
        help=
        "Select the initial group"
    )

    parser.add_argument(
        "--enable-structure-filter",
        action="store_true",
        help="Enable structure filtering during optimization"
    )

    # Optional postprocessing: aggregate and select after optimization
    parser.add_argument(
        "--post-aggregate",
        action="store_true",
        default=True,
        help="After optimization completes, aggregate CSV logs in the experiment folder and select top-diverse",
    )
    parser.add_argument(
        "--post-aggregate-pattern",
        type=str,
        default="**/*selected.csv",
        help="Glob pattern within experiment folder for CSVs to include",
    )
    parser.add_argument(
        "--post-k",
        type=int,
        default=1000,
        help="Number of molecules to select in post-aggregation",
    )
    parser.add_argument(
        "--post-tanimoto-threshold",
        type=float,
        default=0.4,
        help="Tanimoto threshold for post-aggregation selection",
    )
    parser.add_argument(
        "--post-leniency",
        type=int,
        default=0,
        help="Leniency for post-aggregation selection",
    )

    args = parser.parse_args()

    # Override parameters for debug mode
    if args.debug:
        print("🔧 DEBUG MODE ENABLED - Using smaller parameters for testing")
        args.budget = min(args.budget, 100)
        args.population_size = min(args.population_size, 25)
        args.offspring_size = min(args.offspring_size, 15)
        args.mutation_size = min(args.mutation_size, 5)

    # No selection-only branch here anymore. Selection lives in a dedicated script.

    print("=" * 60)
    print("         MOLECULAR OPTIMIZATION")
    if args.debug:
        print("              (DEBUG MODE)")
    print("=" * 60)
    selected_targets = args.targets if args.targets else (
        [args.target] if args.target else [])
    if not selected_targets:
        raise ValueError("You must specify either --target or --targets")
    if len(selected_targets) == 1:
        print(
            f"Target: {selected_targets[0].upper()} ({AVAILABLE_TARGETS[selected_targets[0]]})"
        )
    else:
        readable = ", ".join(f"{t.upper()} ({AVAILABLE_TARGETS[t]})"
                             for t in selected_targets)
        print(f"Targets: {readable}")
    print(
        f"Budget: {args.budget} evaluations{' (debug)' if args.debug else ''}")
    print(
        f"Population size: {args.population_size}{' (debug)' if args.debug else ''}"
    )
    print(
        f"Offspring size: {args.offspring_size}{' (debug)' if args.debug else ''}"
    )
    print(
        f"Mutation size: {args.mutation_size}{' (debug)' if args.debug else ''}"
    )
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Create optimizer
    config = {
        "population_size": args.population_size,
        "offspring_size": args.offspring_size,
        "mutation_size": args.mutation_size,
        "oracle_budget": args.budget,
        "seed": args.seed,
        "survival_selection_method": args.survival_selection_method,
        "objective_combiner": args.combiner,
        "enable_structure_filter": args.enable_structure_filter,
    }

    # Apply elitism config if specified
    if args.no_elitism:
        config["elitism_fraction"] = 0.0
    elif args.elitism_fraction is not None:
        config["elitism_fraction"] = args.elitism_fraction

    print("\nInitializing optimizer...")
    optimizer = create_optimizer(config, init_group=args.init_group)

    # Create objectives
    if len(selected_targets) == 1:
        print(f"Setting up {selected_targets[0].upper()} objective...")
    else:
        print(
            f"Setting up multi-objective: {', '.join(t.upper() for t in selected_targets)}"
        )
    objectives: List[Objective] = [
        create_objective(t) for t in selected_targets
    ]

    # Optional manual weights
    if args.weights:
        weights = [
            float(w) for w in args.weights.split(',') if w.strip() != ''
        ]
        if len(weights) != len(objectives):
            raise ValueError(
                "Number of --weights must match number of objectives")
        optimizer.set_objectives_weights(weights)
        print(f"Objective weights: {weights}")

    # Validate objectives
    optimizer.check_objectives(objectives)
    print("✓ Objective validation passed")
    optimizer.objectives_weights = [1.0 for _ in objectives]
        
    df_new = pd.read_csv("/gpfs/radev/home/tl688/pitl688/scileoagent_drug/large_scale_molecule.csv")
    mole_data = df_new['smiles'].values
    candidates = [
    Candidate(representation=optimizer._sanitize_smiles_value(smiles))
    for smiles in mole_data
    ]
    from tqdm import tqdm
    popu_data = Population(candidates=candidates)
    score_list = []
    for item in tqdm(popu_data):
        score = optimizer._compute_candidate_score(item, objectives)
        score_list.append(score)
    pd.DataFrame(score_list).to_csv("allscore.csv")
    
if __name__ == "__main__":
    main()
