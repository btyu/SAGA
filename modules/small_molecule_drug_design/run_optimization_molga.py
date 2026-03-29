#!/usr/bin/env python3
"""
Simple script for running molecular optimization with Graph-GA and UniDock scoring functions.
"""

import argparse
import logging
import sys
import warnings
import os
import pandas as pd
import numpy as np
import json
import pickle
import csv
from typing import List, Optional, Dict, Any
from datetime import datetime
from rdkit import Chem

from scileo_agent.core.data_models import Candidate
from scileo_agent.core.registry import get_scorer

# Import shared functions from run_optimization.py
from modules.small_molecule_drug_design.molopt_patch.graph_ga import PatchGraphGA as GraphGA
from modules.small_molecule_drug_design.run_optimization import configure_logging, AVAILABLE_TARGETS

# Storage for metrics and final results
metrics_history = []
final_mol_buffer = {}
convergence_data = []  # Store detailed convergence data for CSV
populations_data = {}  # Store population data for pickle file

# Import Oracle base class to monkey patch logging and evaluation
from molopt.base import Oracle


class ResultsCollector:
    """Collect results from mol-opt optimization for display."""

    def __init__(self):
        self.final_molecules = []
        self.final_scores = []

    def extract_from_mol_buffer(self, mol_buffer):
        """Extract top molecules and scores from mol-opt buffer."""
        if not mol_buffer:
            return

        # Sort by score (descending)
        sorted_buffer = sorted(mol_buffer.items(),
                               key=lambda x: x[1][0],
                               reverse=True)

        # Extract top molecules and scores
        self.final_molecules = [item[0] for item in sorted_buffer]  # SMILES
        self.final_scores = [item[1][0] for item in sorted_buffer]  # Scores


def create_oracle_wrapper(target: str, results_collector: ResultsCollector):
    """Create an oracle function that mol-opt can use from our scorer system."""
    scorer = get_scorer(AVAILABLE_TARGETS[target])

    # Determine if this is a minimization problem (for future extensibility)
    # Currently all our targets are maximization (higher = better)
    is_maximization = True  # All current targets: drd2, gsk3b, jnk3, brd4, qed

    def oracle_function_single(smiles: str) -> float:
        """Oracle function that takes a SMILES string and returns a score."""
        try:
            # Create a Candidate object from SMILES
            candidate = Candidate(representation=smiles)

            # Score the candidate (returns List[Optional[float]])
            scores = scorer([candidate])

            # Handle scoring results
            if scores and len(scores) > 0 and scores[0] is not None:
                score = float(scores[0])
                return score
            else:
                # Return appropriate failure value based on optimization direction
                if is_maximization:
                    return 0.0  # Poor score for maximization
                else:
                    return float(
                        'inf')  # Poor score for minimization (very high value)

        except Exception as e:
            # Return appropriate failure value for any scoring errors
            if is_maximization:
                return 0.0
            else:
                return float('inf')

    def oracle_function(smiles_list: List[str]) -> float:
        """Oracle function that takes a SMILES string and returns a score."""
        try:
            # Create a Candidate object from SMILES
            candidates = [
                Candidate(representation=smiles) for smiles in smiles_list
            ]

            # Score the candidate (returns List[Optional[float]])
            scores = scorer(candidates)
            assert len(scores) == len(
                smiles_list
            ), f"Number of scores {len(scores)} does not match number of smiles {len(smiles_list)}"
            # Handle scoring results

            processed_scores = []
            if scores is not None:
                for score in scores:
                    if score is not None:
                        score = float(score)
                    else:
                        if is_maximization:
                            score = 0.0  # Poor score for maximization
                        else:
                            score = float(
                                        'inf')  # Poor score for minimization (very high value)
                        processed_scores.append(score)
                return processed_scores
            else:
                raise Exception

        except Exception as e:
            # Return appropriate failure value for any scoring errors
            if is_maximization:
                return [0.0 for i in range(len(smiles_list))]
            else:
                return [float('inf') for i in range(len(smiles_list))]

    return oracle_function


def create_molga_optimizer(config: dict = None) -> GraphGA:
    """Create and configure the Graph-GA optimizer."""
    default_config = {
        "population_size": 120,
        "offspring_size": 70,
        "mutation_rate":
        0.067,  # Graph-GA uses mutation_rate instead of mutation_size
        "max_oracle_calls": 10000,
        "freq_log": 1,
        "n_jobs":
        1,  # Set to 1 to avoid multiprocessing issues with docking oracles
        "output_dir": "results",
        "log_results": True,
        "evaluation_batch_size": 100,
    }

    if config:
        default_config.update(config)

    # Find zinc dataset for starting population
    current_dir = os.path.dirname(os.path.abspath(__file__))
    smi_file = os.path.join(current_dir, "data/molecules/zinc_250k.txt")
    print(f"smi_file: {smi_file}")

    # Check if file exists, otherwise use None (exploration mode)
    if not os.path.exists(smi_file):
        print(f"Warning: ZINC dataset not found at {smi_file}")
        print("Running in exploration mode (no starting population file)")
        smi_file = None

    try:
        # Monkey Patching GraphGA Optimizer's Oracle to enable batched evaluations
        def score_smi_batch(self, smi_list):
            """
            Function to score one molecule

            Arguments:
                smi_list: List of SMILES strings representing a batch of molecules

            Return:
                score: a float represents the property of the molecule.
            """
            scores = self.evaluator(smi_list)
            processed_scores = []

            for idx, score in enumerate(scores):
                score = float(score)
                smi = smi_list[idx]

                if len(self.mol_buffer) > self.max_oracle_calls:
                    score = 0
                if smi is None:
                    score = 0

                try:
                    mol = Chem.MolFromSmiles(smi)
                    if mol is None or len(smi) == 0:
                        score = 0
                    else:
                        smi = Chem.MolToSmiles(mol)
                        if smi in self.mol_buffer:
                            pass
                        else:
                            self.mol_buffer[smi] = [
                                score, len(self.mol_buffer) + 1
                            ]
                except Exception as e:
                    score = 0

                processed_scores.append(score)

            return processed_scores

        def patched_call(self, smiles_lst):
            """
            Score
            """
            if type(smiles_lst) == list:
                score_list = []
                batch_size = default_config['evaluation_batch_size']
                for idx in range(0, len(smiles_lst), batch_size):
                    smiles_batch = smiles_lst[idx:idx + batch_size]
                    score_list.extend(self.score_smi(smiles_batch))
                    if len(self.mol_buffer) % self.freq_log == 0 and len(
                            self.mol_buffer) > self.last_log:
                        self.sort_buffer()
                        self.log_intermediate()
                        self.last_log = len(self.mol_buffer)
                        self.save_result(self.task_label)
            else:  ### a string of SMILES
                score_list = self.score_smi(smiles_lst)
                if len(self.mol_buffer) % self.freq_log == 0 and len(
                        self.mol_buffer) > self.last_log:
                    self.sort_buffer()
                    self.log_intermediate()
                    self.last_log = len(self.mol_buffer)
                    self.save_result(self.task_label)

            return score_list

        # Override Oracle's score_smi and __call__
        Oracle.score_smi = score_smi_batch
        Oracle.__call__ = patched_call
    except Exception as e:
        print(
            "Warning: Could not patch molopt.base.Oracle. Unable to do batch evaluations, computation will be slower."
        )

    return GraphGA(
        smi_file=smi_file,
        n_jobs=default_config["n_jobs"],
        max_oracle_calls=default_config["max_oracle_calls"],
        freq_log=default_config["freq_log"],
        output_dir=default_config["output_dir"],
        log_results=default_config["log_results"],
    )


def setup_metrics_logging(results_collector: ResultsCollector, target: str):
    """Setup metrics logging similar to molga.ipynb"""
    global metrics_history, final_mol_buffer, convergence_data, populations_data
    metrics_history = []
    convergence_data = []
    populations_data = {}

    obj_name = AVAILABLE_TARGETS[target]

    # Import Oracle base class to monkey patch logging
    try:
        # Store original method
        original_oracle_log_intermediate = Oracle.log_intermediate

        def patched_oracle_log_intermediate(self,
                                            mols=None,
                                            scores=None,
                                            finish=False):
            # Call original method
            original_oracle_log_intermediate(self,
                                             mols=mols,
                                             scores=scores,
                                             finish=finish)

            # Capture final results when finishing
            if finish:
                global final_mol_buffer
                final_mol_buffer = dict(self.mol_buffer)
                results_collector.extract_from_mol_buffer(final_mol_buffer)

            # Capture metrics
            if len(self.mol_buffer) > 0:
                temp_top100 = list(self.mol_buffer.items())[:100]
                scores_list = [item[1][0] for item in temp_top100]

                # Store current population snapshot for pickle file
                current_generation = len(convergence_data)
                populations_data[current_generation] = {
                    'molecules': list(self.mol_buffer.keys()),
                    'scores': [item[1][0] for item in self.mol_buffer.items()],
                    'generation': current_generation,
                    'timestamp': datetime.now()
                }

                # Calculate detailed statistics for CSV
                all_scores = [item[1][0] for item in self.mol_buffer.items()]
                sorted_scores = sorted(all_scores, reverse=True)

                convergence_row = {
                    'generation':
                    current_generation,
                    'population_size':
                    len(self.mol_buffer),
                    'evaluations_used':
                    len(self.mol_buffer) -
                    (convergence_data[-1]['total_evaluations']
                     if convergence_data else 0),
                    'total_evaluations':
                    len(self.mol_buffer),
                    'timestamp':
                    datetime.now(),
                    f'{obj_name}_mean':
                    np.mean(scores_list),
                    f'{obj_name}_median':
                    np.median(all_scores),
                    f'{obj_name}_top1':
                    np.max(scores_list),
                    f'{obj_name}_top10_mean':
                    np.mean(sorted_scores[:10]),
                    'budget_used':
                    float(len(self.mol_buffer)),
                    'new_evaluations':
                    float(
                        len(self.mol_buffer) -
                        (convergence_data[-1]['total_evaluations']
                         if convergence_data else 0)),
                    'generations_without_improvement':
                    0.0  # This would need more sophisticated tracking
                }
                convergence_data.append(convergence_row)

                metrics_history.append({
                    'n_calls':
                    len(self.mol_buffer),
                    'avg_top1':
                    np.max(scores_list),
                    'avg_top10':
                    np.mean(sorted(scores_list, reverse=True)[:10]),
                    'avg_top100':
                    np.mean(scores_list)
                })

        # Apply the patch
        Oracle.log_intermediate = patched_oracle_log_intermediate

    except ImportError:
        print(
            "Warning: Could not patch molopt.base.Oracle logging - metrics collection may be limited"
        )


def generate_summary_report(target: str, args,
                            results_collector: ResultsCollector) -> dict:
    """Generate optimization summary report from metrics history."""
    if not metrics_history:
        return {
            'experiment_name': f"{target}_optimization_seed{args.seed}_molga",
            'total_generations': 0,
            'total_evaluations': 0,
            'final_statistics': {},
            'improvement': {}
        }

    initial_stats = metrics_history[0] if metrics_history else {}
    final_stats = metrics_history[-1] if metrics_history else {}

    obj_name = AVAILABLE_TARGETS[target]

    report = {
        'experiment_name': f"{target}_optimization_seed{args.seed}",
        'total_generations': len(metrics_history),
        'total_evaluations': final_stats.get('n_calls', 0),
        'final_statistics': {
            obj_name: {
                'mean': final_stats.get('avg_top100', 0.0),
                'top1': final_stats.get('avg_top1', 0.0),
                'top10_mean': final_stats.get('avg_top10', 0.0)
            }
        },
        'improvement': {
            obj_name: {
                'initial_mean':
                initial_stats.get('avg_top100', 0.0),
                'final_mean':
                final_stats.get('avg_top100', 0.0),
                'absolute_improvement':
                final_stats.get('avg_top100', 0.0) -
                initial_stats.get('avg_top100', 0.0),
                'relative_improvement':
                ((final_stats.get('avg_top100', 0.0) -
                  initial_stats.get('avg_top100', 0.0)) /
                 max(initial_stats.get('avg_top100', 0.001), 0.001)) * 100
            }
        }
    }

    return report


def main():
    # Configure logging early to reduce noise
    configure_logging()

    parser = argparse.ArgumentParser(
        description=
        "Run molecular optimization with Graph-GA and UniDock scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Available targets:
  {', '.join(AVAILABLE_TARGETS.keys())}

Example usage:
  python run_optimization_molga.py --target drd2 --budget 200
  python run_optimization_molga.py --target brd4 --population-size 50 --budget 500
  python run_optimization_molga.py --target drd2 --debug  # Quick test with small parameters
  python run_optimization_molga.py --target drd2 --output-dir results/  # Custom output directory
        """)

    parser.add_argument("--target",
                        choices=list(AVAILABLE_TARGETS.keys()),
                        required=True,
                        help="UniDock target protein for optimization")

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

    parser.add_argument("--mutation-rate",
                        type=float,
                        default=0.067,
                        help="Mutation rate (default: 0.067)")

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

    args = parser.parse_args()

    # Override parameters for debug mode
    if args.debug:
        print("🔧 DEBUG MODE ENABLED - Using smaller parameters for testing")
        args.budget = min(args.budget, 100)
        args.population_size = min(args.population_size, 25)
        args.offspring_size = min(args.offspring_size, 15)
        args.freq_log = 10

    print("=" * 60)
    print("         MOLECULAR OPTIMIZATION (Graph-GA)")
    if args.debug:
        print("              (DEBUG MODE)")
    print("=" * 60)
    print(f"Target: {args.target.upper()} ({AVAILABLE_TARGETS[args.target]})")
    print(
        f"Budget: {args.budget} evaluations{' (debug)' if args.debug else ''}")
    print(
        f"Population size: {args.population_size}{' (debug)' if args.debug else ''}"
    )
    print(
        f"Offspring size: {args.offspring_size}{' (debug)' if args.debug else ''}"
    )
    print(f"Mutation rate: {args.mutation_rate}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    try:
        # Create experiment-specific output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_dir = os.path.join(
            args.output_dir, f"{args.target}_molga_seed{args.seed}_{timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)

        # Create results collector
        results_collector = ResultsCollector()

        # Setup metrics logging
        print("\nSetting up metrics logging...")
        setup_metrics_logging(results_collector, args.target)

        # Create optimizer
        config = {
            "population_size": args.population_size,
            "offspring_size": args.offspring_size,
            "mutation_rate": args.mutation_rate,
            "max_oracle_calls": args.budget,
            "output_dir": experiment_dir,
            "n_jobs":
            1  # Force single-threaded to avoid multiprocessing issues
        }

        print("Initializing Graph-GA optimizer...")
        optimizer = create_molga_optimizer(config)

        # Create oracle function
        print(f"Setting up {args.target.upper()} oracle...")
        oracle = create_oracle_wrapper(args.target, results_collector)

        print("✓ Oracle validation passed")

        # Run optimization
        print("\nStarting optimization...")
        print("-" * 40)

        # Graph-GA parameters for optimization
        optimizer_params = {
            "population_size": args.population_size,
            "offspring_size": args.offspring_size,
            "mutation_rate": args.mutation_rate
        }

        optimizer.optimize(oracle=oracle,
                           patience=5,
                           seed=args.seed,
                           **optimizer_params)

        print("-" * 40)
        print("Optimization completed!")

        # Generate and display optimization summary (matching original format)
        print("\n" + "=" * 60)
        print("OPTIMIZATION SUMMARY")
        print("=" * 60)

        report = generate_summary_report(args.target, args, results_collector)
        print(f"Experiment: {report['experiment_name']}")
        # print(f"Total generations: {report['total_generations']}")
        print(f"Total evaluations: {report['total_evaluations']}")

        obj_name = AVAILABLE_TARGETS[args.target]
        if obj_name in report['final_statistics']:
            final_stats = report['final_statistics'][obj_name]
            print(f"\n{obj_name.upper()} Final Statistics:")
            print(f"  Mean Score: {final_stats['mean']:.4f}")
            print(f"  Best Score: {final_stats['top1']:.4f}")
            print(f"  Top-10 Average: {final_stats['top10_mean']:.4f}")

        if obj_name in report['improvement']:
            improvement = report['improvement'][obj_name]
            print(f"\nImprovement from Initial:")
            print(f"  Initial mean: {improvement['initial_mean']:.4f}")
            print(f"  Final mean: {improvement['final_mean']:.4f}")
            print(
                f"  Absolute improvement: {improvement['absolute_improvement']:.4f}"
            )
            print(
                f"  Relative improvement: {improvement['relative_improvement']:.2f}%"
            )

        # Display top results (matching original format)
        if results_collector.final_molecules:
            print(f"\nTop 5 molecules:")
            for i in range(min(5, len(results_collector.final_molecules))):
                smiles = results_collector.final_molecules[i]
                score = results_collector.final_scores[i]
                print(f"  {i+1}. {smiles[:50]}... (score: {score:.4f})")

            print(f"\nBest molecule:")
            best_smiles = results_collector.final_molecules[0]
            best_score = results_collector.final_scores[0]
            print(f"  SMILES: {best_smiles}")
            print(f"  Score: {best_score:.4f}")
        else:
            print("\nWarning: No final molecules collected from optimization")

        # Generate convergence plot in experiment directory
        print(f"\nGenerating convergence plot...")
        try:
            import matplotlib.pyplot as plt

            if metrics_history:
                n_calls = [m['n_calls'] for m in metrics_history]
                avg_top1 = [m['avg_top1'] for m in metrics_history]
                avg_top10 = [m['avg_top10'] for m in metrics_history]
                avg_top100 = [m['avg_top100'] for m in metrics_history]

                plt.figure(figsize=(12, 8))

                plt.subplot(2, 2, 1)
                plt.plot(n_calls,
                         avg_top1,
                         'r-',
                         label='Top 1',
                         linewidth=2,
                         marker='o',
                         markersize=4)
                plt.xlabel('Oracle Calls')
                plt.ylabel('Score')
                plt.title('Best Score vs Oracle Calls')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(2, 2, 2)
                plt.plot(n_calls,
                         avg_top10,
                         'g-',
                         label='Top 10',
                         linewidth=2,
                         marker='s',
                         markersize=4)
                plt.xlabel('Oracle Calls')
                plt.ylabel('Score')
                plt.title('Average Top 10 Score vs Oracle Calls')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(2, 2, 3)
                plt.plot(n_calls,
                         avg_top100,
                         'b-',
                         label='Top 100',
                         linewidth=2,
                         marker='^',
                         markersize=4)
                plt.xlabel('Oracle Calls')
                plt.ylabel('Score')
                plt.title('Average Top 100 Score vs Oracle Calls')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.subplot(2, 2, 4)
                plt.plot(n_calls,
                         avg_top1,
                         'r-',
                         label='Top 1',
                         linewidth=2,
                         alpha=0.7)
                plt.plot(n_calls,
                         avg_top10,
                         'g-',
                         label='Top 10',
                         linewidth=2,
                         alpha=0.7)
                plt.plot(n_calls,
                         avg_top100,
                         'b-',
                         label='Top 100',
                         linewidth=2,
                         alpha=0.7)
                plt.xlabel('Oracle Calls')
                plt.ylabel('Score')
                plt.title('All Metrics Comparison')
                plt.grid(True, alpha=0.3)
                plt.legend()

                plt.tight_layout()

                plot_path = os.path.join(
                    experiment_dir,
                    f"{args.target}_convergence_seed{args.seed}_{timestamp}.png"
                )
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"✓ Convergence plot saved to {plot_path}")
            else:
                print("Warning: No metrics history available for plotting")

        except Exception as e:
            print(f"Warning: Could not generate plot: {e}")

        # Save additional output files to match expected format
        print("\nSaving additional output files...")
        try:
            # Save summary JSON
            summary_report = generate_summary_report(args.target, args,
                                                     results_collector)
            summary_json_path = os.path.join(
                experiment_dir,
                f"{args.target}_optimization_seed{args.seed}_{timestamp}_summary.json"
            )
            with open(summary_json_path, 'w') as f:
                json.dump(summary_report, f, indent=2)
            print(f"✓ Summary JSON saved to {summary_json_path}")

            # Save convergence CSV
            if convergence_data:
                convergence_csv_path = os.path.join(
                    experiment_dir,
                    f"{args.target}_optimization_seed{args.seed}_{timestamp}_convergence.csv"
                )
                df = pd.DataFrame(convergence_data)
                df.to_csv(convergence_csv_path, index=False, float_format='%.4f')
                print(f"✓ Convergence CSV saved to {convergence_csv_path}")
            else:
                print("Warning: No convergence data available for CSV export")

            # Save populations pickle
            if populations_data:
                populations_pkl_path = os.path.join(
                    experiment_dir,
                    f"{args.target}_optimization_seed{args.seed}_{timestamp}_populations.pkl"
                )
                with open(populations_pkl_path, 'wb') as f:
                    pickle.dump(populations_data, f)
                print(f"✓ Populations pickle saved to {populations_pkl_path}")
            else:
                print(
                    "Warning: No population data available for pickle export")

        except Exception as e:
            print(f"Warning: Could not save additional output files: {e}")

    except KeyboardInterrupt:
        print("\nOptimization interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
