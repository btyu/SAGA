import pandas as pd
import numpy as np
import os
import matplotlib
# Force non-interactive backend in headless/WSL unless explicitly overridden
if os.environ.get("MPLBACKEND") is None:
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("DISPLAY",
                                                           "") == "":
        matplotlib.use("Agg")
import matplotlib.pyplot as plt
import json
import pickle
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
from scileo_agent.core.data_models import Population, Candidate, Objective


class GALogger:
    """
    Logger for tracking genetic algorithm optimization progress.
    
    Features:
    - Store population statistics per generation
    - Track average, median, top1, top10 scores
    - Visualization and data export
    """

    def __init__(self,
                 objectives: List[Objective],
                 store_top_k: int = 100,
                 experiment_name: str = None,
                 output_dir: str = "logs"):
        """
        Initialize GA logger.
        
        Args:
            objectives: Optimization objectives
            store_top_k: Number of top candidates to store per generation
            experiment_name: Name for this run
            output_dir: Output directory for logs and plots
        """
        self.objectives = objectives
        self.objective_names = [obj.name for obj in objectives]
        self.store_top_k = store_top_k
        self.output_dir = output_dir
        self.experiment_name = experiment_name or f"GA_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.created_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._output_path = Path(self.output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)

        # Storage
        self.generation_stats = []
        self.populations_history = []
        self.convergence_data = pd.DataFrame()

        # Tracking
        self.current_generation = 0
        self.total_evaluations = 0
        self.start_time = datetime.now()

    def log_generation(self,
                       population: Population,
                       generation: int,
                       evaluations_used: int = 0,
                       additional_metrics: Dict[str, Any] = None):
        """Log data for a generation."""
        self.current_generation = generation
        self.total_evaluations += evaluations_used

        generation_data = {
            'generation': generation,
            'population_size': len(population.candidates),
            'evaluations_used': evaluations_used,
            'total_evaluations': self.total_evaluations,
            'timestamp': datetime.now()
        }

        # Calculate statistics for each objective
        for objective in self.objectives:
            obj_name = objective.name
            scores = self._extract_scores(population, obj_name)

            if scores:
                stats = self._calculate_statistics(scores)
                for stat_name, value in stats.items():
                    generation_data[f"{obj_name}_{stat_name}"] = value
            else:
                for stat_name in ['mean', 'median', 'top1', 'top10_mean']:
                    generation_data[f"{obj_name}_{stat_name}"] = np.nan

        if additional_metrics:
            generation_data.update(additional_metrics)

        self.generation_stats.append(generation_data)
        self._store_population(population, generation)
        self._update_convergence_data()
        # Persist CSV and plots every generation (overwrite)
        try:
            self._write_incremental_outputs()
        except Exception:
            pass

    def _extract_scores(self, population: Population,
                        objective_name: str) -> List[float]:
        """Extract scores for objective from population."""
        scores = []
        for candidate in population.candidates:
            score = candidate.scores.get(objective_name)
            if score is not None:
                scores.append(float(score))
        return scores

    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics."""
        scores_array = np.array(scores)
        sorted_scores = np.sort(scores_array)[::-1]  # Descending order

        return {
            'mean':
            np.mean(scores_array),
            'median':
            np.median(scores_array),
            'top1':
            sorted_scores[0] if len(sorted_scores) > 0 else np.nan,
            'top10_mean':
            np.mean(sorted_scores[:10])
            if len(sorted_scores) >= 10 else np.mean(sorted_scores)
        }

    def _store_population(self, population: Population, generation: int):
        """Store top candidates to save memory."""
        sorted_candidates = sorted(
            population.candidates,
            key=lambda c: self._compute_aggregate_score(c),
            reverse=True)

        top_candidates = sorted_candidates[:self.store_top_k]

        self.populations_history.append({
            'generation':
            generation,
            'top_candidates':
            top_candidates,
            'population_size':
            len(population.candidates)
        })

    def _compute_aggregate_score(self, candidate: Candidate) -> float:
        """Compute aggregate score across objectives."""
        total_score = 0.0
        for objective in self.objectives:
            score = candidate.scores.get(objective.name, 0.0)
            if score is not None:
                if objective.is_maximization:
                    total_score += score
                else:
                    total_score -= score
        return total_score

    def _update_convergence_data(self):
        """Update convergence DataFrame."""
        self.convergence_data = pd.DataFrame(self.generation_stats)

    def _write_incremental_outputs(self):
        """Write convergence CSV and plots each generation (overwrite)."""
        if self.convergence_data is None or self.convergence_data.empty:
            return
        base_name = f"{self.experiment_name}_{self.created_at}"
        csv_path = self._output_path / f"{base_name}_convergence.csv"
        self.convergence_data.to_csv(csv_path,
                                     index=False,
                                     float_format='%.4f')
        # Save plots
        conv_png = self._output_path / f"{base_name}_convergence.png"
        div_png = self._output_path / f"{base_name}_diversity.png"
        self.plot_convergence(save_path=str(conv_png))
        self.plot_diversity(save_path=str(div_png))

    def get_best_candidates(self, top_k: int = 10) -> List[Candidate]:
        """Get top K candidates across all generations."""
        all_candidates = []
        for pop_data in self.populations_history:
            all_candidates.extend(pop_data['top_candidates'])

        return sorted(all_candidates,
                      key=lambda c: self._compute_aggregate_score(c),
                      reverse=True)[:top_k]

    def plot_convergence(self,
                         objectives: List[str] = None,
                         metrics: List[str] = None,
                         save_path: str = None,
                         figsize: tuple = (12, 8)):
        """Plot convergence curves."""
        if self.convergence_data.empty:
            print("No data to plot.")
            return

        objectives = objectives or self.objective_names
        metrics = metrics or ['mean', 'top1', 'top10_mean']

        fig, axes = plt.subplots(len(objectives),
                                 len(metrics),
                                 figsize=figsize,
                                 squeeze=False)

        for i, obj_name in enumerate(objectives):
            for j, metric in enumerate(metrics):
                ax = axes[i, j]
                column_name = f"{obj_name}_{metric}"

                if column_name in self.convergence_data.columns:
                    ax.plot(self.convergence_data['generation'],
                            self.convergence_data[column_name],
                            marker='o',
                            linewidth=2,
                            markersize=4)
                    ax.set_title(f"{obj_name} - {metric}")
                    ax.set_xlabel("Generation")
                    ax.set_ylabel("Score")
                    ax.grid(True, alpha=0.3)
                else:
                    ax.text(0.5,
                            0.5,
                            f"No data for\n{column_name}",
                            ha='center',
                            va='center',
                            transform=ax.transAxes)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def plot_diversity(self, save_path: str = None, figsize: tuple = (8, 4)):
        """Plot diversity metrics over generations if available."""
        if self.convergence_data.empty:
            print("No data to plot.")
            return

        has_internal = "internal_diversity" in self.convergence_data.columns
        has_tanimoto = "mean_tanimoto" in self.convergence_data.columns
        if not (has_internal or has_tanimoto):
            print("No diversity metrics found in convergence data.")
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if has_internal:
            ax.plot(self.convergence_data["generation"],
                    self.convergence_data["internal_diversity"],
                    marker="o",
                    linewidth=2,
                    markersize=4,
                    label="Internal diversity")
        if has_tanimoto:
            ax.plot(self.convergence_data["generation"],
                    self.convergence_data["mean_tanimoto"],
                    marker="s",
                    linewidth=2,
                    markersize=4,
                    label="Mean Tanimoto")
        ax.set_title("Diversity over generations")
        ax.set_xlabel("Generation")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        ax.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

    def get_summary_report(self) -> Dict[str, Any]:
        """Generate summary report."""
        if not self.generation_stats:
            return {"error": "No data available"}

        final_stats = self.generation_stats[-1]
        initial_stats = self.generation_stats[0]

        report = {
            'experiment_name': self.experiment_name,
            'total_generations': self.current_generation,
            'total_evaluations': self.total_evaluations,
            'final_statistics': {},
            'improvement': {}
        }

        for obj_name in self.objective_names:
            # Final stats
            report['final_statistics'][obj_name] = {
                'mean': final_stats.get(f"{obj_name}_mean", "N/A"),
                'top1': final_stats.get(f"{obj_name}_top1", "N/A"),
                'top10_mean': final_stats.get(f"{obj_name}_top10_mean", "N/A")
            }

            # Improvement
            initial_mean = initial_stats.get(f"{obj_name}_mean", 0)
            final_mean = final_stats.get(f"{obj_name}_mean", 0)
            improvement = final_mean - initial_mean if initial_mean and final_mean else 0

            report['improvement'][obj_name] = {
                'initial_mean':
                initial_mean,
                'final_mean':
                final_mean,
                'absolute_improvement':
                improvement,
                'relative_improvement':
                (improvement / initial_mean * 100) if initial_mean != 0 else 0
            }

            # Top-1 improvement
            initial_top1 = initial_stats.get(f"{obj_name}_top1", 0)
            final_top1 = final_stats.get(f"{obj_name}_top1", 0)
            top1_improvement = final_top1 - initial_top1 if initial_top1 and final_top1 else 0
            report.setdefault('improvement_top1', {})[obj_name] = {
                'initial_top1':
                initial_top1,
                'final_top1':
                final_top1,
                'absolute_improvement':
                top1_improvement,
                'relative_improvement': (top1_improvement / initial_top1 *
                                         100) if initial_top1 != 0 else 0,
            }

        return report

    def save_log(self, directory: str = None):
        """Save optimization log."""
        directory = directory or self.output_dir
        log_dir = Path(directory)
        log_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = f"{self.experiment_name}_{timestamp}"

        # Save convergence data
        csv_path = log_dir / f"{base_name}_convergence.csv"
        self.convergence_data.to_csv(csv_path,
                                     index=False,
                                     float_format='%.4f')

        # Save populations
        pkl_path = log_dir / f"{base_name}_populations.pkl"
        with open(pkl_path, 'wb') as f:
            pickle.dump(self.populations_history, f)

        # Save summary
        json_path = log_dir / f"{base_name}_summary.json"
        with open(json_path, 'w') as f:
            json.dump(self.get_summary_report(), f, indent=2, default=str)

        print(f"Logs saved to {log_dir}:")
        print(f"  - Convergence: {csv_path}")
        print(f"  - Populations: {pkl_path}")
        print(f"  - Summary: {json_path}")

        return {
            'convergence': csv_path,
            'populations': pkl_path,
            'summary': json_path
        }

    def print_generation_summary(self, generation: int = None):
        """Print summary for a generation."""
        if not self.generation_stats:
            print("No generation data available")
            return

        gen_data = self.generation_stats[generation or -1]
        print(f"\nGeneration {gen_data['generation']} Summary:")
        print(f"  Population size: {gen_data['population_size']}")
        print(f"  Evaluations used: {gen_data['evaluations_used']}")

        for obj_name in self.objective_names:
            mean_score = gen_data.get(f"{obj_name}_mean", "N/A")
            top1_score = gen_data.get(f"{obj_name}_top1", "N/A")
            top10_mean = gen_data.get(f"{obj_name}_top10_mean", "N/A")

            print(f"  {obj_name}:")
            print(f"    Mean: {mean_score:.4f}" if mean_score !=
                  "N/A" else f"    Mean: {mean_score}")
            print(f"    Top1: {top1_score:.4f}" if top1_score !=
                  "N/A" else f"    Top1: {top1_score}")
            print(f"    Top10 avg: {top10_mean:.4f}" if top10_mean !=
                  "N/A" else f"    Top10 avg: {top10_mean}")
