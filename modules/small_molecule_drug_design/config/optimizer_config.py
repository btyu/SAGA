"""
Type-safe configuration for molecular genetic algorithm optimizer.
"""

from dataclasses import dataclass, field
from typing import List, Optional

try:
    from typing import Literal
except ImportError:
    # Python < 3.8
    from typing_extensions import Literal


@dataclass
class GeneticAlgorithmConfig:
    """Type-safe configuration for molecular GA optimizer."""

    # Population parameters
    population_size: int = 120
    offspring_size: int = 100
    mutation_size: int = 20
    oracle_budget: int = 10000
    seed: int = 42

    # Mutation settings
    crossover_mode: Literal["llm", "gb_ga"] = "llm"
    mutation_mode: Literal["llm", "non_llm", "gb_ga"] = "llm"
    non_llm_mutation_rate: float = 1.0

    # Selection strategy
    selection_strategy: str = "objective_summation"  # or "pareto_set_selection"

    # Survival selection
    survival_selection_method: str = "tournament"  # "tournament", "fitness", "diverse_top", "butina_cluster"
    survival_tanimoto_threshold: float = 0.4

    # Elitism configuration
    elitism_fraction: float = 0.05
    elitism_fields: List[str] = field(default_factory=list)

    # Early stopping
    enable_early_stopping: bool = True
    early_stopping_threshold: float = 1e-3
    early_stopping_patience: int = 10
    convergence_threshold: float = 1e-6

    # Similarity filtering
    similarity_threshold: Optional[float] = None

    # Parallelization
    max_workers: Optional[int] = None

    # Parent selection (rank-based)
    rank_percentile_bins: List[float] = field(
        default_factory=lambda: [0.1, 0.3, 0.6, 1.0])
    rank_percentile_weights: List[float] = field(
        default_factory=lambda: [4.0, 2.0, 1.0, 0.5])

    # Structure filtering
    enable_structure_filter: bool = False

    # Objective combiner
    objective_combiner: str = "simple_product"

    # Initial population
    init_group: str = "diverse_10k"
    file_sample_size: int = 10

    # 3D docking
    add_3d_docked_pose_info: bool = False

    # Optional objective weights (if not using LLM for weight selection)
    objectives_weights: Optional[List[float]] = None

    def validate(self) -> None:
        """Validate configuration consistency."""
        if self.mutation_size > self.population_size:
            raise ValueError(
                f"mutation_size ({self.mutation_size}) cannot exceed "
                f"population_size ({self.population_size})")

        if self.offspring_size > self.oracle_budget:
            raise ValueError(
                f"offspring_size ({self.offspring_size}) should not exceed "
                f"oracle_budget ({self.oracle_budget})")

        if self.elitism_fraction < 0 or self.elitism_fraction > 1:
            raise ValueError(
                f"elitism_fraction must be in [0, 1], got {self.elitism_fraction}"
            )

        if len(self.rank_percentile_bins) != len(self.rank_percentile_weights):
            raise ValueError(
                f"rank_percentile_bins and rank_percentile_weights must have same length"
            )

        if self.rank_percentile_bins and self.rank_percentile_bins[-1] < 1.0:
            raise ValueError(
                f"Last rank_percentile_bin must be >= 1.0, got {self.rank_percentile_bins[-1]}"
            )

        if self.crossover_mode not in ["llm", "gb_ga"]:
            raise ValueError(
                f"crossover_mode must be 'llm' or 'gb_ga', got {self.crossover_mode}"
            )

        if self.mutation_mode not in ["llm", "non_llm", "gb_ga"]:
            raise ValueError(
                f"mutation_mode must be 'llm', 'non_llm', or 'gb_ga', got {self.mutation_mode}"
            )

    def to_dict(self) -> dict:
        """Convert to dictionary for compatibility with existing code."""
        return {
            "population_size": self.population_size,
            "offspring_size": self.offspring_size,
            "mutation_size": self.mutation_size,
            "oracle_budget": self.oracle_budget,
            "seed": self.seed,
            "crossover_mode": self.crossover_mode,
            "mutation_mode": self.mutation_mode,
            "non_llm_mutation_rate": self.non_llm_mutation_rate,
            "selection_strategy": self.selection_strategy,
            "survival_selection_method": self.survival_selection_method,
            "survival_tanimoto_threshold": self.survival_tanimoto_threshold,
            "elitism_fraction": self.elitism_fraction,
            "elitism_fields": self.elitism_fields,
            "enable_early_stopping": self.enable_early_stopping,
            "early_stopping_threshold": self.early_stopping_threshold,
            "early_stopping_patience": self.early_stopping_patience,
            "convergence_threshold": self.convergence_threshold,
            "similarity_threshold": self.similarity_threshold,
            "max_workers": self.max_workers,
            "rank_percentile_bins": self.rank_percentile_bins,
            "rank_percentile_weights": self.rank_percentile_weights,
            "enable_structure_filter": self.enable_structure_filter,
            "objective_combiner": self.objective_combiner,
            "file_sample_size": self.file_sample_size,
            "add_3d_docked_pose_info": self.add_3d_docked_pose_info,
            "objectives_weights": self.objectives_weights,
        }

    @classmethod
    def from_dict(cls, config_dict: dict) -> "GeneticAlgorithmConfig":
        """Create config from dictionary."""
        # Filter to only known fields
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {
            k: v
            for k, v in config_dict.items() if k in valid_fields
        }
        return cls(**filtered_dict)
