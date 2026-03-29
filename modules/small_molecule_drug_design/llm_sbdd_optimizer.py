import re
import logging
import numpy as np
from typing import List, Optional, Dict, Any, Tuple
import pandas as pd
import random
from tqdm import tqdm
import asyncio
import yaml
import time
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import AllChem
from modules.small_molecule_drug_design.utils.mutation_ops import mutate_smiles
from modules.small_molecule_drug_design.utils.gbga_ops import (
    gbga_crossover,
    gbga_mutate,
)
from scileo_agent.core.registry import register_module, get_scorer, list_scorers
from scileo_agent.core.modules.optimizer import OptimizerModule
from scileo_agent.core.data_models import (
    Population,
    Objective,
    Candidate,
    ObjectiveIndex,
)
from scileo_agent.core.config import LLMConfig
from .combiner import (
    ObjectiveCombiner,
    SimpleSumCombiner,
    SimpleProductCombiner,
    WeightedSumCombiner,
)
from modules.small_molecule_drug_design.utils.rdkit_utils import (
    calculate_population_diversity,
    select_top_diverse_modes,
    structure_filter,
)
from modules.small_molecule_drug_design.selection import (
    FitnessSurvivalSelection,
    DiverseTopSurvivalSelection,
    ButinaClusterSurvivalSelection,
)
from modules.small_molecule_drug_design.selection.parent_selection import (
    TournamentSelector,
)
from modules.small_molecule_drug_design.ga_logging import (
    GALogger,
    ChemistLogger,
    LLMExample,
)
from modules.small_molecule_drug_design.prompting.common_prompts import (
    SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT,
)
from modules.small_molecule_drug_design.prompting.multiobj_prompts import (
    build_multiobj_crossover_prompt,
    build_multiobj_mutation_prompt,
    build_multiobjective_weightage_prompt,
)
from modules.small_molecule_drug_design.prompting.barebone_prompts import (
    build_barebone_crossover_prompt,
    build_barebone_mutation_prompt,
)
import os

module_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(module_dir)  # modules/

# TODO log:
# - implement the multi-objective optimization

# Example objective:
# QED:
# OBJECTIVE: has a higher QED score
# TASK: QED scores
# OBJECTIVE_DESCRIPTION: The QED score measures the drug-likeness of the molecule.

# ENUM class
from enum import Enum


class SelectionStrategy(Enum):
    OBJECTIVE_SUMMATION = "objective_summation"
    PARETO_SET_SELECTION = "pareto_set_selection"


@register_module(
    "llm_sbdd_optimizer", "1.0.0"
)  # Register the optimizer with a unique name and version ("x.y.z.")
class LLMSBDDOptimizer(OptimizerModule):
    """
    Custom optimizer for your specific domain.
    """

    def __init__(
        self,
        module_id: str,  # The unique identifier for the optimizer instance for running
        config: Dict[str, Any] = None,  # The configuration for the optimizer
        llm_config: Optional[
            LLMConfig
        ] = None,  # The LLM configuration for the optimizer, None if no LLM is needed,
        init_group: str = "zinc",
    ):
        super().__init__(module_id=module_id, config=config, llm_config=llm_config)

        # Basic genetic algorithm parameters
        self.population_size = self.config.get("population_size", 120)
        self.offspring_size = self.config.get("offspring_size", 70)
        self.mutation_size = self.config.get("mutation_size", 7)  # 120 * 0.667
        self.oracle_budget = self.config.get("oracle_budget", 10000)

        # For multi-objective optimization, two strategies are investigated: (1) Objective summation, where
        # the summation of individual objectives is used as a single objective, and the nc fittest members are
        # retained; and (2) Pareto set selection, where only the Pareto frontier of the current population is kept.
        self.selection_strategy = self.config.get(
            "selection_strategy", SelectionStrategy.OBJECTIVE_SUMMATION
        )

        # Early stopping parameters
        self.convergence_threshold = self.config.get("convergence_threshold", 1e-6)
        self.seed = self.config.get("seed", 42)

        # Similarity filtering threshold (None means no threshold)
        self.similarity_threshold = self.config.get("similarity_threshold", None)

        # Maximum number of parallel LLM workers for concurrency control
        # Default to 20 to avoid overwhelming API rate limits, None means unlimited
        max_workers_config = self.config.get("max_workers")
        if max_workers_config is None:
            # Default to 20 if not specified to avoid overwhelming API rate limits
            self.max_workers = 20
        else:
            self.max_workers = max_workers_config

        # Create semaphore for controlling concurrent LLM calls
        # This prevents overwhelming API rate limits while still allowing parallelism
        # Note: Semaphore is created here but will be used in async methods
        self._llm_semaphore = asyncio.Semaphore(self.max_workers)
        logging.info(
            f"LLM concurrency limit set to {self.max_workers} concurrent requests"
        )

        # Mutation mode: 'non_llm' (RDKit-based) or 'llm'
        self.mutation_mode = self.config.get("mutation_mode", "non_llm")
        self.non_llm_mutation_rate = float(
            self.config.get("non_llm_mutation_rate", 1.0)
        )

        # Early stopping parameters
        self.early_stopping_threshold = self.config.get(
            "early_stopping_threshold", 1e-3
        )
        self.early_stopping_patience = self.config.get("early_stopping_patience", 25)

        # Parent selection: tournament selection
        self.tournament_size = self.config.get("tournament_size", 3)

        # Track all evaluated SMILES to avoid counting duplicates toward budget
        self._evaluated_smiles: set = set()

        # Survivor selection configuration
        # Options:
        #  - "fitness": current approach (sort by aggregated score)
        #  - "diverse_top": use select_top_diverse_modes to enforce diversity
        self.survival_selection_method = self.config.get(
            "survival_selection_method", "fitness"
        )
        self.survival_tanimoto_threshold = 0.4

        # Elitism configuration: fraction of current population to carry over as elites
        # Defaults to 10%
        self.elitism_fraction = float(self.config.get("elitism_fraction", 0.025))
        raw_elitism_fields = self.config.get("elitism_fields", [])
        self.elitism_fields = list(raw_elitism_fields) if raw_elitism_fields else []

        # Init group: prefer config value, fall back to constructor param
        self.init_group = self.config.get("init_group", init_group)
        self.initial_population = None

        # Initialize survival selection strategy object
        self.survival_selector = self._build_survival_selector(
            self.survival_selection_method
        )

        # Structure filtering configuration
        self.enable_structure_filter = self.config.get("enable_structure_filter", False)

        # Objective combiner selection (default: simple_product)
        self.objective_combiner = self.config.get(
            "objective_combiner", "simple_product"
        )
        self.combiner: ObjectiveCombiner = self._build_combiner(self.objective_combiner)

        # Determine whether an LLM client is available; fall back to manual ops when absent
        self.llm_available = self.has_llm()
        self.manual_genetic_ops = (
            bool(self.config.get("force_manual_genetic_ops", False))
            or not self.llm_available
        )
        if not self.llm_available:
            logging.warning(
                "LLM client is not available. Falling back to manual crossover/mutation heuristics."
            )
        if self.manual_genetic_ops and self.config.get("mutation_mode") != "non_llm":
            logging.info("Disabling LLM mutations in manual mode.")
            self.config["mutation_mode"] = "non_llm"

        # Refresh mutation mode in case manual settings changed it
        self.mutation_mode = self.config.get("mutation_mode", "non_llm")

        self.enable_early_stopping = self.config.get("enable_early_stopping", False)

        self.add_3d_docked_pose_info = self.config.get("add_3d_docked_pose_info", False)
        # init_group already set above

        # Timing tracking for performance analysis
        self.timing_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})
        self.gen_timing_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

    def set_objectives_weights(self, objectives_weights: Optional[List[float]] = None):
        """
        Set the objectives for the optimizer.

        Args:
            objectives_weights: Optional list of weights for each objective
        """
        # Accept explicit weights or leave unset to be initialized later in optimize()
        self.objectives_weights = objectives_weights

    def _create_initial_population(self) -> Population:
        """Initialize the population by sampling molecules from enamine_diverse_10k dataset."""
        smiles_list = self._sample_smiles(
            self.population_size, seed=self.seed, init_group=self.init_group
        )
        candidates = [
            Candidate(representation=self._sanitize_smiles_value(smiles))
            for smiles in smiles_list
        ]
        return Population(candidates=candidates)

    def _sample_smiles(
        self, num_samples: int, seed: Optional[int] = None, init_group=None
    ) -> List[str]:
        """Sample up to num_samples SMILES from multiple sources with reproducibility.

        Supported init_group values:
          - 'zinc': load modules/.../zinc_250k.csv (column 'smiles')
          - 'diverse_10k': load enamine_diverse_10k.csv (column 'SMILES')
          - 'enamine': load large_scale_molecule.csv (column 'smiles')
          - 'covid': load bundled known_covid.smi (one SMILES per line)
          - '@<path>': load a .smi/.txt (one SMILES per line) or .csv with column 'smiles' or 'SMILES'
        """

        rng = random.Random(seed if seed is not None else self.seed)

        # Normalize and detect file-based sources
        group = (
            (init_group or "").strip() if isinstance(init_group, str) else init_group
        )
        file_smiles: Optional[List[str]] = None

        try:
            if isinstance(group, str) and group.startswith("@"):
                file_path = group[1:]
                if file_path.lower().endswith(".csv"):
                    df = pd.read_csv(file_path)
                    smiles_col = None
                    if "smiles" in df.columns:
                        smiles_col = "smiles"
                    elif "SMILES" in df.columns:
                        smiles_col = "SMILES"
                    if smiles_col is None:
                        raise RuntimeError(
                            f"CSV '{file_path}' must contain a 'smiles' or 'SMILES' column"
                        )
                    file_smiles = [
                        self._sanitize_smiles_value(s)
                        for s in df[smiles_col].astype(str).tolist()
                    ]
                else:
                    with open(file_path, "r") as fh:
                        file_smiles = [
                            self._sanitize_smiles_value(line)
                            for line in fh
                            if line.strip()
                        ]
            elif group == "covid":
                covid_path = os.path.join(
                    module_dir, "data", "molecules", "known_covid.smi"
                )
                with open(covid_path, "r") as fh:
                    file_smiles = [
                        self._sanitize_smiles_value(line) for line in fh if line.strip()
                    ]
        except Exception as e:
            raise RuntimeError(f"Failed to load init_group file '{group}': {e}")

        if file_smiles is not None and len(file_smiles) > 0:
            smiles = [str(s) for s in file_smiles]
            if num_samples >= len(smiles):
                return list(smiles)
            indices = list(range(len(smiles)))
            rng.shuffle(indices)
            selected = indices[:num_samples]
            return [smiles[i] for i in selected]

        if group == "zinc":
            df = pd.read_csv(module_dir + "/data/molecules/zinc_250k.csv")
            smiles = df["smiles"].astype(str).tolist()
        elif group == "diverse_10k":
            df = pd.read_csv(module_dir + "/data/molecules/enamine_diverse_10k.csv")
            smiles = df["SMILES"].astype(str).tolist()
        elif group == "enamine_top500":
            df = pd.read_csv(
                "/gpfs/radev/home/tl688/pitl688/scileoagent_drug/examine_extracted_500.csv"
            )
            smiles = df["smiles"].astype(str).tolist()
        elif group == "enamine":
            df = pd.read_csv(module_dir + "/data/molecules/enamine_diverse_10k.csv")
            smiles = df["smiles"].astype(str).tolist()
        elif group == "mproknownbinder":
            df = pd.read_csv(
                os.path.join(
                    module_dir, "data", "molecules", "mpro_protein_bindinglist.csv"
                )
            )
            smiles = df["smiles"].astype(str).tolist()
        elif group == "mpro_examine_mix":
            df = pd.read_csv(module_dir + "/data/molecules/mpro_update_3090.csv")
            smiles = df["smiles"].astype(str).tolist()
        else:
            # Default fallback to diverse_10k if unknown
            df = pd.read_csv(module_dir + "/data/molecules/enamine_diverse_10k.csv")
            smiles = df["SMILES"].astype(str).tolist()

        if num_samples >= len(smiles):
            return list(smiles)
        indices = list(range(len(smiles)))
        rng.shuffle(indices)
        selected = indices[:num_samples]
        return [smiles[i] for i in selected]

    @property
    def requires_initial_population(self) -> bool:
        """Whether this optimizer requires an initial population."""
        return False

    async def create_random_candidates(
        self, num_candidates: int, **additional_kwargs: Dict[str, Any]
    ) -> List[Candidate]:
        """
        Create random candidates by sampling from the initial dataset.

        Args:
            num_candidates: Number of candidates to create

        Returns:
            List of randomly created candidates
        """
        smiles_list = self._sample_smiles(
            num_candidates,
            seed=getattr(self, "seed", None),
            init_group=getattr(self, "init_group", "diverse_10k"),
        )
        candidates = [
            Candidate(representation=self._sanitize_smiles_value(smiles))
            for smiles in smiles_list
        ]
        return candidates

    # ================

    # This is an optional method.
    # If you want to check the input objectives are compatible with this optimizer,
    # for example, you want to check if the necessary objectives are present,
    # you can implement this method.
    # It will be called by the framework before your `optimize()` method.
    def check_objectives(self, objectives: List[Objective]) -> None:
        """Validate the input objectives are compatible with this optimizer."""
        pass

    def _compute_candidate_score(
        self,
        candidate: Candidate,
        objectives: List[Objective] = None,
    ) -> float:
        """
        Aggregate the objective values of the candidate.

        Args:
            candidate: The candidate to aggregate the objective values of
            objectives: The objectives to aggregate the objective values of

        Returns:
            The aggregated objective values of the candidate
        """
        # assert len(objectives) == len(
        #     self.objectives_weights
        # ), "Objectives and weights must have the same length"

        # Delegate aggregation to the configured combiner
        multiobj_score = self.combiner.combine(
            candidate, objectives, self.objectives_weights
        )
        candidate.scores["multiobj_score"] = multiobj_score
        return multiobj_score

    def _filter_population(self, population: Population) -> Population:
        """
        Filter population to remove molecules containing fluorine (F) or sulfur (S).

        Args:
            population: Population to filter

        Returns:
            Filtered population with molecules containing F or S removed
        """
        if not self.enable_structure_filter:
            return population

        filtered_candidates = []
        for candidate in population:
            try:
                check_f = structure_filter("F", candidate.representation)
                check_s = structure_filter("S", candidate.representation)
                if not check_f and not check_s:
                    filtered_candidates.append(candidate)
            except:
                continue

        return Population(candidates=filtered_candidates)

    def _sanitize_smiles_value(self, molecule_str: str) -> str:
        """Normalize a SMILES string extracted from LLM output.

        - Strip surrounding quotes/backticks
        - Remove CXSMILES annotations (anything after the first '|')
        - Trim trailing commas/semicolons
        - Keep only the first whitespace-delimited token (SMILES must be single token)
        """
        s = str(molecule_str).strip()
        if len(s) >= 2 and (
            (s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")
        ):
            s = s[1:-1].strip()
        if s.startswith("`") and s.endswith("`") and len(s) >= 2:
            s = s[1:-1].strip()
        if "|" in s:
            s = s.split("|", 1)[0].strip()
        s = s.rstrip(",;").strip()
        # Keep only the first token to avoid inline comments or extra fields
        if any(ch.isspace() for ch in s):
            s = s.split()[0]
        return s

    def _is_valid_smiles(self, smiles: str) -> bool:
        """Quick SMILES validity check using RDKit."""
        if not smiles:
            return False
        try:
            return Chem.MolFromSmiles(smiles) is not None
        except Exception:
            return False

    def _manual_crossover(self, parents: List[Candidate]) -> Candidate:
        """
        Graph-GA style crossover for manual (non-LLM) mode.
        """
        smiles_a = self._sanitize_smiles_value(parents[0].representation or "")
        smiles_b = self._sanitize_smiles_value(parents[1].representation or "")
        attempts = int(self.config.get("gbga_crossover_attempts", 5))
        for _ in range(max(1, attempts)):
            child_smiles = gbga_crossover(smiles_a, smiles_b)
            if child_smiles:
                sanitized = self._sanitize_smiles_value(child_smiles)
                if self._is_valid_smiles(sanitized):
                    return Candidate(representation=sanitized)
            # Swap ordering to expose different fragments
            smiles_a, smiles_b = smiles_b, smiles_a
        # Fallback: mutate one of the parents if crossover fails
        fallback_parent = random.choice(parents)
        return self._non_llm_mutation(fallback_parent)

    def _non_llm_mutation(self, candidate: Candidate) -> Candidate:
        """Graph-GA mutation helper shared by manual + explicit non-LLM modes."""
        base_smiles = self._sanitize_smiles_value(candidate.representation or "")
        attempts = int(self.config.get("gbga_mutation_attempts", 5))
        for _ in range(max(1, attempts)):
            mutant_smiles = gbga_mutate(base_smiles, self.non_llm_mutation_rate)
            if mutant_smiles:
                sanitized = self._sanitize_smiles_value(mutant_smiles)
                if self._is_valid_smiles(sanitized):
                    return Candidate(representation=sanitized)
        # If all attempts fail, return original (invalid SMILES will later be filtered)
        return Candidate(representation=base_smiles)

    def _manual_mutation(self, candidate: Candidate) -> Candidate:
        """Wrapper to keep semantics explicit in manual mode."""
        return self._non_llm_mutation(candidate)

    # Removed local prompt constants; using prompting builders instead
    def _parse_multiobjective_weightage_response(
        self, response: str
    ) -> Tuple[List[float], str]:
        """
        Parse the LLM response to get multiobjective weightage list

        Expected YAML response format:
        explanation: some text
        weights: [0.8, 0.2, ...]
        """
        try:
            # Parse YAML response
            parsed = yaml.safe_load(response["content"].strip())
            explanation = parsed["explanation"]
            multiobj_weights = parsed["weights"]

            return multiobj_weights, explanation
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(
                f"Could not parse multiobjective weightage response: {e}. Response: {response}"
            )

    def _get_multiobjective_weights(
        self, objectives: List[Objective], human_logger=None
    ) -> List[float]:
        """
        Get the multi-objective weights from LLM based on the objectives

        Args:
            objectives: The objectives to optimize

        Returns:
            A new offspring candidate
        """
        if not self.llm_available:
            logging.warning(
                "LLM unavailable; defaulting to equal multi-objective weights."
            )
            return [1.0 for _ in objectives]

        prompt = build_multiobjective_weightage_prompt(objectives)
        response = self.call_llm_with_prompt(
            prompt, system_prompt=SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT
        )
        try:
            (
                multiobj_weights,
                explanation,
            ) = self._parse_multiobjective_weightage_response(response)
        except Exception as e:
            print(
                f"Multi-objective weightage parsing failed: {e}. Using equal weights instead."
            )
            multiobj_weights = [1.0 for _ in objectives]
        # Optional human logger: record prompt/response/weights
        if human_logger is not None:
            try:
                human_logger.log_weight_prompt(
                    str(prompt),
                    str(response.get("content", response)),
                    multiobj_weights,
                )
                human_logger.set_objectives_info(objectives, multiobj_weights)
            except Exception as e:
                logging.warning(
                    "Failed to record weight prompt/response in human_logger: %s",
                    e,
                    exc_info=True,
                )
        return multiobj_weights

    def _select_parents(
        self, population: Population, objectives: List[Objective], k=2
    ) -> List[Candidate]:
        """
        Select k unique parents using tournament selection.

        Args:
            population: The population of candidates
            k: The number of parents to select (default is 2)

        Returns:
            A list of k unique parents
        """
        selected = []
        candidates = population.candidates

        for _ in range(k):
            # Randomly select tournament_size candidates
            tournament_size = min(self.tournament_size, len(candidates))
            tournament_indices = np.random.choice(
                len(candidates), size=tournament_size, replace=False
            )

            # Find best in tournament
            best_idx = None
            best_score = float("-inf")
            for idx in tournament_indices:
                candidate = candidates[idx]
                score = self._compute_candidate_score(candidate, objectives)
                if score > best_score:
                    best_score = score
                    best_idx = idx

            selected.append(candidates[best_idx])

        return selected

    def _format_crossover_prompt(
        self, parents: List[Candidate], objectives: List[Objective]
    ) -> str:
        """
        Format the prompt for the LLM.
        """
        # Compute aggregated scores to include in prompt
        agg_a = self._compute_candidate_score(parents[0], objectives)
        agg_b = self._compute_candidate_score(parents[1], objectives)
        aggregation_equation = self.combiner.aggregation_equation(
            objectives, self.objectives_weights
        )

        # Use barebone prompts by default (shorter, faster)
        use_barebone = self.config.get("use_barebone_prompts", True)

        if use_barebone:
            return build_barebone_crossover_prompt(
                parents[0],
                parents[1],
                objectives,
                aggregation_equation,
                agg_a,
                agg_b,
            )
        else:
            # 3D docked pose pocket residue map
            residue_map_a = parents[0].metadata.get("docking_residue_map", None)
            residue_map_b = parents[1].metadata.get("docking_residue_map", None)
            return build_multiobj_crossover_prompt(
                parents,
                objectives,
                aggregation_equation,
                agg_a,
                agg_b,
                self.add_3d_docked_pose_info,
                residue_map_a,
                residue_map_b,
            )

    def _parse_crossover_response(self, response: str) -> Tuple[Candidate, str]:
        """
        Parse the LLM response to create a new offspring candidate.

        Handles two formats:
        1. Barebone prompts: Just a SMILES string
        2. Detailed prompts: YAML with explanation and molecule keys
        """
        try:
            # Extract content from response dict if needed
            if isinstance(response, dict):
                content = response.get("content", "").strip()
            else:
                content = str(response).strip()

            if not content:
                raise ValueError("Empty response content")

            # Check if this is a barebone response (just SMILES, no YAML markers)
            # Barebone prompts return only SMILES string
            use_barebone = self.config.get("use_barebone_prompts", True)

            if use_barebone:
                # For barebone prompts, the response should be just a SMILES string
                # Check if it looks like structured YAML/JSON
                if (
                    "molecule:" not in content
                    and "explanation:" not in content
                    and "---" not in content
                ):
                    # Likely just a SMILES string - use it directly
                    molecule = self._sanitize_smiles_value(content)
                    explanation = "Barebone prompt response"
                    return Candidate(representation=molecule), explanation

            # Try to extract molecule and explanation using regex patterns (for detailed prompts)
            molecule_pattern = r"molecule:\s*([^\n\r]+)"
            molecule_match = re.search(molecule_pattern, content, re.IGNORECASE)

            explanation_pattern = r"explanation:\s*(.*?)(?=\n\s*molecule:|$)"
            explanation_match = re.search(
                explanation_pattern, content, re.IGNORECASE | re.DOTALL
            )

            if molecule_match and explanation_match:
                molecule = self._sanitize_smiles_value(molecule_match.group(1))
                explanation = explanation_match.group(1).strip()
                return Candidate(representation=molecule), explanation

            # Fallback: try YAML parsing
            if "---" in content:
                documents = content.split("---")
                parsed = yaml.safe_load(documents[-1].strip())
            else:
                parsed = yaml.safe_load(content)

            if parsed and isinstance(parsed, dict):
                explanation = parsed.get("explanation", "No explanation provided")
                molecule = self._sanitize_smiles_value(parsed.get("molecule", ""))
                if molecule:
                    return Candidate(representation=molecule), explanation

            # Last resort: try to extract SMILES from the content directly
            # This handles cases where LLM returns SMILES with some extra text
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not ":" in line:
                    # Try to parse as SMILES
                    sanitized = self._sanitize_smiles_value(line)
                    if self._is_valid_smiles(sanitized):
                        return (
                            Candidate(representation=sanitized),
                            "Extracted from response",
                        )

            raise ValueError(f"Could not extract SMILES from response: {content[:200]}")

        except (yaml.YAMLError, KeyError, AttributeError, ValueError) as e:
            raise ValueError(
                f"Could not parse crossover response: {e}. Response: {response}"
            )

    def _crossover(
        self,
        parents: List[Candidate],
        objectives: List[Objective],
        return_artifacts: bool = False,
    ):
        """
        Crossover two parents to create a new offspring.

        Args:
            parents: A list of two parents
            objective: The objective to optimize

        Returns:
            A new offspring candidate
        """
        if self.manual_genetic_ops:
            offspring = self._manual_crossover(parents)
            if return_artifacts:
                return offspring, "MANUAL_CROSSOVER", {"content": "manual"}
            return offspring

        prompt = self._format_crossover_prompt(parents, objectives)
        start_time = time.time()
        response = self.call_llm_with_prompt(
            prompt, system_prompt=SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT
        )
        elapsed = time.time() - start_time
        self.timing_stats["llm_crossover"]["count"] += 1
        self.timing_stats["llm_crossover"]["total_time"] += elapsed
        self.gen_timing_stats["llm_crossover"]["count"] += 1
        self.gen_timing_stats["llm_crossover"]["total_time"] += elapsed
        offspring, explanation = self._parse_crossover_response(response)
        if return_artifacts:
            return offspring, prompt, response
        return offspring

    async def _crossover_async(
        self,
        parents: List[Candidate],
        objectives: List[Objective],
        return_artifacts: bool = False,
    ):
        """
        Async version of crossover for parallel execution.

        Args:
            parents: A list of two parents
            objectives: The objectives to optimize
            return_artifacts: Whether to return prompt/response artifacts

        Returns:
            A new offspring candidate (and optionally artifacts)
        """
        if self.manual_genetic_ops:
            offspring = self._manual_crossover(parents)
            if return_artifacts:
                return offspring, "MANUAL_CROSSOVER", {"content": "manual"}
            return offspring

        prompt = self._format_crossover_prompt(parents, objectives)
        start_time = time.time()

        # Use semaphore to control concurrency (prevents overwhelming API rate limits)
        async with self._llm_semaphore:
            response = await self.call_llm_with_prompt_async(
                prompt, system_prompt=SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT
            )

        elapsed = time.time() - start_time
        self.timing_stats["llm_crossover"]["count"] += 1
        self.timing_stats["llm_crossover"]["total_time"] += elapsed
        self.gen_timing_stats["llm_crossover"]["count"] += 1
        self.gen_timing_stats["llm_crossover"]["total_time"] += elapsed
        offspring, explanation = self._parse_crossover_response(response)
        if return_artifacts:
            return offspring, prompt, response
        return offspring

    def _format_mutation_prompt(
        self, candidate: Candidate, objectives: List[Objective]
    ) -> str:
        """
        Format the mutation prompt for the LLM.

        Args:
            candidate: The candidate to mutate
            objective: The objective to optimize

        Returns:
            Formatted prompt string
        """
        # Compute aggregated score to include in prompt
        agg = self._compute_candidate_score(candidate, objectives)
        aggregation_equation = self.combiner.aggregation_equation(
            objectives, self.objectives_weights
        )

        # Use barebone prompts by default (shorter, faster)
        use_barebone = self.config.get("use_barebone_prompts", True)

        if use_barebone:
            return build_barebone_mutation_prompt(
                candidate, objectives, aggregation_equation, agg
            )
        else:
            return build_multiobj_mutation_prompt(
                candidate, objectives, aggregation_equation, agg
            )

    # --- End non-LLM helpers (moved to utils.mutation_ops) ---

    def _parse_mutation_response(self, response: str) -> Tuple[Candidate, str]:
        """
        Parse the LLM mutation response to create a new mutated candidate.

        Handles two formats:
        1. Barebone prompts: Just a SMILES string
        2. Detailed prompts: YAML with explanation and molecule keys

        Args:
            response: LLM response dict or string

        Returns:
            Tuple of (mutated candidate, explanation)
        """
        try:
            # Extract content from response dict if needed
            if isinstance(response, dict):
                content = response.get("content", "").strip()
            else:
                content = str(response).strip()

            if not content:
                raise ValueError("Empty response content")

            # Check if this is a barebone response (just SMILES, no YAML markers)
            # Barebone prompts return only SMILES string
            use_barebone = self.config.get("use_barebone_prompts", True)

            if use_barebone:
                # For barebone prompts, the response should be just a SMILES string
                # Check if it looks like structured YAML/JSON
                if (
                    "molecule:" not in content
                    and "explanation:" not in content
                    and "---" not in content
                ):
                    # Likely just a SMILES string - use it directly
                    molecule = self._sanitize_smiles_value(content)
                    explanation = "Barebone prompt response"
                    return Candidate(representation=molecule), explanation

            # Try to extract molecule and explanation using regex patterns (for detailed prompts)
            molecule_pattern = r"molecule:\s*([^\n\r]+)"
            molecule_match = re.search(molecule_pattern, content, re.IGNORECASE)

            explanation_pattern = r"explanation:\s*(.*?)(?=\n\s*molecule:|$)"
            explanation_match = re.search(
                explanation_pattern, content, re.IGNORECASE | re.DOTALL
            )

            if molecule_match and explanation_match:
                molecule = self._sanitize_smiles_value(molecule_match.group(1))
                explanation = explanation_match.group(1).strip()
                return Candidate(representation=molecule), explanation

            # Fallback: try YAML parsing
            if "---" in content:
                documents = content.split("---")
                parsed = yaml.safe_load(documents[-1].strip())
            else:
                parsed = yaml.safe_load(content)

            if parsed and isinstance(parsed, dict):
                explanation = parsed.get("explanation", "No explanation provided")
                molecule = self._sanitize_smiles_value(parsed.get("molecule", ""))
                if molecule:
                    return Candidate(representation=molecule), explanation

            # Last resort: try to extract SMILES from the content directly
            # This handles cases where LLM returns SMILES with some extra text
            lines = content.split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#") and not ":" in line:
                    # Try to parse as SMILES
                    sanitized = self._sanitize_smiles_value(line)
                    if self._is_valid_smiles(sanitized):
                        return (
                            Candidate(representation=sanitized),
                            "Extracted from response",
                        )

            raise ValueError(f"Could not extract SMILES from response: {content[:200]}")

        except (yaml.YAMLError, KeyError, AttributeError, ValueError) as e:
            raise ValueError(
                f"Could not parse mutation response: {e}. Response: {response}"
            )

    def _mutate(
        self,
        candidate: Candidate,
        objectives: List[Objective],
        return_artifacts: bool = False,
    ):
        """
        Mutate a single candidate using either GB-GA heuristics (non-LLM modes)
        or LLM prompts, depending on configuration.
        """
        if self.manual_genetic_ops:
            mutated_candidate = self._manual_mutation(candidate)
            if return_artifacts:
                return mutated_candidate, "MANUAL_MUTATION", {"content": "manual"}
            return mutated_candidate

        if getattr(self, "mutation_mode", "non_llm") in ("non_llm", "gb_ga"):
            mutated_candidate = self._non_llm_mutation(candidate)
            if return_artifacts:
                return mutated_candidate, "NON_LLM_MUTATION", {"content": "non-llm"}
            return mutated_candidate
        prompt = self._format_mutation_prompt(candidate, objectives)
        start_time = time.time()
        response = self.call_llm_with_prompt(
            prompt, system_prompt=SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT
        )
        elapsed = time.time() - start_time
        self.timing_stats["llm_mutation"]["count"] += 1
        self.timing_stats["llm_mutation"]["total_time"] += elapsed
        self.gen_timing_stats["llm_mutation"]["count"] += 1
        self.gen_timing_stats["llm_mutation"]["total_time"] += elapsed
        mutated_candidate, explanation = self._parse_mutation_response(response)
        if return_artifacts:
            return mutated_candidate, prompt, response
        return mutated_candidate

    async def _mutate_async(
        self,
        candidate: Candidate,
        objectives: List[Objective],
        return_artifacts: bool = False,
    ):
        """
        Async version of mutation for parallel execution.

        Args:
            candidate: The candidate to mutate
            objectives: The objectives to optimize
            return_artifacts: Whether to return prompt/response artifacts

        Returns:
            A mutated candidate (and optionally artifacts)
        """
        if self.manual_genetic_ops:
            mutated_candidate = self._manual_mutation(candidate)
            if return_artifacts:
                return mutated_candidate, "MANUAL_MUTATION", {"content": "manual"}
            return mutated_candidate

        if getattr(self, "mutation_mode", "non_llm") in ("non_llm", "gb_ga"):
            mutated_candidate = self._non_llm_mutation(candidate)
            if return_artifacts:
                return mutated_candidate, "NON_LLM_MUTATION", {"content": "non-llm"}
            return mutated_candidate
        prompt = self._format_mutation_prompt(candidate, objectives)
        start_time = time.time()

        # Use semaphore to control concurrency (prevents overwhelming API rate limits)
        async with self._llm_semaphore:
            response = await self.call_llm_with_prompt_async(
                prompt, system_prompt=SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT
            )

        elapsed = time.time() - start_time
        self.timing_stats["llm_mutation"]["count"] += 1
        self.timing_stats["llm_mutation"]["total_time"] += elapsed
        self.gen_timing_stats["llm_mutation"]["count"] += 1
        self.gen_timing_stats["llm_mutation"]["total_time"] += elapsed
        mutated_candidate, explanation = self._parse_mutation_response(response)
        if return_artifacts:
            return mutated_candidate, prompt, response
        return mutated_candidate

    def _initialize_loggers(
        self,
        objectives: List[Objective],
        iteration_dir: Optional[str],
        logger: Optional[GALogger],
        human_logger: Optional[ChemistLogger],
    ) -> Tuple[GALogger, ChemistLogger, str]:
        """
        Initialize or use provided loggers for optimization tracking.

        Args:
            objectives: List of objectives being optimized
            iteration_dir: Directory for this iteration's outputs
            logger: Optional pre-configured GALogger
            human_logger: Optional pre-configured ChemistLogger

        Returns:
            Tuple of (logger, human_logger, experiment_name)
        """
        # Use simple experiment name without objectives to avoid filename length issues
        experiment_name = "optimization_run"

        # Initialize human_logger if not provided
        if human_logger is None:
            if iteration_dir:
                human_logger_dir = os.path.join(iteration_dir, "per_run")
            else:
                default_run_root = self.config.get("output_dir") or os.path.join(
                    "runs", self.module_id or "llm_sbdd_optimizer"
                )
                human_logger_dir = self.config.get(
                    "human_logger_output_dir"
                ) or os.path.join(default_run_root, "human_logs")
            max_examples = int(self.config.get("human_logger_max_examples", 3))
            human_logger = ChemistLogger(
                experiment_name=experiment_name,
                output_dir=human_logger_dir,
                max_examples_per_generation=max(1, max_examples),
                random_seed=self.seed,
            )

        # Initialize logger if not provided
        if logger is None:
            logger_output_dir = iteration_dir or self.config.get("output_dir") or "logs"
            logger = GALogger(
                objectives=objectives,
                experiment_name=experiment_name,
                output_dir=logger_output_dir,
            )

        return logger, human_logger, experiment_name

    async def _generate_offspring(
        self,
        population: Population,
        objectives: List[Objective],
        capture_examples: bool = False,
        max_examples: int = 3,
    ) -> Tuple[List[Candidate], List[Tuple[str, Dict, str]]]:
        """
        Generate offspring through crossover operations with parallel async LLM calls.

        Args:
            population: Current population to select parents from
            objectives: List of objectives for fitness evaluation
            capture_examples: Whether to capture LLM examples for logging
            max_examples: Maximum number of examples to capture

        Returns:
            Tuple of (offspring candidates, captured examples)
        """
        parent_pairs = [
            self._select_parents(population, objectives, k=2)
            for _ in range(self.offspring_size)
        ]

        # Create all async tasks for parallel execution
        tasks = [
            self._crossover_async(parents, objectives, True) for parents in parent_pairs
        ]

        # Execute all crossovers in parallel with wall-clock timing
        wall_clock_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_clock_elapsed = time.time() - wall_clock_start

        # Track wall-clock time for parallel execution
        num_calls = len([r for r in results if not isinstance(r, Exception)])
        if num_calls > 0:
            self.timing_stats["llm_crossover_wallclock"] = {
                "count": 1,
                "total_time": wall_clock_elapsed,
                "calls_in_batch": num_calls,
            }
            self.gen_timing_stats["llm_crossover_wallclock"] = {
                "count": 1,
                "total_time": wall_clock_elapsed,
                "calls_in_batch": num_calls,
            }

        offsprings = []
        examples = []
        for result in results:
            if isinstance(result, Exception):
                logging.warning(f"Crossover failed: {result}")
                continue
            try:
                offspring, prompt, response = result
                offsprings.append(offspring)

                if capture_examples and len(examples) < max_examples:
                    examples.append((prompt, response, offspring.representation))
            except Exception as e:
                logging.warning(f"Crossover result processing failed: {e}")
                continue

        return offsprings, examples

    async def _generate_mutations(
        self,
        population: Population,
        objectives: List[Objective],
        capture_examples: bool = False,
        max_examples: int = 3,
    ) -> Tuple[List[Candidate], List[Tuple[str, Dict, str]]]:
        """
        Generate mutations from top candidates in the population with parallel async LLM calls.

        Args:
            population: Current population to select candidates from
            objectives: List of objectives for fitness evaluation
            capture_examples: Whether to capture LLM examples for logging
            max_examples: Maximum number of examples to capture

        Returns:
            Tuple of (mutated candidates, captured examples)
        """
        sorted_candidates = sorted(
            population.candidates,
            key=lambda c: self._compute_candidate_score(c, objectives),
            reverse=True,
        )
        assert (
            len(sorted_candidates) >= self.mutation_size
        ), "Population size is less than the mutation size"

        top_candidates = sorted_candidates[: self.mutation_size]
        candidates_to_mutate = [
            random.choice(top_candidates) for _ in range(self.mutation_size)
        ]

        # Create all async tasks for parallel execution
        tasks = [
            self._mutate_async(candidate, objectives, True)
            for candidate in candidates_to_mutate
        ]

        # Execute all mutations in parallel with wall-clock timing
        wall_clock_start = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        wall_clock_elapsed = time.time() - wall_clock_start

        # Track wall-clock time for parallel execution
        num_calls = len([r for r in results if not isinstance(r, Exception)])
        if num_calls > 0:
            self.timing_stats["llm_mutation_wallclock"] = {
                "count": 1,
                "total_time": wall_clock_elapsed,
                "calls_in_batch": num_calls,
            }
            self.gen_timing_stats["llm_mutation_wallclock"] = {
                "count": 1,
                "total_time": wall_clock_elapsed,
                "calls_in_batch": num_calls,
            }

        mutated_candidates = []
        examples = []
        for result in results:
            if isinstance(result, Exception):
                logging.warning(f"Mutation failed: {result}")
                continue
            try:
                mutant, prompt, response = result
                mutated_candidates.append(mutant)

                if capture_examples and len(examples) < max_examples:
                    examples.append((prompt, response, mutant.representation))
            except Exception as e:
                logging.warning(f"Mutation result processing failed: {e}")
                continue

        return mutated_candidates, examples

    def _log_genetic_operations(
        self,
        human_logger: ChemistLogger,
        generation: int,
        new_population_with_scores: Population,
        crossover_only: List[Candidate],
        mutated_candidates: List[Candidate],
        examples_cx: List[Tuple[str, Dict, str]],
        examples_mut: List[Tuple[str, Dict, str]],
        objectives: List[Objective],
    ) -> None:
        """
        Log genetic operation results to the human-readable logger.

        Args:
            human_logger: ChemistLogger instance for human-readable logging
            generation: Current generation number
            new_population_with_scores: Evaluated population with scores
            crossover_only: List of crossover offspring (before mutation added)
            mutated_candidates: List of mutated candidates
            examples_cx: Captured crossover examples (prompt, response, smiles)
            examples_mut: Captured mutation examples (prompt, response, smiles)
            objectives: List of objectives
        """
        try:
            rep_to_candidate = {
                c.representation: c for c in new_population_with_scores.candidates
            }
            cx_reps = {c.representation for c in crossover_only}
            mut_reps = {c.representation for c in mutated_candidates}
            evaluated_crossover = [
                rep_to_candidate[r] for r in cx_reps if r in rep_to_candidate
            ]
            evaluated_mutations = [
                rep_to_candidate[r] for r in mut_reps if r in rep_to_candidate
            ]

            # Log crossover stage
            for c in evaluated_crossover:
                self._compute_candidate_score(c, objectives)
            human_logger.log_population_stage(
                generation + 1, "crossover", evaluated_crossover, objectives
            )

            if examples_cx:
                human_logger.log_llm_examples(
                    generation + 1,
                    "crossover",
                    [
                        LLMExample(
                            kind="crossover",
                            prompt=p,
                            response=str(r.get("content", r)),
                            smiles=s,
                        )
                        for (p, r, s) in examples_cx
                    ],
                )

            # Log mutation stage
            for c in evaluated_mutations:
                self._compute_candidate_score(c, objectives)
            human_logger.log_population_stage(
                generation + 1, "mutation", evaluated_mutations, objectives
            )

            if examples_mut:
                human_logger.log_llm_examples(
                    generation + 1,
                    "mutation",
                    [
                        LLMExample(
                            kind="mutation",
                            prompt=p,
                            response=str(r.get("content", r)),
                            smiles=s,
                        )
                        for (p, r, s) in examples_mut
                    ],
                )
        except Exception as e:
            logging.warning(
                "Failed to record crossover/mutation stages/examples in human_logger: %s",
                e,
                exc_info=True,
            )

    async def optimize(
        self,
        current_population: Optional[Population],
        objectives: List[Objective],
        **additional_kwargs: Dict[str, Any],
    ) -> Population:
        """
        Main optimization method - implement your algorithm here.

        Features:
        - Parallel LLM-guided crossover and mutation operations
        - Early stopping based on mean score of top 100 molecules
        - Progress tracking with detailed statistics
        - Optional logging integration

        Args:
            current_population: Current population of candidates
            objectives: List of objectives with scorer functions
            **additional_kwargs: Additional arguments including logger, human_logger, etc.

        Returns:
            Population of optimized candidates with their scores
        """
        # Extract kwargs
        iteration_dir = additional_kwargs.get("iteration_dir")

        # Initialize loggers
        logger, human_logger, experiment_name = self._initialize_loggers(
            objectives=objectives,
            iteration_dir=iteration_dir,
            logger=additional_kwargs.get("logger"),
            human_logger=additional_kwargs.get("human_logger"),
        )

        # Initialize objective weights from config or set equal weights if missing
        # if getattr(self, "objectives_weights", None) is None:
        #     config_weights = None
        #     if isinstance(self.config, dict):
        #         config_weights = self.config.get("objectives_weights")
        #     if config_weights is not None:
        #         if len(config_weights) != len(objectives):
        #             raise ValueError(
        #                 "Length of objectives_weights in config must equal number of objectives"
        #             )
        #         self.objectives_weights = list(config_weights)
        #     else:
        #         # Default to equal weights for all objectives
        self.objectives_weights = [1.0 for _ in objectives]

        # Initialize population
        if current_population is None:
            current_population = self._create_initial_population()
            self.initial_population = current_population

        # Apply structure filtering if enabled
        if self.enable_structure_filter:
            print("filter wrong moleculars")
            current_population = self._filter_population(current_population)

        # Evaluate initial population and count toward budget
        current_population = await self._evaluate_population_with_timing(
            current_population, objectives, force_evaluation=True
        )
        # Track evaluated SMILES to avoid counting duplicates
        self._evaluated_smiles = {
            c.representation for c in current_population.candidates
        }
        budget_used = len(self._evaluated_smiles)
        generation = 0

        # Attach run context to human logger (if provided)
        if human_logger is not None:
            try:
                human_logger.set_run_context(
                    {
                        "seed": self.seed,
                        "population_size": self.population_size,
                        "offspring_size": self.offspring_size,
                        "mutation_size": self.mutation_size,
                        "oracle_budget": self.oracle_budget,
                        "survival_selection_method": self.survival_selection_method,
                        "survival_tanimoto_threshold": self.survival_tanimoto_threshold,
                        "selection_strategy": getattr(
                            self.selection_strategy,
                            "value",
                            str(self.selection_strategy),
                        ),
                    }
                )
            except Exception as e:
                logging.warning(
                    "Failed to attach run context to human_logger: %s", e, exc_info=True
                )

        # Log initial generation
        logger.log_generation(current_population, generation, budget_used)
        # Compute and print initial diversity
        try:
            init_diversity = calculate_population_diversity(
                [c.representation for c in current_population.candidates]
            )
            print(f"Gen {generation} | Internal diversity: {init_diversity:.4f}")
        except Exception as e:
            logging.warning("Failed to compute initial diversity: %s", e, exc_info=True)
        # Human-friendly logging for initial population
        if human_logger is not None:
            try:
                for c in current_population.candidates:
                    self._compute_candidate_score(c, objectives)
                human_logger.log_population_stage(
                    generation, "original", current_population.candidates, objectives
                )
                human_logger.log_diversity(
                    generation,
                    [c.representation for c in current_population.candidates],
                )
            except Exception as e:
                logging.warning(
                    "Failed to record initial population stage/diversity in human_logger: %s",
                    e,
                    exc_info=True,
                )

        # Initialize early stopping tracking (track top1, top10 mean, top100 mean)
        best_top1_score = self._get_best_score(current_population, objectives)
        best_mean_top10_score = self._get_mean_top_k_score(
            current_population, objectives, top_k=10
        )
        best_mean_top100_score = self._get_mean_top_k_score(
            current_population, objectives, top_k=100
        )
        generations_without_improvement = 0

        # Initialize progress bar
        pbar = tqdm(
            total=self.oracle_budget,
            desc="Optimization Progress",
            initial=budget_used,
            unit="evals",
        )

        # Main optimization loop until budget is exhausted
        while budget_used < self.oracle_budget:

            # 1. Perform crossover (now async with parallel LLM calls)
            capture_examples = human_logger is not None
            offsprings, examples_cx = await self._generate_offspring(
                current_population, objectives, capture_examples
            )

            # 2. Perform mutation (now async with parallel LLM calls)
            mutated_candidates, examples_mut = await self._generate_mutations(
                current_population, objectives, capture_examples
            )

            # Preserve crossover-only list for human logging
            crossover_only = list(offsprings)
            offsprings.extend(mutated_candidates)

            # Safety check: if no offspring generated, stop optimization
            if len(offsprings) == 0:
                pbar.set_description("No offspring generated, stopping")
                break

            # 3. Create new population from offspring
            new_population = Population(candidates=offsprings)

            # Count only NEW unique SMILES toward budget (skip duplicates)
            new_evaluations = 0
            new_smiles_this_gen = set()
            for candidate in new_population.candidates:
                smiles = candidate.representation
                # Only count if: not already evaluated AND not already scored
                if smiles not in self._evaluated_smiles:
                    if objectives[0].name not in candidate.scores:
                        new_evaluations += 1
                        new_smiles_this_gen.add(smiles)

            logging.debug(
                f"New evaluations this generation: {new_evaluations} "
                f"(skipped {len(offsprings) - new_evaluations} duplicates)"
            )
            budget_used += new_evaluations
            # Track newly evaluated SMILES
            self._evaluated_smiles.update(new_smiles_this_gen)

            # 4. Score a population of candidates using a list of objectives
            new_population_with_scores = await self._evaluate_population_with_timing(
                new_population, objectives, force_evaluation=False
            )

            # Log genetic operations to human-friendly logger
            if human_logger is not None:
                self._log_genetic_operations(
                    human_logger=human_logger,
                    generation=generation,
                    new_population_with_scores=new_population_with_scores,
                    crossover_only=crossover_only,
                    mutated_candidates=mutated_candidates,
                    examples_cx=examples_cx,
                    examples_mut=examples_mut,
                    objectives=objectives,
                )

            # Update progress bar
            pbar.update(new_evaluations)

            # 5. Survival selection: combine, deduplicate and select next generation
            current_population = self._select_survivors(
                current_population,
                new_population_with_scores,
                objectives,
            )

            # Ensure survivors (including any backfilled molecules) are evaluated
            # Only count NEW unique SMILES toward budget
            post_selection_evals = 0
            post_selection_new_smiles = set()
            for candidate in current_population.candidates:
                smiles = candidate.representation
                if objectives and objectives[0].name not in candidate.scores:
                    if smiles not in self._evaluated_smiles:
                        post_selection_evals += 1
                        post_selection_new_smiles.add(smiles)
            if post_selection_evals > 0:
                current_population = await self._evaluate_population_with_timing(
                    current_population, objectives, force_evaluation=False
                )
                budget_used += post_selection_evals
                pbar.update(post_selection_evals)
                new_evaluations += post_selection_evals
                self._evaluated_smiles.update(post_selection_new_smiles)

            generation += 1

            # Update early stopping trackers before logging
            should_stop = False
            if self.enable_early_stopping:
                (
                    should_stop,
                    best_top1_score,
                    best_mean_top10_score,
                    best_mean_top100_score,
                    generations_without_improvement,
                ) = self._check_early_stopping(
                    current_population,
                    objectives,
                    best_top1_score,
                    best_mean_top10_score,
                    best_mean_top100_score,
                    generations_without_improvement,
                )

            # Log this generation
            try:
                internal_diversity = calculate_population_diversity(
                    [c.representation for c in current_population.candidates]
                )
                mean_tanimoto = 1.0 - internal_diversity
            except Exception as e:
                logging.warning(
                    "Failed to compute diversity metrics for generation %s: %s",
                    generation,
                    e,
                    exc_info=True,
                )
                internal_diversity = None
                mean_tanimoto = None

            additional_metrics = {
                "budget_used": budget_used,
                "new_evaluations": new_evaluations,
                "generations_without_improvement": generations_without_improvement,
                "internal_diversity": internal_diversity,
                "mean_tanimoto": mean_tanimoto,
            }
            logger.log_generation(
                current_population, generation, new_evaluations, additional_metrics
            )

            # Update progress bar description with current stats including diversity
            best_score = self._get_best_score(current_population, objectives)
            top_10_avg = self._get_mean_top_k_score(
                current_population, objectives, top_k=10
            )
            early_stop_info = (
                f" | ES: {generations_without_improvement}/{self.early_stopping_patience}"
                if generations_without_improvement > 0
                else ""
            )
            div_str = (
                f" | Div: {internal_diversity:.4f}"
                if internal_diversity is not None
                else ""
            )
            pbar.set_description(
                f"Gen {generation} | Best: {best_score:.4f} | Top10Avg: {top_10_avg:.4f}{div_str}{early_stop_info}"
            )
            if should_stop and self.enable_early_stopping:
                pbar.set_description(
                    f"Early stopping: no improvement for {self.early_stopping_patience} generations"
                )
                break

            # Print timing stats for this generation
            self._print_timing_stats(generation)

            # Log timing stats to human logger
            if human_logger is not None:
                try:
                    timing_dict = {}
                    for op_name, stats in self.gen_timing_stats.items():
                        if stats["count"] > 0:
                            timing_dict[op_name] = {
                                "count": stats["count"],
                                "total_time": stats["total_time"],
                                "avg_time": stats["total_time"] / stats["count"],
                            }
                    if timing_dict:
                        human_logger.log_timing(generation, timing_dict)
                except Exception as e:
                    logging.warning(
                        "Failed to log timing stats to human_logger: %s",
                        e,
                        exc_info=True,
                    )

            # Reset generation-specific stats
            self.gen_timing_stats = defaultdict(lambda: {"count": 0, "total_time": 0.0})

        # Close progress bar
        pbar.close()

        # Print final timing summary
        print("\n" + "=" * 80)
        print("FINAL TIMING SUMMARY")
        print("=" * 80)
        self._print_timing_stats(None, final=True)

        # Save optimization log
        log_files = logger.save_log()
        print(f"\nOptimization completed. Logs saved:")
        for log_type, path in log_files.items():
            print(f"  - {log_type}: {path}")

        # Save human-readable report if enabled
        if human_logger is not None:
            try:
                paths = human_logger.save_human_report()
                print(f"Human-readable report saved: {paths['report']}")
            except Exception as e:
                logging.warning(
                    "Failed to save human-readable report: %s", e, exc_info=True
                )

        return current_population

    def _get_best_score(
        self, population: Population, objectives: List[Objective]
    ) -> float:
        """Get the best score from the current population."""
        if not population.candidates:
            return 0.0

        scores = [
            self._compute_candidate_score(candidate, objectives)
            for candidate in population.candidates
        ]
        return max(scores) if scores else 0.0

    def _get_top_scores(
        self, population: Population, objectives: List[Objective], top_k: int = 10
    ) -> List[float]:
        """Get the top k scores from the current population."""
        if not population.candidates:
            return []

        scores = [
            self._compute_candidate_score(candidate, objectives)
            for candidate in population.candidates
        ]

        # Sort scores in descending order and take top k
        sorted_scores = sorted(scores, reverse=True)
        return sorted_scores[: min(top_k, len(sorted_scores))]

    def _get_mean_top_k_score(
        self, population: Population, objectives: List[Objective], top_k: int = 100
    ) -> float:
        """Get the mean score of top k molecules for early stopping."""
        if not population.candidates:
            return 0.0

        scores = [
            self._compute_candidate_score(candidate, objectives)
            for candidate in population.candidates
        ]

        # Sort scores in descending order and take top k
        sorted_scores = sorted(scores, reverse=True)
        top_k_scores = sorted_scores[: min(top_k, len(sorted_scores))]

        return sum(top_k_scores) / len(top_k_scores) if top_k_scores else 0.0

    def _check_early_stopping(
        self,
        population: Population,
        objectives: List[Objective],
        best_top1_score: float,
        best_mean_top10_score: float,
        best_mean_top100_score: float,
        generations_without_improvement: int,
    ) -> Tuple[bool, float, float, float, int]:
        """
        Check if early stopping criteria are met across top1, top10 mean, and top100 mean.

        Args:
            population: Current population
            objectives: List of objectives
            best_top1_score: Best top1 score seen so far
            best_mean_top10_score: Best top10 mean seen so far
            best_mean_top100_score: Best top100 mean seen so far
            generations_without_improvement: Count of generations without improvement across all tracked metrics

        Returns:
            Tuple of (should_stop, updated_best_top1, updated_best_mean_top10, updated_best_mean_top100, updated_generations_without_improvement)
        """
        current_top1 = self._get_best_score(population, objectives)
        current_top10_mean = self._get_mean_top_k_score(
            population, objectives, top_k=10
        )
        current_top100_mean = self._get_mean_top_k_score(
            population, objectives, top_k=100
        )

        improved_top1 = current_top1 > best_top1_score + self.early_stopping_threshold
        improved_top10 = (
            current_top10_mean > best_mean_top10_score + self.early_stopping_threshold
        )
        improved_top100 = (
            current_top100_mean > best_mean_top100_score + self.early_stopping_threshold
        )

        if improved_top1 or improved_top10 or improved_top100:
            # Improvement in any tracked metric resets counter and updates respective bests
            return (
                False,
                max(best_top1_score, current_top1),
                max(best_mean_top10_score, current_top10_mean),
                max(best_mean_top100_score, current_top100_mean),
                0,
            )
        else:
            # No improvement across all metrics, increment counter
            new_generations_without_improvement = generations_without_improvement + 1
            should_stop = (
                new_generations_without_improvement >= self.early_stopping_patience
            )
            return (
                should_stop,
                best_top1_score,
                best_mean_top10_score,
                best_mean_top100_score,
                new_generations_without_improvement,
            )

    def _remove_duplicates(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove duplicate candidates based on their representation (SMILES)."""
        seen_representations = set()
        unique_candidates = []

        for candidate in candidates:
            if candidate.representation not in seen_representations:
                seen_representations.add(candidate.representation)
                unique_candidates.append(candidate)

        return unique_candidates

    # --- Survival selection strategies ---
    def _build_survival_selector(self, method: str):
        common_kwargs = dict(
            population_size=self.population_size,
            elitism_fraction=self.elitism_fraction,
            survival_tanimoto_threshold=self.survival_tanimoto_threshold,
            seed=self.seed,
            init_group=self.init_group,
            remove_duplicates=self._remove_duplicates,
            compute_candidate_score=self._compute_candidate_score,
            get_elite_candidates=self._get_elite_candidates,
            sample_smiles=self._sample_smiles,
            sanitize_smiles_value=self._sanitize_smiles_value,
            elitism_fields=self.elitism_fields,
        )
        if method == "diverse_top":
            return DiverseTopSurvivalSelection(**common_kwargs)
        if method == "butina_cluster":
            return ButinaClusterSurvivalSelection(**common_kwargs)
        return FitnessSurvivalSelection(**common_kwargs)

    # --- Objective combiner factory ---
    def _build_combiner(self, name_or_path: str) -> ObjectiveCombiner:
        """
        name_or_path:
          - 'simple_product' (default - multiplicative combination)
          - 'simple_sum' (additive combination)
          - 'weighted_sum'
          - 'antibiotic_geomean' (hard-coded domain combiner)
          - 'covid_simple' (Mpro-focused weighted sum with PAINS/Brenk gates)
          - 'pkg.module:ClassName' (must subclass ObjectiveCombiner)
        """
        if name_or_path == "simple_product":
            return SimpleProductCombiner()
        if name_or_path == "simple_sum":
            return SimpleSumCombiner()
        if name_or_path == "weighted_sum":
            return WeightedSumCombiner()
        if name_or_path == "antibiotic_geomean":
            from modules.small_molecule_drug_design.combiner.antibiotic_combiner import (
                AntibioticGeoMeanCombiner,
            )

            return AntibioticGeoMeanCombiner()
        if name_or_path == "covid_simple":
            from modules.small_molecule_drug_design.combiner.covid_simple_combiner import (
                CovidSimpleCombiner,
            )

            return CovidSimpleCombiner()
        if ":" in name_or_path:
            mod_path, cls_name = name_or_path.split(":", 1)
            import importlib

            mod = importlib.import_module(mod_path)
            cls = getattr(mod, cls_name)
            if not issubclass(cls, ObjectiveCombiner):
                raise TypeError(f"{cls} must subclass ObjectiveCombiner")
            return cls()
        raise ValueError(f"Unknown objective_combiner: {name_or_path}")

    def _select_survivors(
        self,
        current_population: Population,
        new_population_with_scores: Population,
        objectives: List[Objective],
    ) -> Population:
        """Delegate survivor selection to the configured strategy object."""
        return self.survival_selector.select(
            current_population, new_population_with_scores, objectives
        )

    # --- Helper methods for elitism and selection ---
    async def _evaluate_population_with_timing(
        self,
        population: Population,
        objectives: List[Objective],
        force_evaluation: bool = False,
    ) -> Population:
        """
        Evaluate population with per-objective timing tracking.

        Args:
            population: Population to evaluate
            objectives: List of objectives
            force_evaluation: Whether to force re-evaluation

        Returns:
            Evaluated population
        """
        if population is None or population.is_empty:
            raise ValueError("Population is None or empty for evaluation.")

        num_compounds = len(population.candidates)
        total_time = 0.0

        # Track timing per objective
        for objective in objectives:
            start_time = time.time()
            population = await objective.score(
                population, force_evaluation=force_evaluation
            )
            elapsed = time.time() - start_time
            total_time += elapsed

            # Track per-objective timing
            obj_key = f"scoring_{objective.name}"
            self.timing_stats[obj_key]["count"] += num_compounds
            self.timing_stats[obj_key]["total_time"] += elapsed
            self.gen_timing_stats[obj_key]["count"] += num_compounds
            self.gen_timing_stats[obj_key]["total_time"] += elapsed

        # Track total scoring time (all objectives combined)
        self.timing_stats["scoring"]["count"] += num_compounds
        self.timing_stats["scoring"]["total_time"] += total_time
        self.gen_timing_stats["scoring"]["count"] += num_compounds
        self.gen_timing_stats["scoring"]["total_time"] += total_time

        return population

    def _print_timing_stats(
        self, generation: Optional[int] = None, final: bool = False
    ):
        """Print timing statistics for different operations."""
        if final:
            prefix = "Total"
            stats = self.timing_stats
        elif generation is not None:
            prefix = f"Gen {generation}"
            stats = self.gen_timing_stats
        else:
            prefix = "Current"
            stats = self.timing_stats

        print(f"\n{prefix} Timing Statistics:")
        print("-" * 80)

        # LLM crossover stats
        if stats["llm_crossover"]["count"] > 0:
            cx_count = stats["llm_crossover"]["count"]
            cx_time = stats["llm_crossover"]["total_time"]
            avg_time = cx_time / cx_count if cx_count > 0 else 0
            print(
                f"  LLM Crossover: {cx_count} calls took {cx_time:.2f}s (avg: {avg_time:.3f}s/call)"
            )

            # Show wall-clock time if available (parallel execution time)
            if "llm_crossover_wallclock" in stats:
                wc_stats = stats["llm_crossover_wallclock"]
                wc_time = wc_stats["total_time"]
                wc_calls = wc_stats.get("calls_in_batch", cx_count)
                speedup = cx_time / wc_time if wc_time > 0 else 1.0
                print(
                    f"    → Wall-clock time: {wc_time:.2f}s for {wc_calls} calls (parallel speedup: {speedup:.2f}x)"
                )

        # LLM mutation stats
        if stats["llm_mutation"]["count"] > 0:
            mut_count = stats["llm_mutation"]["count"]
            mut_time = stats["llm_mutation"]["total_time"]
            avg_time = mut_time / mut_count if mut_count > 0 else 0
            print(
                f"  LLM Mutation: {mut_count} calls took {mut_time:.2f}s (avg: {avg_time:.3f}s/call)"
            )

            # Show wall-clock time if available (parallel execution time)
            if "llm_mutation_wallclock" in stats:
                wc_stats = stats["llm_mutation_wallclock"]
                wc_time = wc_stats["total_time"]
                wc_calls = wc_stats.get("calls_in_batch", mut_count)
                speedup = mut_time / wc_time if wc_time > 0 else 1.0
                print(
                    f"    → Wall-clock time: {wc_time:.2f}s for {wc_calls} calls (parallel speedup: {speedup:.2f}x)"
                )

        # Scoring stats - total
        if stats["scoring"]["count"] > 0:
            score_count = stats["scoring"]["count"]
            score_time = stats["scoring"]["total_time"]
            avg_time = score_time / score_count if score_count > 0 else 0
            print(
                f"  Scoring (all objectives): {score_count} compounds took {score_time:.2f}s (avg: {avg_time:.4f}s/compound)"
            )

        # Scoring stats - per objective
        scoring_keys = [
            k for k in stats.keys() if k.startswith("scoring_") and k != "scoring"
        ]
        if scoring_keys:
            print(f"  Scoring per objective:")
            for obj_key in sorted(scoring_keys):
                obj_name = obj_key.replace("scoring_", "")
                obj_stats = stats[obj_key]
                if obj_stats["count"] > 0:
                    obj_count = obj_stats["count"]
                    obj_time = obj_stats["total_time"]
                    obj_avg = obj_time / obj_count if obj_count > 0 else 0
                    print(
                        f"    - {obj_name}: {obj_count} compounds, {obj_time:.2f}s total (avg: {obj_avg:.4f}s/compound)"
                    )

        # Total LLM time
        total_llm_time = (
            stats["llm_crossover"]["total_time"] + stats["llm_mutation"]["total_time"]
        )
        if total_llm_time > 0:
            print(f"  Total LLM time: {total_llm_time:.2f}s")

        print("-" * 80)

    def _get_elite_candidates(
        self,
        current_population: Population,
        objectives: List[Objective],
        elite_count: int,
        elitism_fields: Optional[List[str]] = None,
    ) -> List[Candidate]:
        if elite_count <= 0 or not current_population.candidates:
            return []
        fields = [field for field in (elitism_fields or []) if field]
        objective_by_name = {obj.name: obj for obj in objectives or []}

        def field_value(candidate: Candidate, field_name: str) -> float:
            score = candidate.scores.get(field_name)
            if score is None:
                return float("-inf")
            try:
                value = float(score)
            except (TypeError, ValueError):
                value = 1.0 if bool(score) else 0.0
            objective = objective_by_name.get(field_name)
            if objective and objective.optimization_direction == "minimize":
                value = -value
            return value

        def build_key(candidate: Candidate) -> tuple:
            field_scores = [field_value(candidate, name) for name in fields]
            aggregate = self._compute_candidate_score(candidate, objectives)
            field_scores.append(aggregate)
            return tuple(field_scores)

        sorted_current = sorted(
            current_population.candidates,
            key=build_key,
            reverse=True,
        )
        return sorted_current[:elite_count]
