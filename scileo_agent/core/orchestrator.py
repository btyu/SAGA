"""
Main orchestration module for the SciLeo Agent framework.

This module contains the OptimizationOrchestrator class that coordinates
all modules in the optimization workflow:

1. Initialization: Analyzes initial population (if provided)
2. Planning: Decides objectives for the next iteration
3. Scorer Creation: Creates/retrieves scoring functions for objectives
4. Optimization: Runs optimization algorithms until convergence
5. Analysis: Analyzes results and generates comprehensive reports
6. Termination Decision: Decides whether to continue or terminate

The orchestrator manages the complete optimization lifecycle and coordinates
communication between all framework modules.
"""

from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
import json
import random

from .config import FrameworkConfig
from .data_models import Candidate, Population, Objective
from .data_models.results import OptimizationResult
from .modules.base import BaseModule
from .registry.module_registry import get_module_class
from ..utils.logging import setup_logging, get_logger
from .registry.serializer_registry import get_serializer
from .registry.scorer_registry import list_scorers, get_scorer_metadata, ScorerManager
from .run_tracker import RunTracker


class OptimizationOrchestrator:
    """
    Main orchestrator for the optimization workflow.
    
    This class coordinates all modules according to the new workflow:
    1. Initialization with analysis of initial population
    2. Planning phase - decide objectives for next iteration
    3. Scorer creation phase - create/collect scorers
    4. Optimization phase - run optimization until convergence
    5. Analysis phase - analyze results and generate reports
    6. Decision to continue or terminate
    """
    
    def __init__(
        self,
        config: FrameworkConfig,
        run_name: str,
        run_id: Optional[str] = None
    ):
        """
        Initialize the orchestrator.

        Args:
            config: Framework configuration
        """
        self.__already_run = False

        self.config = config
        self.run_name = run_name
        if run_id is None:
            run_id = f"{run_name}-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        self.run_id = run_id

        run_dir = Path(f"runs/{run_id}")
        log_dir = run_dir / "logs"
        output_dir = run_dir / "outputs"

        self.run_dir = run_dir
        self.log_dir = log_dir
        self.output_dir = output_dir

        log_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get existing logger or create new one, preserving the current level
        existing_logger = get_logger()
        current_level = existing_logger.level if existing_logger else "INFO"
        self.logger = setup_logging(level=current_level, log_dir=str(log_dir))

        # Auto-create modules from config using registry
        self.planner = self._create_module_from_config("planner")
        self.scorer_creator = self._create_module_from_config("scorer_creator")
        self.optimizer = self._create_module_from_config("optimizer")
        self.analyzer = self._create_module_from_config("analyzer")
        self.knowledge_manager = self._create_module_from_config("knowledge_manager")
        
        # Initialize optimization result
        self.result = OptimizationResult(
            run_id=run_id,
            run_name=run_name,
            config=config.model_dump()
        )
        
        # Initialize state
        self.current_iteration = 0
        self.is_running = False
        self.current_population: Optional[Population] = None
        self.current_objectives: List[Objective] = []
        self.current_analysis_report: Optional[str] = None
        self.serializer_name: Optional[str] = None

        # Initialize run tracker
        self.run_tracker = RunTracker(
            run_id=run_id,
            run_name=run_name,
            output_dir=output_dir
        )

        # Configure scorer manager based on loop_config
        run_scorers_in_docker = config.loop_config.get("run_scorers_in_docker", True)
        scorer_manager = ScorerManager(run_in_docker=run_scorers_in_docker)
        scorer_manager.set_run_in_docker(run_scorers_in_docker)

        # Log module configurations
        self._log_module_configurations()

    def _create_module_from_config(self, module_type: str, **kwargs) -> BaseModule:
        """
        Create a module instance from configuration using the registry.

        Args:
            module_type: Type of module to create (e.g., 'planner', 'optimizer')

        Returns:
            Instantiated module

        Raises:
            ValueError: If module configuration is invalid or module class not found
        """
        # Get module configurations for this type
        module_configs = self.config.get_module_configs_by_type(module_type)
        if len(module_configs) == 0:
            raise ValueError(f"No module configurations found for module type: {module_type}")
        elif len(module_configs) > 1:
            raise ValueError(f"Multiple module configurations found for module type: {module_type}")

        module_config = module_configs[0]

        # Get module class from registry
        module_class = get_module_class(
            module_config.module_type,
            module_config.module_name,
            module_config.module_version
        )

        if module_class is None:
            raise ValueError(
                f"Module class not found in registry for {module_type}: "
                f"name={module_config.module_name}, version={module_config.module_version}"
            )

        # Instantiate module
        config = {}
        config.update(module_config.config)
        config.update(kwargs)
        module = module_class(
            module_id=module_config.module_id,
            config=config,
            llm_config=module_config.llm_config
        )

        return module

    async def run(
        self,
        high_level_goal: str,
        context_information: Optional[str] = None,
        serializer_name: Optional[str] = None,
        initial_objectives: Optional[List[Objective]] = None,
        initial_population: Optional[Population] = None,
    ) -> OptimizationResult:
        """
        Run the main optimization loop according to the new workflow (async).

        Args:
            high_level_goal: The high-level goal for optimization
            run_name: Name of the optimization run
            context_information: Additional context information
            serializer_name: Optional serializer name, needed for the scorer creator to create a new scorer
            initial_objectives: Optional initial list of objectives
            initial_population: Optional initial population of candidates
            run_id: Optional run ID

        Returns:
            OptimizationResult containing the final results
        """
        if self.__already_run:
            raise RuntimeError("This OptimizationOrchestrator instance has already been run. Please create a new instance for another run.")

        self.__already_run = True

        try:
            self.is_running = True
            self.result.start_time = datetime.now()
            max_iters = self.config.loop_config.get("max_iterations", 10)

            self.result.high_level_goal = high_level_goal
            self.result.context_information = context_information
            self.serializer_name = serializer_name

            # Track run inputs and configuration
            self.run_tracker.set_inputs(
                high_level_goal=high_level_goal,
                context_information=context_information,
                serializer_name=serializer_name,
                initial_objectives=initial_objectives,
                initial_population=initial_population,
                max_iterations=max_iters
            )
            self.run_tracker.set_configuration(self.config)
            self.run_tracker.save()

            # Log run start
            self.logger.info("=" * 80)
            self.logger.info("=" * 80)
            self.logger.info(f"{'OPTIMIZATION RUN START':^80}")
            self.logger.info("=" * 80)
            self.logger.info("=" * 80)
            self.logger.info(f"Run ID: {self.run_id}")
            self.logger.info(f"Goal: {high_level_goal}")
            if context_information:
                self.logger.info(f"Context: {context_information}")
            self.logger.debug(f"Loop configuration: {self.config.loop_config}")
            
            # Step 0: Initialization with analysis
            analysis_report = self._initialization_phase(initial_objectives, initial_population)

            # Save results after initialization
            self._save_intermediate_results()

            # Main optimization loop
            early_stop = False
            for iteration in range(1, max_iters + 1):
                self.logger.info("=" * 80)
                self.logger.info(f"{'ITERATION ' + str(iteration):^80}")
                self.logger.info("=" * 80)
                self.current_iteration = iteration

                # Start tracking this iteration
                self.run_tracker.start_iteration(iteration)

                # Step 1: Planning - decide objectives for this iteration
                planning_result = await self._planning_phase(high_level_goal, context_information=context_information, initial_objectives=initial_objectives, analysis_report=analysis_report)
                self._save_intermediate_results()  # Save after planning

                # Step 2: Scorer Creation - create/collect scorers
                objectives_with_scorers = await self._scorer_creation_phase(planning_result, serializer_name=serializer_name)
                self._save_intermediate_results()  # Save after scorer creation

                # Step 3: Optimization - run optimization until convergence
                optimized_population, random_replacement_info = await self._optimization_phase(self.current_population, objectives_with_scorers, high_level_goal)
                self._save_intermediate_results()  # Save after optimization

                # Step 4: Analysis - analyze results and generate reports
                analysis_result = await self._analysis_phase(
                    high_level_goal,
                    context_information,
                    optimized_population,
                    objectives_with_scorers,
                    historical_info=self.knowledge_manager.get_historical_summary(self.current_iteration),
                    random_replacement_info=random_replacement_info
                )
                analysis_report = analysis_result["analysis_report"]
                self._save_intermediate_results()  # Save after analysis

                # End tracking this iteration
                self.run_tracker.end_iteration(iteration)
                self.run_tracker.save()

                # Save intermediate results after each iteration
                self._save_intermediate_results()

                # Stop all MCP scorer servers to free memory
                self.logger.debug("Stopping MCP scorer servers to free memory")
                scorer_manager = ScorerManager(run_in_docker=self.config.loop_config.get("run_scorers_in_docker", True))
                scorer_manager.stop_all_mcp_servers()

                if analysis_result["should_stop"]:
                    self.result.termination_reason = "early_stop"
                    self.logger.info(f"Terminating optimization after iteration {iteration}")
                    self.logger.info(f"Termination reasoning: {analysis_result['reasoning']}")
                    early_stop = True
                    break

                # Update result
                self.result.total_generations = iteration
            
            # Natural termination
            if not early_stop:
                self.result.termination_reason = "max_iterations"
                self.logger.info(f"Optimization completed after {max_iters} iterations")
            
            # Mark as completed
            self.result.status = "completed"

            # Finalize results
            await self._finalize_optimization()

            # Log run end
            self.logger.info("=" * 80)
            self.logger.info("=" * 80)
            self.logger.info(f"{'OPTIMIZATION RUN COMPLETED':^80}")
            self.logger.info("=" * 80)
            self.logger.info("=" * 80)

        except KeyboardInterrupt:
            self.logger.warning("Optimization interrupted by user")
            # Save intermediate results before stopping
            try:
                self._save_intermediate_results()
            except Exception as save_error:
                self.logger.error(f"Failed to save results during interrupt: {str(save_error)}")
            await self._stop_optimization()

        except Exception as e:
            self.logger.critical(f"Optimization failed with unrecoverable error: {str(e)}")

            self.result.status = "failed"
            self.result.termination_reason = "error"
            self.result.error_message = str(e)
            self.result.end_time = datetime.now()

            # Track the error
            self.run_tracker.set_outputs(
                termination_reason="error",
                total_iterations=self.current_iteration,
                final_population=self.current_population,
                final_analysis_report=self.current_analysis_report,
                all_candidates_population=None,
                error_message=str(e)
            )
            self.run_tracker.finalize("failed")
            self.run_tracker.save()

            # Save intermediate results before finalizing
            try:
                self._save_intermediate_results()
            except Exception as save_error:
                self.logger.error(f"Failed to save results during error handling: {str(save_error)}")

            await self._finalize_optimization()
            raise
        finally:
            self.is_running = False

        return self.result
    
    def _initialization_phase(
        self,
        initial_objectives: Optional[List[Objective]] = None,
        initial_population: Optional[Population] = None
    ) -> Optional[str]:
        """
        Initialize the optimization with analysis of initial population.

        Args:
            initial_objectives: Initial list of objectives
            initial_candidates: Optional initial population of candidates
        """
        self.logger.info("-" * 80)
        self.logger.info(f"{'INITIALIZATION PHASE':^80}")
        self.logger.info("-" * 80)

        # Check if initial population is required for the optimizer
        if self.optimizer.requires_initial_population and initial_population is None:
            self.logger.error("Initial population is required but not provided")
            raise ValueError("Initial population is required for the optimizer")

        # Store initial objectives
        self.knowledge_manager.store_objectives(initial_objectives, self.current_iteration)
        if initial_objectives:
            # Build objective details string
            objectives_lines = [f"Received {len(initial_objectives)} initial objectives:"]
            for idx, obj in enumerate(initial_objectives, 1):
                # Format the type with weight information
                type_str = f'type="{obj.type}"'
                weight_str = f"weight={obj.weight}" if obj.weight is not None else "weight=None"

                # Format optimization direction (filters don't have direction)
                if obj.type == "filter":
                    direction_str = "filter"
                else:
                    direction_str = obj.optimization_direction if obj.optimization_direction else "no direction"

                # Add objective header and description
                objectives_lines.append(f"   {idx}. {obj.name} ({type_str}, {weight_str}, {direction_str})")
                objectives_lines.append(f"      Description: {obj.description}")

            # Print all objective details in a single log call
            self.logger.info("\n".join(objectives_lines))
        else:
            self.logger.info("No initial objectives provided")

        # Store initial population
        analysis_report = None
        if initial_population:
            self.logger.info(f"Received initial population with {initial_population.size} candidates")
            self.knowledge_manager.store_population(initial_population, self.current_iteration)
            self.current_population = initial_population
        else:
            self.logger.info("No initial population provided")
            self.current_population = None

        # TODO: Print out the scorer library details

        # Track initialization phase
        self.run_tracker.add_phase(
            iteration_number=0,
            phase_name="initialization",
            phase_data={
                "inputs": {
                    "initial_objectives_count": len(initial_objectives) if initial_objectives else 0,
                    "initial_population_size": initial_population.size if initial_population else 0
                },
                "outputs": {
                    "stored_objectives_count": len(initial_objectives) if initial_objectives else 0,
                    "stored_population_size": self.current_population.size if self.current_population else 0,
                    "analysis_report_generated": analysis_report is not None
                }
            }
        )
        self.run_tracker.save()

        return analysis_report
    
    async def _planning_phase(
        self,
        high_level_goal: str,
        context_information: Optional[str]=None,
        initial_objectives: Optional[List[Objective]]=None,
        analysis_report: Optional[str]=None,
        mode: Optional[str]="normal",
        additional_information: Optional[Dict[str, Any]]=None
    ) -> Dict[str, Any]:
        """
        Plan objectives for the current iteration.

        Args:
            initial_objectives: Initial list of objectives

        Returns:
            List of planned objectives for this iteration
        """
        if mode == "normal":
            self.logger.info("-" * 80)
            self.logger.info(f"{'PLANNING PHASE':^80}")
            self.logger.info("-" * 80)
        else:
            self.logger.info(f"--- Planning Phase (Retry Mode) ---")

        # Log input context
        self.logger.debug(f"Planning phase mode: {mode}, iteration: {self.current_iteration}, "
                         f"has_analysis_report: {analysis_report is not None}")

        planning_result = await self.planner.process(
            iteration_number=self.current_iteration,
            high_level_goal=high_level_goal,
            context_information=context_information,
            initial_objectives=initial_objectives,
            analysis_report=analysis_report,
            mode=mode,
            additional_information=additional_information
        )

        # TODO: Make sure in the planner that it doesn't propose objectives with duplicate names

        # Store planned objectives
        self.knowledge_manager.store_objectives(planning_result["objectives"], self.current_iteration)
        self.current_objectives = planning_result["objectives"]

        # Log planned objectives
        objectives_lines = [f"Planned {len(planning_result['objectives'])} objective(s):"]
        for i, obj in enumerate(planning_result["objectives"], 1):
            # Format the type with weight information
            type_str = f'type="{obj.type}"'
            weight_str = f"weight={obj.weight}" if obj.weight is not None else "weight=None"

            # Format optimization direction (filters don't have direction)
            if obj.type == "filter":
                direction_str = "filter"
            else:
                direction_str = obj.optimization_direction if obj.optimization_direction else "no direction"

            # Add objective header and description
            objectives_lines.append(f"   {i}. {obj.name} ({type_str}, {weight_str}, {direction_str})")
            objectives_lines.append(f"      Description: {obj.description}")

        # Print all objective details in a single log call
        self.logger.info("\n".join(objectives_lines))

        # Log detailed output context
        self.logger.debug(f"Planning completed with {len(planning_result['objectives'])} objectives")

        # Track planning phase
        self.run_tracker.add_phase(
            iteration_number=self.current_iteration,
            phase_name="planning" if mode == "normal" else "planning_retry",
            phase_data={
                "inputs": {
                    "mode": mode,
                    "has_analysis_report": analysis_report is not None,
                    "has_additional_information": additional_information is not None
                },
                "outputs": {
                    "planned_objectives": [
                        {
                            "name": obj.name,
                            "description": obj.description,
                            "optimization_direction": obj.optimization_direction,
                            "weight": obj.weight,
                            "type": obj.type
                        }
                        for obj in planning_result["objectives"]
                    ],
                    "objectives_count": len(planning_result["objectives"]),
                    "original_output": planning_result,
                }
            }
        )
        self.run_tracker.save()

        return planning_result
    
    async def _scorer_creation_phase(self, planning_result: Dict[str, Any], serializer_name: Optional[str] = None) -> List[Objective]:
        """
        Create or collect scorers for the given objectives.

        Args:
            objectives: List of objectives needing scorers
            serializer_name: Optional serializer name, needed for the scorer creator to create a new scorer

        Returns:
            List of objectives with scorer functions attached
        """
        self.logger.info("-" * 80)
        self.logger.info(f"{'SCORER CREATION PHASE':^80}")
        self.logger.info("-" * 80)

        objectives = planning_result["objectives"]

        # Log objective details
        objectives_lines = [f"Matching {len(objectives)} objective(s) with available scorers"]
        for i, obj in enumerate(objectives, 1):
            objectives_lines.append(f"  Objective {i}: '{obj.name}' (type: {obj.type})")
        self.logger.info("\n".join(objectives_lines))

        test_candidates = None
        # Randomly select some candidates from the current population for testing scorers
        if self.current_population and self.current_population.size > 0:
            num_test_candidates = min(10, self.current_population.size)
            test_candidates = random.sample(self.current_population.candidates, num_test_candidates)
            self.logger.debug(f"Selected {num_test_candidates} candidate(s) from current population for scorer testing")
        
        self.logger.debug("--- Inside Scorer Creator ---")
        # TODO: Make the logging inside the scorer creator better, sequential for each objectives
        scorer_creation_result = await self.scorer_creator.process(objectives, serializer_name=serializer_name, test_candidates=test_candidates)
        self.logger.debug("-------------")

        # Log scorer creation results
        matched_count = len(scorer_creation_result["matched_objectives"])
        unmatched_count = len(scorer_creation_result["unmatched_objectives"])
        available_count = len(scorer_creation_result["available_objectives"])

        self.logger.info(f"Matching results: {matched_count} matched, {unmatched_count} unmatched")
        self.logger.debug(f"Available scorers in the scorer library: {available_count}")

        max_objective_planning_retries = self.config.loop_config.get("max_objective_planning_retries", 3)
        if unmatched_count > 0:
            self.logger.warning(f"{unmatched_count} unmatched objective(s) - retry required")
            for obj in scorer_creation_result["unmatched_objectives"]:
                self.logger.warning(f"  ✗ '{obj.name}'")
            self.logger.debug(f"Maximum planning retries allowed: {max_objective_planning_retries}")
        
        # Initialize objectives_with_scorers from the initial result
        objectives_with_scorers = scorer_creation_result["matched_objectives"]
        all_objectives_matched = len(scorer_creation_result["unmatched_objectives"]) == 0
        count = 0
        
        while not all_objectives_matched:
            count += 1
            self.logger.info(f"Retry {count}/{max_objective_planning_retries}: Revising objectives to match available scorers")

            if count >= max_objective_planning_retries:
                unmatched_names = [obj.name for obj in scorer_creation_result["unmatched_objectives"]]
                self.logger.error(f"Failed to match all objectives after {max_objective_planning_retries} retries")
                self.logger.error(f"Still unmatched: {', '.join(unmatched_names)}")
                raise RuntimeError(f"Planning retry failed after {max_objective_planning_retries} retries, as there are still unmatched objectives")

            # Prepare additional information for retry
            additional_information = {
                **scorer_creation_result
            }

            # Revise objectives via planning
            planning_result = await self._planning_phase(
                high_level_goal=None,
                context_information=None,
                initial_objectives=None,
                analysis_report=None,
                mode="retry",
                additional_information=additional_information
            )

            revised_objectives = planning_result["objectives"]
            self.logger.debug(f"Planner proposed {len(revised_objectives)} revised objectives")

            # Try matching again
            scorer_creation_result = await self.scorer_creator.process(planning_result["objectives"], serializer_name=serializer_name)

            # Log retry results
            new_matched_count = len(scorer_creation_result["matched_objectives"])
            new_unmatched_count = len(scorer_creation_result["unmatched_objectives"])

            self.logger.info(f"Retry {count} results: {new_matched_count} matched, {new_unmatched_count} unmatched")

            if new_unmatched_count > 0:
                for obj in scorer_creation_result["unmatched_objectives"]:
                    self.logger.warning(f"  ✗ Still unmatched: '{obj.name}'")

            objectives_with_scorers = scorer_creation_result["matched_objectives"]
            all_objectives_matched = len(scorer_creation_result["unmatched_objectives"]) == 0

            if all_objectives_matched:
                self.logger.info(f"✓ All objectives matched after {count} retry(ies)")

        # self.logger.info(f"Scorer creation completed: {len(objectives_with_scorers)} scorer(s) ready")

        # # Log detailed output
        # for i, obj in enumerate(objectives_with_scorers, 1):
        #     self.logger.debug(f"  {i}. {obj.name} ({obj.optimization_direction}, weight={obj.weight})")

        self.knowledge_manager.store_objectives(objectives_with_scorers, self.current_iteration)

        # Track scorer creation phase
        self.run_tracker.add_phase(
            iteration_number=self.current_iteration,
            phase_name="scorer_creation",
            phase_data={
                "inputs": {
                    "requested_objectives_count": len(objectives),
                    "serializer_name": serializer_name
                },
                "outputs": {
                    "objectives_with_scorers": [
                        {
                            "name": obj.name,
                            "description": obj.description,
                            "optimization_direction": obj.optimization_direction,
                            "weight": obj.weight,
                            "population_wise": obj.population_wise,
                            "has_scorer": obj.scorer is not None
                        }
                        for obj in objectives_with_scorers
                    ],
                    "successfully_matched_count": len(objectives_with_scorers),
                    "retry_count": count if count > 0 else 0
                }
            }
        )
        self.run_tracker.save()

        return objectives_with_scorers
    
    async def _optimization_phase(self, population: Optional[Population], objectives: List[Objective], high_level_goal: str) -> Tuple[Population, Dict[str, Any]]:
        """
        Run optimization until convergence.

        Args:
            objectives: List of objectives with scorers

        Returns:
            Tuple of (optimized population, random replacement info dict)
        """
        self.logger.info("-" * 80)
        self.logger.info(f"{'OPTIMIZATION PHASE':^80}")
        self.logger.info("-" * 80)

        # Log input status with objective details
        objectives_lines = [f"Running optimizer with {len(objectives)} objective(s):"]
        for idx, obj in enumerate(objectives, 1):
            # Format the type with weight information
            type_str = f'type="{obj.type}"'
            weight_str = f"weight={obj.weight}" if obj.weight is not None else "weight=None"

            # Format optimization direction (filters don't have direction)
            if obj.type == "filter":
                direction_str = "filter"
            else:
                direction_str = obj.optimization_direction if obj.optimization_direction else "no direction"

            # Add objective header and description
            objectives_lines.append(f"   {idx}. {obj.name} ({type_str}, {weight_str}, {direction_str})")
            objectives_lines.append(f"      Description: {obj.description}")

        self.logger.info("\n".join(objectives_lines))
        self.logger.info(f"Input population: size {population.size if population else 0} from iteration {self.current_iteration - 1}")

        # Handle random candidate replacement (skip iteration 1)
        random_candidate_ratio = self.config.loop_config.get("random_candidate_ratio", 0.0)
        original_population = population
        replaced_candidates_info = None

        if self.current_iteration > 1 and random_candidate_ratio > 0.0 and population is not None and population.size > 0:
            num_to_replace = int(population.size * random_candidate_ratio)
            if num_to_replace > 0:
                # self.logger.info(f"Replacing {num_to_replace} candidates ({random_candidate_ratio*100:.1f}%) with random candidates")

                # Store original population for tracking
                original_population = population

                # Get serializer for representation
                serializer = None
                if self.serializer_name:
                    serializer = get_serializer(self.serializer_name)
                    if serializer is None:
                        self.logger.warning(f"Serializer '{self.serializer_name}' not found, using raw representation")

                # Create random candidates
                random_candidates = await self.optimizer.create_random_candidates(num_to_replace)

                # Replace random subset of current population
                indices_to_replace = random.sample(range(population.size), num_to_replace)
                indices_to_replace.sort()

                # Create new population with replacements
                new_candidates = []
                replaced_info = []
                random_idx = 0

                for i, candidate in enumerate(population.candidates):
                    if i in indices_to_replace:
                        # Get serialized representations
                        original_repr = serializer.serialize(candidate) if serializer else candidate.representation
                        new_repr = serializer.serialize(random_candidates[random_idx]) if serializer else random_candidates[random_idx].representation

                        # Record replacement info
                        replaced_info.append({
                            "position": i,
                            "original_id": candidate.id,
                            "original_representation": original_repr,
                            "original_scores": candidate.scores.copy() if candidate.scores else {},
                            "new_id": random_candidates[random_idx].id,
                            "new_representation": new_repr
                        })
                        new_candidates.append(random_candidates[random_idx])
                        random_idx += 1
                    else:
                        new_candidates.append(candidate)

                population = Population(candidates=new_candidates)
                replaced_candidates_info = replaced_info

                # self.logger.info(f"Replaced {len(replaced_info)} candidates at positions: {[info['position'] for info in replaced_info]}")
                self.logger.info(f"Replacd {num_to_replace} candidates ({random_candidate_ratio*100:.1f}%) with random candidates")

        # Run optimization
        self.logger.info("Running optimizer...")
        self.logger.debug("--- Inside Optimizer ---")
        optimized_population = await self.optimizer.process(
            current_population=population,
            objectives=objectives,
            high_level_goal=high_level_goal
        )
        self.logger.debug("-------------")

        self.logger.info(f"Generated {optimized_population.size} optimized candidate(s)")

        # Get serializer if available
        serializer = None
        if self.serializer_name:
            serializer = get_serializer(self.serializer_name)
        if serializer is None:
                raise ValueError(f"Serializer '{self.serializer_name}' not found in registry")

        # Log top candidates
        if optimized_population.size > 0:
            top_n = min(5, optimized_population.size)

            candidates_lines = [f"Top {top_n} candidate(s):"]
            for i, candidate in enumerate(optimized_population.candidates[:top_n], 1):
                scores_str = ", ".join([f"{name}: {score:.4f}" for name, score in candidate.scores.items()])

                serialized_repr = serializer.serialize(candidate)
                candidates_lines.append(f"  {i}. {serialized_repr}")
                candidates_lines.append(f"      Scores: [{scores_str}]")

            self.logger.info("\n".join(candidates_lines))
        else:
            self.logger.warning("No candidates generated by optimizer")

        # Store optimized candidates
        self.knowledge_manager.store_population(optimized_population, self.current_iteration)
        self.current_population = optimized_population

        # Track optimization phase
        top_candidates = []
        if optimized_population.size > 0:
            top_n = min(5, optimized_population.size)
            for candidate in optimized_population.candidates[:top_n]:
                top_candidates.append({
                    "id": candidate.id[:8],
                    "representation": serializer.serialize(candidate),
                    "scores": candidate.scores
                })

        # Prepare phase data
        phase_data = {
            "inputs": {
                "initial_population_size": original_population.size if original_population else 0,
                "objectives_count": len(objectives),
                "objective_names": [obj.name for obj in objectives]
            },
            "outputs": {
                "optimized_population_size": optimized_population.size,
                "top_candidates": top_candidates
            }
        }

        # Add random candidate replacement info if applicable
        if replaced_candidates_info is not None:
            phase_data["random_candidate_replacement"] = {
                "enabled": True,
                "ratio": random_candidate_ratio,
                "num_replaced": len(replaced_candidates_info),
                "replaced_positions": [info["position"] for info in replaced_candidates_info],
                "replacement_details": [
                    {
                        "position": info["position"],
                        "original_id": info["original_id"][:8],
                        "original_representation": info["original_representation"],
                        "original_scores": info["original_scores"],
                        "new_id": info["new_id"][:8],
                        "new_representation": info["new_representation"]
                    }
                    for info in replaced_candidates_info
                ]
            }
        else:
            phase_data["random_candidate_replacement"] = {
                "enabled": random_candidate_ratio > 0.0,
                "ratio": random_candidate_ratio,
                "num_replaced": 0,
                "reason": "iteration_1" if self.current_iteration == 1 else ("no_population" if population is None or population.size == 0 else "ratio_is_zero")
            }

        self.run_tracker.add_phase(
            iteration_number=self.current_iteration,
            phase_name="optimization",
            phase_data=phase_data
        )
        self.run_tracker.save()

        # Prepare random replacement info for analyzer
        analyzer_random_replacement_info = {
            "occurred": replaced_candidates_info is not None,
            "ratio": random_candidate_ratio,
            "num_replaced": len(replaced_candidates_info) if replaced_candidates_info else 0
        }

        return optimized_population, analyzer_random_replacement_info
    
    async def _analysis_phase(
        self,
        high_level_goal: str,
        context_information: Optional[str],
        population: Population,
        objectives: List[Objective],
        historical_info: Optional[Dict[str, Any]] = None,
        random_replacement_info: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze results and generate comprehensive reports.

        Args:
            candidates: List of candidates to analyze
            objectives: List of objectives used
            historical_info: Optional historical information from previous iterations
            random_replacement_info: Optional information about random candidate replacement

        Returns:
            Dict with keys:
              - analysis_report (str)
              - should_stop (bool)
              - reasoning (str)
        """
        self.logger.info("-" * 80)
        self.logger.info(f"{'ANALYSIS PHASE':^80}")
        self.logger.info("-" * 80)

        # Log input status
        self.logger.info(f"Analyzing {population.size} candidate(s) with {len(objectives)} objective(s)")
        self.logger.debug(f"Has historical info: {historical_info is not None}")
        if random_replacement_info and random_replacement_info.get("occurred"):
            self.logger.info(f"Random replacement context: {random_replacement_info['num_replaced']} candidates ({random_replacement_info['ratio']*100:.1f}%) were replaced before optimization")

        # Generate analysis report
        analysis_result = await self.analyzer.process(
            iteration_number=self.current_iteration,
            high_level_goal=high_level_goal,
            context_information=context_information,
            current_population=population,
            current_objectives=objectives,
            serializer_name=self.serializer_name,
            historical_info=historical_info,
            random_replacement_info=random_replacement_info
        )

        analysis_report = analysis_result["analysis_report"]

        # Store analysis report
        self.knowledge_manager.store_analysis_report(analysis_report, self.current_iteration)
        self.current_analysis_report = analysis_report

        # Log analysis results
        self.logger.info("Analysis Report:\n" + analysis_report)

        # Log decision
        if analysis_result.get("should_stop"):
            self.logger.info("Decision: STOP optimization")
            self.logger.info(f"Reasoning: {analysis_result.get('reasoning', 'No reasoning provided')}")
        else:
            self.logger.info("Decision: CONTINUE optimization")

        # Track analysis phase
        self.run_tracker.add_phase(
            iteration_number=self.current_iteration,
            phase_name="analysis",
            phase_data={
                "inputs": {
                    "population_size": population.size,
                    "objectives_count": len(objectives),
                    "has_historical_info": historical_info is not None
                },
                "outputs": {
                    "analysis_report": analysis_report,
                    "should_stop": analysis_result.get("should_stop", False),
                    "reasoning": analysis_result.get("reasoning", ""),
                    "original_output": analysis_result,
                }
            }
        )
        self.run_tracker.save()

        return analysis_result
    
    async def _collect_and_evaluate_all_candidates(self) -> Population:
        """
        Collect all candidates from all iterations and evaluate them with all used objectives (async).

        Returns:
            Population containing all candidates evaluated with all objectives
        """
        self.logger.info("Collecting all candidates from all iterations")

        # Collect all unique objectives (union set) from all iterations
        all_objectives_dict = {}
        for iteration in range(0, self.current_iteration + 1):
            iteration_objectives = self.knowledge_manager.get_objectives(iteration)
            if iteration_objectives:
                for obj in iteration_objectives:
                    all_objectives_dict[obj.name] = obj

        all_objectives = list(all_objectives_dict.values())
        self.logger.debug(f"Found {len(all_objectives)} unique objectives: {[obj.name for obj in all_objectives]}")

        # Collect all candidates from all iterations
        all_candidates = []
        for iteration in range(0, self.current_iteration + 1):
            iteration_population = self.knowledge_manager.get_population(iteration)
            if iteration_population and not iteration_population.is_empty:
                all_candidates.extend(iteration_population.candidates)

        self.logger.info(f"Collected {len(all_candidates)} candidates from iteration 0 - {self.current_iteration}")

        # Create and evaluate combined population
        combined_population = Population(candidates=all_candidates)
        await combined_population.evaluate(all_objectives)

        self.logger.info(f"Evaluated {combined_population.size} candidates with {len(all_objectives)} objectives")
        return combined_population

    def _collect_api_usage_statistics(self) -> Dict[str, Any]:
        """
        Collect API usage statistics from all modules.

        Returns:
            Dictionary with API usage statistics for each module and total
        """
        modules = {
            "planner": self.planner,
            "scorer_creator": self.scorer_creator,
            "optimizer": self.optimizer,
            "analyzer": self.analyzer,
            "knowledge_manager": self.knowledge_manager
        }

        statistics = {}
        total_stats = {
            "call_count": 0,
            "total_tokens": 0,
            "input_tokens": 0,
            "cache_creation_input_tokens": 0,
            "cache_read_input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "by_model_name": {}
        }

        for module_name, module in modules.items():
            if module is None or not hasattr(module, 'llm_client') or module.llm_client is None:
                statistics[module_name] = None
            else:
                module_stats = module.llm_client.get_stats()

                # Process module-level statistics
                module_level_stats = {
                    "call_count": 0,
                    "total_tokens": 0,
                    "input_tokens": 0,
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                    "output_tokens": 0,
                    "cost": 0.0,
                    "by_model_name": {}
                }

                for model_name, model_stats in module_stats.items():
                    # Add to module-level stats
                    module_level_stats["call_count"] += model_stats["call_count"]
                    module_level_stats["total_tokens"] += model_stats["total_tokens"]
                    module_level_stats["input_tokens"] += model_stats["input_tokens"]
                    module_level_stats["cache_creation_input_tokens"] += model_stats["cache_creation_input_tokens"]
                    module_level_stats["cache_read_input_tokens"] += model_stats["cache_read_input_tokens"]
                    module_level_stats["output_tokens"] += model_stats["output_tokens"]
                    module_level_stats["cost"] += model_stats["cost"]
                    module_level_stats["by_model_name"][model_name] = model_stats

                    # Add to total stats
                    total_stats["call_count"] += model_stats["call_count"]
                    total_stats["total_tokens"] += model_stats["total_tokens"]
                    total_stats["input_tokens"] += model_stats["input_tokens"]
                    total_stats["cache_creation_input_tokens"] += model_stats["cache_creation_input_tokens"]
                    total_stats["cache_read_input_tokens"] += model_stats["cache_read_input_tokens"]
                    total_stats["output_tokens"] += model_stats["output_tokens"]
                    total_stats["cost"] += model_stats["cost"]

                    if model_name not in total_stats["by_model_name"]:
                        total_stats["by_model_name"][model_name] = {
                            "call_count": 0,
                            "total_tokens": 0,
                            "input_tokens": 0,
                            "cache_creation_input_tokens": 0,
                            "cache_read_input_tokens": 0,
                            "output_tokens": 0,
                            "cost": 0.0
                        }

                    total_stats["by_model_name"][model_name]["call_count"] += model_stats["call_count"]
                    total_stats["by_model_name"][model_name]["total_tokens"] += model_stats["total_tokens"]
                    total_stats["by_model_name"][model_name]["input_tokens"] += model_stats["input_tokens"]
                    total_stats["by_model_name"][model_name]["cache_creation_input_tokens"] += model_stats["cache_creation_input_tokens"]
                    total_stats["by_model_name"][model_name]["cache_read_input_tokens"] += model_stats["cache_read_input_tokens"]
                    total_stats["by_model_name"][model_name]["output_tokens"] += model_stats["output_tokens"]
                    total_stats["by_model_name"][model_name]["cost"] += model_stats["cost"]

                statistics[module_name] = module_level_stats

        statistics["total"] = total_stats
        return statistics

    def _collect_llm_responses(self) -> Dict[str, Any]:
        """
        Collect LLM responses from all modules.

        Returns:
            Dictionary with LLM responses for each module
        """
        modules = {
            "planner": self.planner,
            "scorer_creator": self.scorer_creator,
            "optimizer": self.optimizer,
            "analyzer": self.analyzer,
            "knowledge_manager": self.knowledge_manager
        }

        responses = {}
        for module_name, module in modules.items():
            if module is None or not hasattr(module, 'llm_client') or module.llm_client is None:
                responses[module_name] = None
            else:
                responses[module_name] = module.llm_client.responses

        return responses

    async def _finalize_optimization(self) -> None:
        """Finalize the optimization process (async)."""
        self.logger.info("=" * 80)
        self.logger.info(f"{'FINALIZATION':^80}")
        self.logger.info("=" * 80)

        # Set end time
        self.result.end_time = datetime.now()
        duration = (self.result.end_time - self.result.start_time).total_seconds()

        # Update final result
        self.result.final_population = self.current_population
        self.result.final_analysis_report = self.current_analysis_report

        # Log summary
        self.logger.info(f"Run completed: {self.current_iteration} iteration(s) in {duration:.2f}s")
        self.logger.info(f"Termination reason: {self.result.termination_reason}")
        if self.current_population:
            self.logger.info(f"Final population: {self.current_population.size} candidate(s)")

        # Collect all candidates if configured
        return_all_candidates = self.config.loop_config.get("return_all_candidates", True)

        if return_all_candidates:
            all_candidates_population = await self._collect_and_evaluate_all_candidates()
            self.result.all_candidates_population = all_candidates_population
        else:
            all_candidates_population = None
            self.logger.debug("Skipping all_candidates collection (return_all_candidates=False)")

        # Track final outputs
        self.run_tracker.set_outputs(
            termination_reason=self.result.termination_reason,
            total_iterations=self.current_iteration,
            final_population=self.current_population,
            final_analysis_report=self.current_analysis_report,
            all_candidates_population=all_candidates_population,
            error_message=self.result.error_message
        )
        self.run_tracker.finalize(self.result.status)
        self.run_tracker.save()

        # Save results if configured
        self._save_results()

        # Save API usage statistics and LLM responses
        self._save_api_usage_statistics()
        self._save_llm_responses()

        self.logger.info("Finalization complete")

    def _save_intermediate_results(self) -> None:
        """
        Save intermediate results, API usage statistics, and LLM responses.
        This is called after each iteration to ensure data is preserved even if execution is interrupted.
        """
        try:
            # Save all three types of files
            self._save_results()
            self._save_api_usage_statistics()
            self._save_llm_responses()
        except Exception as e:
            # Don't fail the run if saving fails, just log the error
            self.logger.error(f"Failed to save intermediate results: {str(e)}")

    def _save_results(self) -> None:
        """Save final results to files."""
        try:
            output_dir = self.output_dir

            # Get the serializer if available
            serializer = None
            if self.serializer_name:
                serializer = get_serializer(self.serializer_name)
                if serializer is None:
                    self.logger.warning(f"Serializer '{self.serializer_name}' not found, Population instances will not be serialized")

            # Convert result to dict
            result_dict = self.result.model_dump()

            # Helper function to serialize a Population instance
            def serialize_population(population: Optional[Population]) -> Optional[Dict[str, Any]]:
                if population is None:
                    return None

                # Get the full population dict
                population_dict = population.model_dump()

                # Replace the candidates list with simplified serialized versions
                serialized_candidates = []
                for candidate in population.candidates:
                    candidate_data = {
                        "id": candidate.id,
                        "scores": candidate.scores
                    }

                    # Add serialized_representation if serializer is available
                    if serializer:
                        try:
                            candidate_data["serialized_representation"] = serializer.serialize(candidate)
                        except Exception as e:
                            self.logger.warning(f"Failed to serialize candidate {candidate.id}: {str(e)}")
                            candidate_data["serialized_representation"] = None
                    else:
                        candidate_data["serialized_representation"] = None

                    serialized_candidates.append(candidate_data)

                # Replace candidates in the dict
                population_dict["candidates"] = serialized_candidates

                return population_dict

            # Serialize final_population
            if "final_population" in result_dict and result_dict["final_population"] is not None:
                result_dict["final_population"] = serialize_population(self.result.final_population)

            # For all_candidates_population, save by iteration instead of merged
            if "all_candidates_population" in result_dict:
                # Remove the merged population from result_dict
                result_dict.pop("all_candidates_population")

                # Save populations by iteration
                all_populations_by_iteration = {}
                for iteration in range(0, self.current_iteration + 1):
                    iteration_population = self.knowledge_manager.get_population(iteration)
                    if iteration_population and not iteration_population.is_empty:
                        all_populations_by_iteration[f"iteration_{iteration}"] = serialize_population(iteration_population)

                result_dict["all_candidates_by_iteration"] = all_populations_by_iteration

            # Save result summary
            result_file = output_dir / "results.json"
            with open(result_file, 'w', encoding='utf-8') as f:
                json.dump(result_dict, f, indent=2, default=str)

            self.logger.info(f"Results saved to {result_file}")

        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")

    def _save_api_usage_statistics(self) -> None:
        """Save API usage statistics to runs/{run_id}/outputs/api_usage_statistics.json."""
        try:
            # Get the run directory (parent of output_directory which is runs/{run_id}/outputs)
            output_dir = self.output_dir
            stats_file = output_dir / "api_usage_statistics.json"

            # Collect statistics from all modules
            statistics = self._collect_api_usage_statistics()

            # Save to file
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(statistics, f, indent=2, default=str)

            self.logger.info(f"API usage statistics saved to {stats_file}")

        except Exception as e:
            self.logger.error(f"Failed to save API usage statistics: {str(e)}")

    def _save_llm_responses(self) -> None:
        """Save LLM responses to runs/{run_id}/outputs/llm_responses.json."""
        try:
            output_dir = self.output_dir
            responses_file = output_dir / "llm_responses.json"

            # Collect responses from all modules
            responses = self._collect_llm_responses()

            # Save to file
            with open(responses_file, 'w', encoding='utf-8') as f:
                json.dump(responses, f, default=str)

            self.logger.info(f"LLM responses saved to {responses_file}")

        except Exception as e:
            self.logger.error(f"Failed to save LLM responses: {str(e)}")

    async def _stop_optimization(self) -> None:
        """Stop the optimization process (async)."""
        self.logger.warning("Optimization interrupted - saving current progress")
        self.is_running = False
        self.result.status = "terminated"
        self.result.termination_reason = "manual_stop"
        self.result.end_time = datetime.now()

        # Track the manual stop
        self.run_tracker.set_outputs(
            termination_reason="manual_stop",
            total_iterations=self.current_iteration,
            final_population=self.current_population,
            final_analysis_report=self.current_analysis_report,
            all_candidates_population=None,
            error_message=None
        )

        # Finalize to save current progress
        await self._finalize_optimization()
    
    def _log_module_configurations(self) -> None:
        """Log module configurations and LLM status."""
        self.logger.info("Framework initialized with modules:")

        modules = [
            ("Planner", self.planner),
            ("ScorerCreator", self.scorer_creator),
            ("Optimizer", self.optimizer),
            ("Analyzer", self.analyzer),
            ("KnowledgeManager", self.knowledge_manager)
        ]

        # Log module information
        for name, module in modules:
            status = module.get_status()
            self.logger.info(f"  {name}: {status['module_name']} v{status['module_version']} (LLM: {status['llm_name']})")

        # Track registered scorers
        try:
            registered_scorer_names = list_scorers()
            scorers_info = []
            for scorer_name in registered_scorer_names:
                try:
                    metadata = get_scorer_metadata(scorer_name)
                    scorer_info = {"name": scorer_name}
                    if metadata:
                        scorer_info["description"] = metadata.get("description", "")
                        scorer_info["population_wise"] = metadata.get("population_wise", False)
                    scorers_info.append(scorer_info)
                except Exception:
                    scorers_info.append({"name": scorer_name})

            self.run_tracker.set_registered_scorers(scorers_info)
            self.logger.info(f"Registered scorers: {len(scorers_info)}")
            self.logger.debug(f"Scorer names: {', '.join([s['name'] for s in scorers_info[:10]])}")
        except Exception as e:
            self.logger.warning(f"Could not retrieve registered scorers: {str(e)}")

        self.run_tracker.save()
