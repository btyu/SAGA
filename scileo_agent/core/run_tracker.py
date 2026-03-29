"""
Run tracking module for the SciLeo Agent framework.

This module provides the RunTracker class that maintains a comprehensive
JSON-serializable record of the orchestrator's execution process.

The generated JSON file (run_process_tracking.json) has the following structure:

{
  "run_metadata": {
    "run_id": "unique-run-id",
    "run_name": "Human-readable name",
    "start_time": "ISO timestamp",
    "end_time": "ISO timestamp",
    "duration_seconds": 123.45,
    "status": "running|completed|failed|terminated"
  },
  "inputs": {
    "high_level_goal": "The optimization goal",
    "context_information": "Additional context",
    "serializer_name": "Name of serializer used",
    "initial_objectives": [...],
    "initial_population_size": 10,
    "max_iterations": 5
  },
  "configuration": {
    // Full framework configuration (same as results.json), includes:
    // - Framework metadata (name, version)
    // - All module configurations with their LLM configs
    // - Loop configuration and other settings
    "framework_name": "SciLeo Agent",
    "framework_version": "...",
    "run_id": "...",
    "modules": {
      "planner": {
        "module_id": "planner_1",
        "module_type": "planner",
        "module_name": "GeneralPlanner",
        "module_version": "1.0.0",
        "config": {...},
        "llm_config": {"model": "...", ...}
      },
      // ... other modules (scorer_creator, optimizer, analyzer, knowledge_manager)
    },
    "loop_config": {
      "max_iterations": 10,
      "max_objective_planning_retries": 3,
      "return_all_candidates": true
    },
    "output_directory": "runs/.../outputs",
    "save_final_results": true
  },
  "registered_scorers": [
    {"name": "scorer1", "description": "...", "population_wise": false},
    ...
  ],
  "iterations": [
    {
      "iteration_number": 1,
      "start_time": "ISO timestamp",
      "end_time": "ISO timestamp",
      "phases": [
        {
          "phase_name": "planning|scorer_creation|optimization|analysis",
          "timestamp": "ISO timestamp",
          "inputs": {...},
          "outputs": {...}
        },
        ...
      ]
    },
    ...
  ],
  "outputs": {
    "termination_reason": "early_stop|max_iterations|error|manual_stop",
    "total_iterations_completed": 3,
    "final_population_size": 100,
    "final_analysis_report": "...",
    "all_candidates_population_size": 300,
    "error_message": null
  }
}
"""

from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path
import json

from .data_models import Population, Objective


class RunTracker:
    """
    Tracks and manages structured information about the orchestrator's execution process.

    This class maintains a comprehensive JSON-serializable record of:
    - Run inputs and configuration
    - Module configurations and registered scorers
    - Iteration/phase details with inputs/outputs
    - Final outputs and termination information
    """

    def __init__(self, run_id: str, run_name: Optional[str], output_dir: Path):
        """
        Initialize the run tracker.

        Args:
            run_id: Unique run identifier
            run_name: Human-readable run name
            output_dir: Directory where the tracking file will be saved
        """
        self.output_dir = output_dir
        self.tracking_file = output_dir / "run_process_tracking.json"

        # Initialize the tracking data structure
        self.data = {
            "run_metadata": {
                "run_id": run_id,
                "run_name": run_name,
                "start_time": datetime.now().isoformat(),
                "end_time": None,
                "duration_seconds": None,
                "status": "running"
            },
            "inputs": {
                "high_level_goal": None,
                "context_information": None,
                "serializer_name": None,
                "initial_objectives": [],
                "initial_population_size": 0,
                "max_iterations": None
            },
            "configuration": {},  # Will store full framework config including modules
            "registered_scorers": [],
            "iterations": [],
            "outputs": {
                "termination_reason": None,
                "total_iterations_completed": 0,
                "final_population_size": 0,
                "final_analysis_report": None,
                "all_candidates_population_size": 0,
                "error_message": None
            }
        }

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def set_inputs(self, high_level_goal: str, context_information: Optional[str],
                   serializer_name: Optional[str], initial_objectives: Optional[List[Objective]],
                   initial_population: Optional[Population], max_iterations: int):
        """Record run inputs."""
        self.data["inputs"]["high_level_goal"] = high_level_goal
        self.data["inputs"]["context_information"] = context_information
        self.data["inputs"]["serializer_name"] = serializer_name
        self.data["inputs"]["initial_objectives"] = [
            {"name": obj.name, "description": obj.description, "optimization_direction": obj.optimization_direction}
            for obj in initial_objectives
        ] if initial_objectives else []
        self.data["inputs"]["initial_population_size"] = initial_population.size if initial_population else 0
        self.data["inputs"]["max_iterations"] = max_iterations

    def set_configuration(self, framework_config):
        """
        Record framework configuration (includes all module configs).

        This stores the full config dump, same as results.json, which includes:
        - Framework metadata
        - All module configurations with their LLM configs
        - Loop configuration
        - Output settings
        """
        self.data["configuration"] = framework_config.model_dump()

    def set_registered_scorers(self, scorers: List[Dict[str, Any]]):
        """
        Record registered scorers.

        Args:
            scorers: List of scorer information dicts
        """
        self.data["registered_scorers"] = scorers

    def start_iteration(self, iteration_number: int):
        """Start tracking a new iteration."""
        iteration_data = {
            "iteration_number": iteration_number,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "phases": []
        }
        self.data["iterations"].append(iteration_data)

    def add_phase(self, iteration_number: int, phase_name: str, phase_data: Dict[str, Any]):
        """
        Add phase information to the current iteration.

        Args:
            iteration_number: Current iteration number (0 for initialization)
            phase_name: Name of the phase (e.g., "planning", "scorer_creation")
            phase_data: Dictionary containing phase inputs, outputs, and metadata
        """
        # Find the iteration
        iteration = None
        for it in self.data["iterations"]:
            if it["iteration_number"] == iteration_number:
                iteration = it
                break

        if iteration is None:
            # Create iteration if it doesn't exist (for iteration 0)
            self.start_iteration(iteration_number)
            iteration = self.data["iterations"][-1]

        # Add phase data
        phase_entry = {
            "phase_name": phase_name,
            "timestamp": datetime.now().isoformat(),
            **phase_data
        }
        iteration["phases"].append(phase_entry)

    def end_iteration(self, iteration_number: int):
        """Mark iteration as completed."""
        for iteration in self.data["iterations"]:
            if iteration["iteration_number"] == iteration_number:
                iteration["end_time"] = datetime.now().isoformat()
                break

    def set_outputs(self, termination_reason: str, total_iterations: int,
                   final_population: Optional[Population], final_analysis_report: Optional[str],
                   all_candidates_population: Optional[Population], error_message: Optional[str] = None):
        """Record final outputs."""
        self.data["outputs"]["termination_reason"] = termination_reason
        self.data["outputs"]["total_iterations_completed"] = total_iterations
        self.data["outputs"]["final_population_size"] = final_population.size if final_population else 0
        self.data["outputs"]["final_analysis_report"] = final_analysis_report
        self.data["outputs"]["all_candidates_population_size"] = all_candidates_population.size if all_candidates_population else 0
        self.data["outputs"]["error_message"] = error_message

    def finalize(self, status: str):
        """Finalize the tracking data."""
        end_time = datetime.now()
        self.data["run_metadata"]["end_time"] = end_time.isoformat()
        self.data["run_metadata"]["status"] = status

        # Calculate duration
        start_time = datetime.fromisoformat(self.data["run_metadata"]["start_time"])
        self.data["run_metadata"]["duration_seconds"] = (end_time - start_time).total_seconds()

    def save(self):
        """Save the tracking data to JSON file."""
        try:
            with open(self.tracking_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, default=str)
        except Exception as e:
            # Don't fail the run if tracking fails
            print(f"Warning: Failed to save run tracking data: {str(e)}")
