"""
LLM interaction handlers for molecular optimization.

Provides consistent prompt building and response parsing for molecular operations.
"""

import logging
import re
from typing import Callable, List, Optional, Tuple

import yaml

from scileo_agent.core.data_models import Candidate, Objective
from modules.small_molecule_drug_design.prompting.common_prompts import (
    SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT,
)
from modules.small_molecule_drug_design.prompting.multiobj_prompts import (
    build_multiobj_crossover_prompt,
    build_multiobj_mutation_prompt,
    build_multiobjective_weightage_prompt,
)


class ResponseParser:
    """Parse and validate LLM responses for molecular operations."""

    def __init__(self, sanitize_fn: Optional[Callable[[str], str]] = None):
        """
        Initialize response parser.

        Args:
            sanitize_fn: Function to sanitize SMILES strings
        """
        self.sanitize_fn = sanitize_fn or self._default_sanitize

    def _default_sanitize(self, smiles: str) -> str:
        """Default SMILES sanitization."""
        s = str(smiles).strip()
        # Remove quotes
        if len(s) >= 2 and ((s[0] == '"' and s[-1] == '"') or (s[0] == "'" and s[-1] == "'")):
            s = s[1:-1].strip()
        # Remove backticks
        if s.startswith("`") and s.endswith("`") and len(s) >= 2:
            s = s[1:-1].strip()
        # Remove CXSMILES annotations
        if "|" in s:
            s = s.split("|", 1)[0].strip()
        # Remove trailing punctuation
        s = s.rstrip(",;").strip()
        # Keep only first token
        if any(ch.isspace() for ch in s):
            s = s.split()[0]
        return s

    def parse_yaml_response(
        self, content: str, expected_keys: List[str]
    ) -> dict:
        """
        Parse YAML response with regex fallback.

        Args:
            content: Response content string
            expected_keys: List of expected keys in response

        Returns:
            Parsed dictionary

        Raises:
            ValueError: If parsing fails or expected keys missing
        """
        # Try regex extraction first (more robust)
        if "molecule" in expected_keys:
            molecule_pattern = r"molecule:\s*([^\n\r]+)"
            molecule_match = re.search(molecule_pattern, content, re.IGNORECASE)

            explanation_pattern = r"explanation:\s*(.*?)(?=\n\s*molecule:|$)"
            explanation_match = re.search(
                explanation_pattern, content, re.IGNORECASE | re.DOTALL
            )

            if molecule_match and explanation_match:
                return {
                    "molecule": self.sanitize_fn(molecule_match.group(1)),
                    "explanation": explanation_match.group(1).strip(),
                }

        # Fallback to YAML parsing
        try:
            if "---" in content:
                documents = content.split("---")
                parsed = yaml.safe_load(documents[-1].strip())
            else:
                parsed = yaml.safe_load(content)

            # Validate expected keys
            missing_keys = [k for k in expected_keys if k not in parsed]
            if missing_keys:
                raise ValueError(f"Missing keys in response: {missing_keys}")

            # Sanitize molecule if present
            if "molecule" in parsed:
                parsed["molecule"] = self.sanitize_fn(parsed["molecule"])

            return parsed

        except (yaml.YAMLError, KeyError, AttributeError) as e:
            raise ValueError(f"Could not parse response: {e}. Response: {content[:200]}")

    def parse_crossover_response(self, response: dict) -> Tuple[Candidate, str]:
        """
        Parse crossover response.

        Args:
            response: LLM response dict

        Returns:
            Tuple of (offspring candidate, explanation)
        """
        content = response.get("content", "").strip()
        parsed = self.parse_yaml_response(content, ["explanation", "molecule"])

        candidate = Candidate(representation=parsed["molecule"])
        return candidate, parsed["explanation"]

    def parse_mutation_response(self, response: dict) -> Tuple[Candidate, str]:
        """
        Parse mutation response.

        Args:
            response: LLM response dict

        Returns:
            Tuple of (mutated candidate, explanation)
        """
        content = response.get("content", "").strip()
        parsed = self.parse_yaml_response(content, ["explanation", "molecule"])

        candidate = Candidate(representation=parsed["molecule"])
        return candidate, parsed["explanation"]

    def parse_weightage_response(self, response: dict) -> Tuple[List[float], str]:
        """
        Parse multi-objective weightage response.

        Args:
            response: LLM response dict

        Returns:
            Tuple of (weights list, explanation)
        """
        content = response.get("content", "").strip()

        try:
            parsed = yaml.safe_load(content)
            explanation = parsed["explanation"]
            weights = parsed["weights"]
            return weights, explanation
        except (yaml.YAMLError, KeyError) as e:
            raise ValueError(
                f"Could not parse weightage response: {e}. Response: {content[:200]}"
            )


class MolecularLLMHandler:
    """Handles LLM interactions for molecular optimization."""

    def __init__(
        self,
        llm_client: Callable,
        combiner,
        compute_score_fn: Callable,
        sanitize_fn: Optional[Callable[[str], str]] = None,
        system_prompt: str = SMALL_MOLECULE_OPTIMIZATION_SYSTEM_PROMPT,
    ):
        """
        Initialize LLM handler.

        Args:
            llm_client: Function to call LLM (takes prompt, returns response)
            combiner: ObjectiveCombiner instance for aggregation equations
            compute_score_fn: Function to compute candidate scores
            sanitize_fn: Function to sanitize SMILES
            system_prompt: System prompt for molecular optimization
        """
        self.llm_client = llm_client
        self.combiner = combiner
        self.compute_score_fn = compute_score_fn
        self.system_prompt = system_prompt
        self.parser = ResponseParser(sanitize_fn)

    def request_crossover(
        self,
        parents: List[Candidate],
        objectives: List[Objective],
        objectives_weights: List[float],
        add_3d_pose_info: bool = False,
        return_artifacts: bool = False,
    ) -> Tuple[Candidate, str] | Tuple[Candidate, str, dict]:
        """
        Execute LLM crossover.

        Args:
            parents: List of parent candidates (length 2)
            objectives: Optimization objectives
            objectives_weights: Objective weights
            add_3d_pose_info: Whether to include 3D docking info
            return_artifacts: Whether to return prompt/response

        Returns:
            (offspring, explanation) or (offspring, explanation, response)
        """
        # Compute aggregate scores
        agg_a = self.compute_score_fn(parents[0], objectives)
        agg_b = self.compute_score_fn(parents[1], objectives)

        # Get aggregation equation
        aggregation_equation = self.combiner.aggregation_equation(
            objectives, objectives_weights
        )

        # Get residue maps if using 3D info
        residue_map_a = None
        residue_map_b = None
        if add_3d_pose_info:
            residue_map_a = parents[0].metadata.get("docking_residue_map", None)
            residue_map_b = parents[1].metadata.get("docking_residue_map", None)

        # Build prompt
        prompt = build_multiobj_crossover_prompt(
            parents,
            objectives,
            aggregation_equation,
            agg_a,
            agg_b,
            add_3d_pose_info,
            residue_map_a,
            residue_map_b,
        )

        # Call LLM
        response = self.llm_client(prompt, system_prompt=self.system_prompt)

        # Parse response
        offspring, explanation = self.parser.parse_crossover_response(response)

        if return_artifacts:
            return offspring, prompt, response
        return offspring, explanation

    def request_mutation(
        self,
        candidate: Candidate,
        objectives: List[Objective],
        objectives_weights: List[float],
        return_artifacts: bool = False,
    ) -> Tuple[Candidate, str] | Tuple[Candidate, str, dict]:
        """
        Execute LLM mutation.

        Args:
            candidate: Candidate to mutate
            objectives: Optimization objectives
            objectives_weights: Objective weights
            return_artifacts: Whether to return prompt/response

        Returns:
            (mutant, explanation) or (mutant, explanation, response)
        """
        # Compute aggregate score
        agg = self.compute_score_fn(candidate, objectives)

        # Get aggregation equation
        aggregation_equation = self.combiner.aggregation_equation(
            objectives, objectives_weights
        )

        # Build prompt
        prompt = build_multiobj_mutation_prompt(
            candidate, objectives, aggregation_equation, agg
        )

        # Call LLM
        response = self.llm_client(prompt, system_prompt=self.system_prompt)

        # Parse response
        mutant, explanation = self.parser.parse_mutation_response(response)

        if return_artifacts:
            return mutant, prompt, response
        return mutant, explanation

    def request_weights(
        self, objectives: List[Objective]
    ) -> Tuple[List[float], str]:
        """
        Get multi-objective weights from LLM.

        Args:
            objectives: Optimization objectives

        Returns:
            Tuple of (weights, explanation)
        """
        prompt = build_multiobjective_weightage_prompt(objectives)
        response = self.llm_client(prompt, system_prompt=self.system_prompt)

        try:
            weights, explanation = self.parser.parse_weightage_response(response)
            return weights, explanation
        except Exception as e:
            logging.warning(
                f"Weight parsing failed: {e}. Using equal weights."
            )
            return [1.0 for _ in objectives], "Equal weights (parsing failed)"
