"""
Population initialization module for molecular genetic algorithms.

Handles loading molecules from various data sources and creating initial populations.
"""

import logging
import os
import random
from typing import Callable, Dict, List, Optional

import pandas as pd

from scileo_agent.core.data_models import Candidate, Population
from modules.small_molecule_drug_design.utils.rdkit_utils import filter_smiles_all


class PopulationInitializer:
    """Handles initialization of molecular populations from various sources."""

    def __init__(
        self,
        data_dir: str,
        seed: int = 42,
        sanitize_fn: Optional[Callable[[str], str]] = None,
    ):
        """
        Initialize the population initializer.

        Args:
            data_dir: Base directory containing molecule data files
            seed: Random seed for reproducibility
            sanitize_fn: Optional function to sanitize SMILES strings
        """
        self.data_dir = data_dir
        self.seed = seed
        self.rng = random.Random(seed)
        self.sanitize_fn = sanitize_fn or self._default_sanitize
        self._custom_sources: Dict[str, Callable] = {}

    def _default_sanitize(self, smiles: str) -> str:
        """Default SMILES sanitization (strip whitespace)."""
        return str(smiles).strip()

    def register_source(self, name: str, loader: Callable[[int], List[str]]) -> None:
        """
        Register a custom data source.

        Args:
            name: Name of the source
            loader: Function that takes num_samples and returns list of SMILES
        """
        self._custom_sources[name] = loader

    def create_population(
        self,
        population_size: int,
        init_group: str = "zinc",
        file_sample_size: int = 10,
        apply_filters: bool = True,
    ) -> Population:
        """
        Create initial population from specified source.

        Args:
            population_size: Number of molecules to sample
            init_group: Data source identifier
            file_sample_size: Max samples from file-based sources
            apply_filters: Whether to apply PAINS/Brenk filters

        Returns:
            Population of candidates
        """
        smiles_list = self._sample_smiles(
            population_size, init_group, file_sample_size
        )

        # Apply filtering if requested
        if apply_filters:
            smiles_list = filter_smiles_all(smiles_list)

        candidates = [
            Candidate(representation=self.sanitize_fn(smiles))
            for smiles in smiles_list
        ]
        return Population(candidates=candidates)

    def _sample_smiles(
        self, num_samples: int, init_group: str, file_sample_size: int = 10
    ) -> List[str]:
        """
        Sample SMILES from specified source.

        Supported init_group values:
          - 'zinc': zinc_250k.csv
          - 'diverse_10k': enamine_diverse_10k.csv
          - 'enamine': large_scale_molecule.csv
          - 'covid': known_covid.smi
          - '@<path>': custom file (.smi/.txt/.csv)
          - custom registered sources

        Args:
            num_samples: Number of samples to draw
            init_group: Source identifier
            file_sample_size: Max samples from file sources

        Returns:
            List of SMILES strings
        """
        # Check custom sources first
        if init_group in self._custom_sources:
            return self._custom_sources[init_group](num_samples)

        # File-based source (starts with @)
        if isinstance(init_group, str) and init_group.startswith("@"):
            return self._load_from_file(
                init_group[1:], num_samples, file_sample_size
            )

        # Bundled source
        if init_group == "covid":
            return self._load_covid(num_samples)

        # Standard CSV sources
        return self._load_from_csv_source(init_group, num_samples)

    def _load_from_file(
        self, file_path: str, num_samples: int, file_sample_size: int
    ) -> List[str]:
        """Load SMILES from custom file, backfill with Enamine if needed."""
        # Resolve relative paths to data/molecules directory
        if not os.path.isabs(file_path) and not os.path.exists(file_path):
            candidate_path = os.path.join(self.data_dir, file_path)
            if os.path.exists(candidate_path):
                file_path = candidate_path

        try:
            if file_path.lower().endswith(".csv"):
                file_smiles = self._load_csv(file_path)
            else:
                # Plain text file (one SMILES per line)
                with open(file_path, "r") as fh:
                    file_smiles = [
                        self.sanitize_fn(line) for line in fh if line.strip()
                    ]
        except Exception as e:
            raise RuntimeError(f"Failed to load file '{file_path}': {e}")

        if not file_smiles:
            logging.warning(f"No SMILES loaded from {file_path}, using default source")
            return self._load_from_csv_source("diverse_10k", num_samples)

        # Sample from file (up to file_sample_size)
        num_from_file = min(file_sample_size, len(file_smiles), num_samples)
        file_indices = list(range(len(file_smiles)))
        self.rng.shuffle(file_indices)
        selected_from_file = [file_smiles[i] for i in file_indices[:num_from_file]]

        logging.info(
            f"Selected {len(selected_from_file)} molecules from {file_path}"
        )

        # Backfill with Enamine if needed
        remaining = num_samples - num_from_file
        if remaining > 0:
            enamine_smiles = self._load_enamine_diverse(remaining)
            combined = selected_from_file + enamine_smiles
            return list(filter_smiles_all(combined))

        return list(filter_smiles_all(selected_from_file))

    def _load_covid(self, num_samples: int) -> List[str]:
        """Load COVID-19 known binders."""
        covid_path = os.path.join(self.data_dir, "known_covid.smi")
        with open(covid_path, "r") as fh:
            smiles_list = [self.sanitize_fn(line) for line in fh if line.strip()]

        if num_samples >= len(smiles_list):
            return smiles_list

        indices = list(range(len(smiles_list)))
        self.rng.shuffle(indices)
        return [smiles_list[i] for i in indices[:num_samples]]

    def _load_from_csv_source(self, source_name: str, num_samples: int) -> List[str]:
        """Load from predefined CSV sources."""
        source_mapping = {
            "zinc": ("zinc_250k.csv", "smiles"),
            "diverse_10k": ("enamine_diverse_10k.csv", "SMILES"),
            "enamine_top500": (
                "/gpfs/radev/home/tl688/pitl688/scileoagent_drug/examine_extracted_500.csv",
                "smiles",
            ),
            "enamine": (
                "/gpfs/radev/home/tl688/pitl688/scileoagent_drug/large_scale_molecule.csv",
                "smiles",
            ),
            "mproknownbinder": ("mpro_protein_bindinglist.csv", "smiles"),
            "mpro_examine_mix": ("mpro_update_3090.csv", "smiles"),
        }

        # Default to diverse_10k if unknown
        file_name, column = source_mapping.get(
            source_name, ("enamine_diverse_10k.csv", "SMILES")
        )

        # Build full path for relative files
        if not os.path.isabs(file_name):
            file_name = os.path.join(self.data_dir, file_name)

        smiles_list = self._load_csv(file_name, column)

        if num_samples >= len(smiles_list):
            return list(filter_smiles_all(smiles_list))

        # Sample randomly
        indices = list(range(len(smiles_list)))
        self.rng.shuffle(indices)
        selected = [smiles_list[i] for i in indices[:num_samples]]
        return list(filter_smiles_all(selected))

    def _load_csv(self, file_path: str, column: Optional[str] = None) -> List[str]:
        """Load SMILES from CSV file."""
        df = pd.read_csv(file_path)

        # Auto-detect column if not specified
        if column is None:
            if "smiles" in df.columns:
                column = "smiles"
            elif "SMILES" in df.columns:
                column = "SMILES"
            else:
                raise RuntimeError(
                    f"CSV '{file_path}' must contain a 'smiles' or 'SMILES' column"
                )

        return [self.sanitize_fn(s) for s in df[column].astype(str).tolist()]

    def _load_enamine_diverse(self, num_samples: int) -> List[str]:
        """Load from Enamine Diverse 10K dataset."""
        return self._load_from_csv_source("diverse_10k", num_samples)
