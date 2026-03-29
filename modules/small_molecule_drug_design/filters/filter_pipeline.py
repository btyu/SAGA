"""
Modular filtering pipeline for molecular candidates.

Provides composable filters for SMARTS patterns, properties, and structural features.
"""

from abc import ABC, abstractmethod
from typing import List

from rdkit import Chem

from scileo_agent.core.data_models import Candidate
from modules.small_molecule_drug_design.utils.rdkit_utils import (
    filter_smiles_all,
    structure_filter,
)


class MoleculeFilter(ABC):
    """Abstract base class for molecule filters."""

    @abstractmethod
    def filter(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Filter candidates based on specific criteria.

        Args:
            candidates: List of candidates to filter

        Returns:
            Filtered list of candidates
        """
        pass

    def __call__(self, candidates: List[Candidate]) -> List[Candidate]:
        """Allow filter to be called as a function."""
        return self.filter(candidates)


class SMARTSFilter(MoleculeFilter):
    """Filter candidates using PAINS/Brenk and custom SMARTS patterns."""

    def __init__(self, use_default_filters: bool = True):
        """
        Initialize SMARTS filter.

        Args:
            use_default_filters: Whether to use built-in PAINS/Brenk filters
        """
        self.use_default_filters = use_default_filters

    def filter(self, candidates: List[Candidate]) -> List[Candidate]:
        """Filter out candidates matching unwanted SMARTS patterns."""
        if not candidates:
            return candidates

        if not self.use_default_filters:
            return candidates

        # Extract SMILES
        smiles_list = [c.representation for c in candidates]

        # Apply filter_smiles_all from rdkit_utils
        allowed_smiles = set(filter_smiles_all(smiles_list))

        # Return candidates whose SMILES passed the filter
        return [c for c in candidates if c.representation in allowed_smiles]


class StructureFilter(MoleculeFilter):
    """Filter molecules by structural features (e.g., presence of specific atoms)."""

    def __init__(self, forbidden_atoms: List[str] = None):
        """
        Initialize structure filter.

        Args:
            forbidden_atoms: List of atom symbols to exclude (e.g., ['F', 'S'])
        """
        self.forbidden_atoms = forbidden_atoms or []

    def filter(self, candidates: List[Candidate]) -> List[Candidate]:
        """Filter out candidates containing forbidden atoms."""
        if not self.forbidden_atoms:
            return candidates

        filtered = []
        for candidate in candidates:
            try:
                # Check if any forbidden atom is present
                has_forbidden = any(
                    structure_filter(atom, candidate.representation)
                    for atom in self.forbidden_atoms
                )

                if not has_forbidden:
                    filtered.append(candidate)
            except Exception:
                # Skip candidates that cause errors
                continue

        return filtered


class PropertyFilter(MoleculeFilter):
    """Filter molecules by molecular properties."""

    def __init__(
        self,
        mw_range: tuple = None,
        logp_range: tuple = None,
        tpsa_range: tuple = None,
        num_atoms_range: tuple = None,
    ):
        """
        Initialize property filter.

        Args:
            mw_range: (min, max) molecular weight
            logp_range: (min, max) LogP
            tpsa_range: (min, max) TPSA
            num_atoms_range: (min, max) number of atoms
        """
        self.mw_range = mw_range
        self.logp_range = logp_range
        self.tpsa_range = tpsa_range
        self.num_atoms_range = num_atoms_range

    def filter(self, candidates: List[Candidate]) -> List[Candidate]:
        """Filter candidates based on property ranges."""
        from rdkit.Chem import Descriptors

        filtered = []
        for candidate in candidates:
            try:
                mol = Chem.MolFromSmiles(candidate.representation)
                if mol is None:
                    continue

                # Check MW
                if self.mw_range:
                    mw = Descriptors.MolWt(mol)
                    if not (self.mw_range[0] <= mw <= self.mw_range[1]):
                        continue

                # Check LogP
                if self.logp_range:
                    logp = Descriptors.MolLogP(mol)
                    if not (self.logp_range[0] <= logp <= self.logp_range[1]):
                        continue

                # Check TPSA
                if self.tpsa_range:
                    tpsa = Descriptors.TPSA(mol)
                    if not (self.tpsa_range[0] <= tpsa <= self.tpsa_range[1]):
                        continue

                # Check num atoms
                if self.num_atoms_range:
                    num_atoms = mol.GetNumAtoms()
                    if not (
                        self.num_atoms_range[0] <= num_atoms <= self.num_atoms_range[1]
                    ):
                        continue

                filtered.append(candidate)
            except Exception:
                continue

        return filtered


class ValidityFilter(MoleculeFilter):
    """Filter out invalid SMILES strings."""

    def filter(self, candidates: List[Candidate]) -> List[Candidate]:
        """Remove candidates with invalid SMILES."""
        filtered = []
        for candidate in candidates:
            try:
                mol = Chem.MolFromSmiles(candidate.representation)
                if mol is not None and mol.GetNumAtoms() > 0:
                    filtered.append(candidate)
            except Exception:
                continue

        return filtered


class FilterPipeline:
    """Chain multiple filters together in a pipeline."""

    def __init__(self, filters: List[MoleculeFilter] = None):
        """
        Initialize filter pipeline.

        Args:
            filters: List of filters to apply in order
        """
        self.filters = filters or []

    def add_filter(self, filter: MoleculeFilter) -> "FilterPipeline":
        """
        Add a filter to the pipeline.

        Args:
            filter: Filter to add

        Returns:
            Self for method chaining
        """
        self.filters.append(filter)
        return self

    def apply(self, candidates: List[Candidate]) -> List[Candidate]:
        """
        Apply all filters in sequence.

        Args:
            candidates: List of candidates to filter

        Returns:
            Filtered list of candidates
        """
        result = candidates
        for filter in self.filters:
            result = filter.filter(result)
            if not result:  # Early exit if no candidates remain
                break
        return result

    def __call__(self, candidates: List[Candidate]) -> List[Candidate]:
        """Allow pipeline to be called as a function."""
        return self.apply(candidates)


def create_default_pipeline(enable_structure_filter: bool = False) -> FilterPipeline:
    """
    Create default filter pipeline for molecular GA.

    Args:
        enable_structure_filter: Whether to enable F/S filtering

    Returns:
        Configured FilterPipeline
    """
    pipeline = FilterPipeline()

    # Always filter validity and SMARTS
    pipeline.add_filter(ValidityFilter())
    pipeline.add_filter(SMARTSFilter(use_default_filters=True))

    # Optionally filter F/S
    if enable_structure_filter:
        pipeline.add_filter(StructureFilter(forbidden_atoms=["F", "S"]))

    return pipeline
