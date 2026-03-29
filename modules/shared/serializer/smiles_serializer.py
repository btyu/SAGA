"""
SMILES serializer for molecular candidates.

This serializer extracts the SMILES string from a candidate's representation field
for use with MCP scorers that expect molecular SMILES strings.
"""

from scileo_agent.core.registry.serializer_registry import (
    Serializer, register_serializer_class
)
from scileo_agent.core.data_models import Candidate


@register_serializer_class(
    name="smiles_serializer",
    description="Serializes molecular candidates by extracting SMILES strings from representation field"
)
class SmilesSerializer(Serializer):
    """
    Serializer for molecular candidates with SMILES representations.
    
    Extracts the SMILES string from candidate.representation for use with 
    MCP scorers that expect molecular SMILES strings as input.
    """
    
    @property
    def sample_schema(self) -> str:
        """Return the data type/format of the serialized output."""
        return "str"
    
    @property
    def sample_description(self) -> str:
        """Return a description of what the serialized data represents."""
        return "the SMILES string of a molecule"
    
    def serialize(self, candidate: Candidate) -> str:
        """
        Serialize a Candidate instance to a SMILES string.
        
        Args:
            candidate: The Candidate instance to serialize
            
        Returns:
            SMILES string from candidate.representation
        """
        if not hasattr(candidate, 'representation') or candidate.representation is None:
            raise ValueError("Candidate must have a 'representation' field with SMILES string")
        
        return str(candidate.representation)
    
    def deserialize(self, data: str) -> Candidate:
        """
        Deserialize a SMILES string back to a Candidate instance.
        
        Args:
            data: The SMILES string to convert back to Candidate
            
        Returns:
            Candidate instance with the SMILES string as representation
        """
        if not isinstance(data, str):
            raise ValueError(f"Expected string SMILES, got {type(data)}")
        
        return Candidate(representation=data)