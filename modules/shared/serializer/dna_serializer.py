"""
DNA sequence serializer for genetic candidates.

This serializer extracts the DNA sequence string from a candidate's representation field
for use with MCP scorers that expect DNA sequence strings.
"""

from scileo_agent.core.registry.serializer_registry import (
    Serializer, register_serializer_class
)
from scileo_agent.core.data_models import Candidate


@register_serializer_class(
    name="dna_serializer", 
    description="Serializes genetic candidates by extracting DNA sequences from representation field"
)
class DnaSerializer(Serializer):
    """
    Serializer for genetic candidates with DNA sequence representations.
    
    Extracts the DNA sequence string from candidate.representation for use with 
    MCP scorers that expect DNA sequence strings as input.
    """
    
    @property
    def sample_schema(self) -> str:
        """Return the data type/format of the serialized output."""
        return "str"
    
    @property
    def sample_description(self) -> str:
        """Return a description of what the serialized data represents."""
        return "the DNA sequence string containing A, T, G, C nucleotides"
    
    def serialize(self, candidate: Candidate) -> str:
        """
        Serialize a Candidate instance to a DNA sequence string.
        
        Args:
            candidate: The Candidate instance to serialize
            
        Returns:
            DNA sequence string from candidate.representation
        """
        if not hasattr(candidate, 'representation') or candidate.representation is None:
            raise ValueError("Candidate must have a 'representation' field with DNA sequence string")
        
        dna_sequence = str(candidate.representation)
        
        return dna_sequence.upper()  # Return uppercase for consistency
    
    def deserialize(self, data: str) -> Candidate:
        """
        Deserialize a DNA sequence string back to a Candidate instance.
        
        Args:
            data: The DNA sequence string to convert back to Candidate
            
        Returns:
            Candidate instance with the DNA sequence string as representation
        """
        if not isinstance(data, str):
            raise ValueError(f"Expected string DNA sequence, got {type(data)}")
        
        return Candidate(representation=data.upper())