from typing import List, Optional
from loguru import logger

import numpy as np
from .scorer_utils import BaseScorer, scorer


# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #


class Scorer(BaseScorer):
    """
    Enformer-based scorers for DNA sequence enhancer design.
    
    This class provides scoring functions for DNA sequence optimization using the Enformer model
    to predict expression levels across different cell types.
    """
    
    def hamming_distance(self, str1: str, str2: str) -> int:
        if len(str1) != len(str2):
            return 0
        return sum(c1 != c2 for c1, c2 in zip(str1, str2))
    
    def _score_diversity(self, samples: List[str]) -> Optional[float]:
        """Score the diversity of input DNA sequences.
           
           Return the diversity score of the input DNA sequence as a group
           
           Args:
               DNA sequences
           Returns:
               A score (0, inf)
        
        """
        str_set = samples
        
        dist_list = []
        for i in range(len(str_set)):
            for j in range(len(str_set)):
                if i!=j:
                    dist_list.append(self.hamming_distance(str_set[i], str_set[j]))
        out_div = np.mean(dist_list)
        return round(out_div ,5)
    
    @scorer(
        name="dna_diversity",
        population_wise=True,  # Score individual candidates
        description = """Sequence diversity score (value range: 0 to +∞). This objective quantifies the average Hamming distance among a group of DNA sequences, providing a measure of how different the sequences are from one another. Diversity is important in DNA design tasks because it reduces redundancy, increases exploration of sequence space, and can improve the chances of identifying functional variants. High values indicate that the sequences are highly diverse, suggesting broad coverage of sequence space and reduced redundancy. Low values suggest that the sequences are very similar or nearly identical, implying limited diversity and potentially reduced effectiveness in exploring novel regulatory patterns.""",
    )
    def score_diversity_expression(self, samples: List[str]) -> Optional[float]:
        """Score the diversity of input DNA sequences, which is the average Hamming distance among a group of DNA sequences.
           
           Args:
               samples: List of DNA sequences
           Returns:
               A score [0, inf), which is the average Hamming distance among a group of DNA sequences.
        
        """
        return self._score_diversity(samples)
