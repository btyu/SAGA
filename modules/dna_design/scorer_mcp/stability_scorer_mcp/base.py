from typing import List, Optional
from loguru import logger

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #


class Scorer(BaseScorer):

    def _score_stability(self, samples: List[str]) -> List[Optional[float]]:
        """Score the stability of input DNA sequences.
           
           Return the list of DNA stability score.
           
           Args:
               DNA sequences
           Returns:
               List of scores (0, inf)
        
        """
        proportion_list = []
        for sample in samples:
            str_dna = sample
            str_dna = str_dna.upper()
            count = str_dna.count('G') + str_dna.count('C')
            proportion_list.append( count / len(str_dna))
        return proportion_list

    @scorer(
        name="dna_stability",
        population_wise=False,  # Score individual candidates
        description= """DNA stability score (value range: 0.0 to 1.0). This objective measures the GC content of a DNA sequence, calculated as the proportion of guanine (G) and cytosine (C) bases relative to total sequence length. GC content is a key determinant of DNA stability because G-C pairs form three hydrogen bonds, compared to two for A-T pairs, leading to stronger and more thermally stable double-stranded structures. High values (>0.5) indicate GC-rich sequences with greater thermodynamic stability, which may enhance structural rigidity but also increase synthesis difficulty. Low values (<0.5) indicate AT-rich sequences with lower stability, which may be easier to unwind for transcription but more prone to structural fragility.""",
    )
    def score_stability_expression(self, samples: List[str]) -> List[Optional[float]]:
        """
        Score the stability of input DNA sequences, which is the proportion of guanine (G) and cytosine (C) bases relative to total sequence length.

        Args:
            samples: List of DNA sequences, where each sequence is a string of DNA bases
        
        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        
        return self._score_stability(samples)

