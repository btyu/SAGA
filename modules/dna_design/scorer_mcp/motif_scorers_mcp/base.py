from typing import List, Optional
from loguru import logger

from .scorer_utils import BaseScorer, scorer

import grelu.io.motifs
from grelu.interpret.motifs import scan_sequences

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #


class Scorer(BaseScorer):
    def __init__(self):
        super().__init__()

        self.jaspar_dbs = {
            None: grelu.io.motifs.get_jaspar(species=None),
            "human": grelu.io.motifs.get_jaspar(species="human"),
        }
        
    def _score_motif(self, samples: List[str], selected_spec = None) -> List[Optional[float]]:
        """Score motifs of the given DNA sequence(s)"""
        dna_seqs = samples
        result = []
        if selected_spec not in self.jaspar_dbs:
            self.jaspar_dbs[selected_spec] = grelu.io.motifs.get_jaspar(species=selected_spec)
        jaspar_db = self.jaspar_dbs[selected_spec]
        for seq in dna_seqs:
            motif_count = scan_sequences([seq], jaspar_db)
            motif_count_sum = motif_count['motif'].value_counts()
            if sum(motif_count_sum) != None:
                out = sum(motif_count_sum)
                out = round(out,4)
                result.append(out)
            else:
                result.append(0)
        return result
    
    @scorer(
        name="dna_motif_num",
        population_wise=False,  # Score individual candidates
        description="""Number of discovered DNA motifs (value range: 0 to +∞). We utilize a database known as JASPAR for matching, and this function contains motifs from all species. This objective quantifies how many transcription factor binding motifs are present in a DNA sequence. Motif counts reflect potential regulatory activity, since motifs are short sequence patterns recognized by transcription factors that control gene expression. A higher motif count suggests richer regulatory potential, while a lower count suggests fewer recognizable regulatory signals. High values indicate that the sequence contains many recognizable regulatory patterns, which may correspond to strong or complex transcriptional regulation. Low values (close to 0) suggest the sequence has few known motifs, implying limited transcription factor binding potential or non-regulatory sequence regions.""",
    )
    def score_motif_allspec(self, samples: List[str]) -> List[Optional[float]]:
        """Score based on the enrichment of all motifs from the input DNA sequences.
           
           Return a list of numbers representing the total number of discovered motifs for each DNA sequence/
           
           Args:
               DNA sequences
           Return:
               List of scores (0,+inf)
        
        """
        return self._score_motif(samples)
    
    @scorer(
        name="dna_motif_num_human",
        population_wise=False,  # Score individual candidates
        description="""Number of discovered DNA motifs (value range: 0 to +∞). We utilize a database known as JASPAR for matching, and this function contains motifs from the human. This objective quantifies how many transcription factor binding motifs are present in a DNA sequence. Motif counts reflect potential regulatory activity, since motifs are short sequence patterns recognized by transcription factors that control gene expression. A higher motif count suggests richer regulatory potential, while a lower count suggests fewer recognizable regulatory signals. High values indicate that the sequence contains many recognizable regulatory patterns, which may correspond to strong or complex transcriptional regulation. Low values (close to 0) suggest the sequence has few known motifs, implying limited transcription factor binding potential or non-regulatory sequence regions.""",
    )
    def score_motif_human(self, samples: List[str]) -> List[Optional[float]]:
        """Score based on the enrichment of human motifs from the input DNA sequences.
           
           Return a list of numbers representing the total number of discovered motifs for each DNA sequence/
           
           Args:
               DNA sequences
           Return:
               List of scores (0,+inf)
        
        """
        return self._score_motif(samples, 'human')
