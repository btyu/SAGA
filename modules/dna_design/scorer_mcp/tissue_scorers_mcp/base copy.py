import os
from typing import List, Optional

# Disable wandb login prompts for programmatic execution
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from loguru import logger
import pandas as pd
from typing import List, Optional, Tuple
from functools import lru_cache
import torch
import grelu.data.dataset
import grelu.resources

from .scorer_utils import BaseScorer, scorer


# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #


class Scorer(BaseScorer):
    def __init__(self):
        super().__init__()
        
        """Initialize the Enformer scorers with default model."""
        self.cell_types = ["hepg2", "k562", "SKNSH"]
        self.num_cell_types = len(self.cell_types)
        
        # Load Enformer model with default checkpoint
        self._load_model()
    
    def _load_model(self):
        """Load the Enformer model from checkpoint."""
        try:
            self.model = grelu.resources.load_model(
                    project="enformer",
                    model_name="human",
                )
            
            logger.debug(f"Pre-trained Enformer model loaded for tissue-specific expression.")
            
        except Exception as e:
            logger.critical(f"Failed to load Enformer model for tissue-specific expression")
            raise
    
    @lru_cache(maxsize=1000)
    def _predict_expression(self, dna_sequences: Tuple[str]) -> List[List[Optional[float]]]:
        """Predict the CAGE-seq expression of given DNA sequences across different tissues.
           
           Args:
               DNA sequences
           Return:
               Array of scores (-inf,inf)
        
        """
        tmp_sequences = []
        result = []
        for seq in dna_sequences:
            if self._is_valid_dna_sequence(seq):
                result.append(0)  # To be replaced by the actual expression level
                tmp_sequences.append('N'*((196608-len(seq))//2) + seq + 'N'*((196608-len(seq))//2))
            else:
                result.append([None] * self.num_cell_types)
        
        if len(tmp_sequences) == 0:
            return result
        
        # Create dataset
        df_raw = pd.DataFrame({'seq': tmp_sequences})
        test_dataset = grelu.data.dataset.DFSeqDataset(df_raw)
        
        if torch.cuda.is_available():
            pred_data = self.model.predict_on_dataset(test_dataset, devices=0)
        else:
            pred_data = self.model.predict_on_dataset(test_dataset, devices='cpu')
        pred_data = pred_data.mean(axis=2)  # Shape: [n_sequences, n_tracks]
        assert pred_data.shape[0] == len(tmp_sequences)
        
        # Replace the 0 with the actual expression level
        k = 0
        for idx, seq_result in enumerate(result):
            if seq_result == 0:
                seq_result = list(pred_data[k, :])
                result[idx] = seq_result
                k += 1

        return result
    
    def _score_expression(self, samples: List[str], cell_index: int) -> List[Optional[float]]:
        """Score based on expression level in a given cell type assigned by cell_index.
           Args:
               DNA sequences, cell type index
           Return:
               List of scores (-inf,inf)
        
        """
        dna_seqs = samples
        
        # Get expression prediction
        predictions = self._predict_expression(tuple(dna_seqs))
        
        expressions = [pred[cell_index] for pred in predictions]
        
        return expressions
    
    def _is_valid_dna_sequence(self, sequence: str) -> bool:
        """Check if a sequence is a valid DNA sequence.
           
           Args:
               DNA sequences
           Return:
               True or False
        """
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Check if all characters are valid DNA bases
        valid_bases = {'A', 'T', 'G', 'C'}
        sequence_upper = sequence.upper()
        return all(base in valid_bases for base in sequence_upper)
    
    @scorer(
        name="dna_hepg2_tissue_cage_expression",
        population_wise=False,  # Score individual candidates
        description="""HepG2 liver tissue CAGE-seq expression score (value range: -∞ to ∞). We utilized the pre-trained model Enformer and liver track for prediction. This objective measures the transcriptional activity of DNA sequences in HepG2 cells, a liver-derived cell line, using CAGE-seq (Cap Analysis of Gene Expression). Since HepG2 originates from liver tissue, sequences with liver-specific regulatory elements (such as enhancers and promoters) should show higher expression scores. This allows evaluation of whether a candidate DNA sequence has functional regulatory potential in liver-related contexts. High values indicate strong transcriptional activity in HepG2/liver tissue, suggesting that the sequence functions as a potent enhancer or promoter in hepatic regulatory programs. Low or negative values suggest weak or no detectable transcriptional initiation in HepG2 cells, implying limited or absent liver-specific regulatory activity.""",
    )
    def score_hepg2_tissue_expression(self, samples: List[str]) -> Optional[float]:
        """Score based on expression level in hepg2 cell type.
           
           Return the Averaged CAGE-seq expression for Liver tissue, which is the source tissue of hepg2 cell type.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4686) #liver
    
    @scorer(
        name="dna_k562_tissue_cage_expression",
        population_wise=False,  # Score individual candidates
        description="""K562 blood tissue CAGE-seq expression score (value range: -∞ to ∞). We utilized the pre-trained model Enformer and blood track for prediction. This objective measures the transcriptional activity of DNA sequences in K562 cells, a blood-derived diseased cell line, using CAGE-seq (Cap Analysis of Gene Expression). Since K562 originates from blood tissue, sequences with blood-specific regulatory elements (such as enhancers and promoters) should show higher expression scores. This allows evaluation of whether a candidate DNA sequence has functional regulatory potential in blood-related contexts. High values indicate strong transcriptional activity in K562/blood tissue, suggesting that the sequence functions as a potent enhancer or promoter in hepatic regulatory programs. Low or negative values suggest weak or no detectable transcriptional initiation in K562 cells, implying limited or absent blood-specific regulatory activity.""",
    )
    def score_k562_tissue_expression(self, samples: List[str]) -> Optional[float]:
        """Score based on expression level in k562 cell type.
           
           Return the Averaged CAGE-seq expression for Blood tissue, which is the source tissue of k562 cell type.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4950) #blood
    
    @scorer(
        name="dna_SKNSH_tissue_cage_expression",
        population_wise=False,  # Score individual candidates
        description="""SKNSH brain tissue CAGE-seq expression score (value range: -∞ to ∞). We utilized the pre-trained model Enformer and brain track for prediction. This objective measures the transcriptional activity of DNA sequences in SKNSH cells, a blood-derived diseased cell line, using CAGE-seq (Cap Analysis of Gene Expression). Since SKNSH originates from brain tissue, sequences with brain-specific regulatory elements (such as enhancers and promoters) should show higher expression scores. This allows evaluation of whether a candidate DNA sequence has functional regulatory potential in brain-related contexts. High values indicate strong transcriptional activity in SKNSH/brain tissue, suggesting that the sequence functions as a potent enhancer or promoter in hepatic regulatory programs. Low or negative values suggest weak or no detectable transcriptional initiation in SKNSH cells, implying limited or absent brain-specific regulatory activity.""",
    )
    def score_SKNSH_tissue_expression(self, samples: List[str]) -> Optional[float]:
        """Score based on expression level in SKNSH cell type.
           
           Return the Averaged CAGE-seq expression for Brain tissue, which is the source tissue of SKNSH cell type.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4680) #brain
    
    @scorer(
        name="dna_sknsh_tissue_cage_expression",
        population_wise=False,  # Score individual candidates
        description="""SKNSH brain tissue CAGE-seq expression score (value range: -∞ to ∞). We utilized the pre-trained model Enformer and brain track for prediction. This objective measures the transcriptional activity of DNA sequences in SKNSH cells, a blood-derived diseased cell line, using CAGE-seq (Cap Analysis of Gene Expression). Since SKNSH originates from brain tissue, sequences with brain-specific regulatory elements (such as enhancers and promoters) should show higher expression scores. This allows evaluation of whether a candidate DNA sequence has functional regulatory potential in brain-related contexts. High values indicate strong transcriptional activity in SKNSH/brain tissue, suggesting that the sequence functions as a potent enhancer or promoter in hepatic regulatory programs. Low or negative values suggest weak or no detectable transcriptional initiation in SKNSH cells, implying limited or absent brain-specific regulatory activity.""",
    )
    def score_SKNSH_tissue_expression(self, samples: List[str]) -> Optional[float]:
        """Score based on expression level in SKNSH cell type.
           
           Return the Averaged CAGE-seq expression for Brain tissue, which is the source tissue of SKNSH cell type.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4680) #brain
    
    @scorer(
        name="dna_a549_tissue_cage_expression",
        population_wise=False,  # Score individual candidates
        description="""A549 lung tissue CAGE-seq expression score (value range: -∞ to ∞). We utilized the pre-trained model Enformer and lung track for prediction. This objective measures the transcriptional activity of DNA sequences in A549 cells, a lung-derived cell line, using CAGE-seq (Cap Analysis of Gene Expression). Since A549 originates from lung tissue, sequences with lung-specific regulatory elements (such as enhancers and promoters) should show higher expression scores. This allows evaluation of whether a candidate DNA sequence has functional regulatory potential in lung-related contexts. High values indicate strong transcriptional activity in A549/lung tissue, suggesting that the sequence functions as a potent enhancer or promoter in hepatic regulatory programs. Low or negative values suggest weak or no detectable transcriptional initiation in A549 cells, implying limited or absent lung-specific regulatory activity.""",
    )
    def score_a549_tissue_expression(self, samples: List[str]) -> Optional[float]:
        """Score based on expression level in A549 cell type.
           
           Return the Averaged CAGE-seq expression for Lung tissue, which is the source tissue of A549 cell type.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4687) #lung
    
    @scorer(
        name="dna_gm12878_tissue_cage_expression",
        population_wise=False,  # Score individual candidates
        description="""GM12878 PBMC tissue CAGE-seq expression score (value range: -∞ to ∞). We utilized the pre-trained model Enformer and PBMC track for prediction. This objective measures the transcriptional activity of DNA sequences in GM12878 cells, a PBMC-derived cell line, using CAGE-seq (Cap Analysis of Gene Expression). Since GM12878 originates from PBMC tissue, sequences with PBMC-specific regulatory elements (such as enhancers and promoters) should show higher expression scores. This allows evaluation of whether a candidate DNA sequence has functional regulatory potential in PBMC-related contexts. High values indicate strong transcriptional activity in GM12878/PBMC tissue, suggesting that the sequence functions as a potent enhancer or promoter in hepatic regulatory programs. Low or negative values suggest weak or no detectable transcriptional initiation in GM12878 cells, implying limited or absent PBMC-specific regulatory activity.""",
    )
    def score_gm12878_tissue_expression(self, samples: List[str]) -> Optional[float]:
        """Score based on expression level in GM12878 cell type.
           
           Return the Averaged CAGE-seq expression for PBMC tissue, which is the source tissue of GM12878 cell type.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4765) #PBMC

