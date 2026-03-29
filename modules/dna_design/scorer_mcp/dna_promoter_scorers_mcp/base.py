import os
from typing import List, Optional, Dict, Any, Tuple, Set
from functools import lru_cache

# Disable wandb login prompts for programmatic execution
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

from loguru import logger
import torch
import pandas as pd
import grelu.lightning
import grelu.data.dataset

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class Scorer(BaseScorer):
    
    def __init__(self):
        super().__init__()

        """Initialize the Enformer scorers with default model."""
        self.cell_types = ['K562', 'HEPG2', 'GM12878', 'SKNSH', 'A549']
        self.num_cell_types = len(self.cell_types)
        
        # Load Enformer model with default checkpoint
        default_checkpoint = os.path.join(CURRENT_FILE_DIR, "scorer_data", "epoch=20-step=22722.ckpt")
        self._load_model(default_checkpoint)
    
    def _load_model(self, checkpoint_path: str):
        """Load the Enformer model from checkpoint."""
        try:
            # Model configuration
            model_params = {
                'model_type': 'EnformerModel',
                'n_tasks': 5,  # Number of cell types to predict
                'crop_len': 0,  # No cropping of the model output
                'n_transformers': 1,  # Number of transformer layers
            }
            
            # Load from checkpoint with proper parameters
            self.model = grelu.lightning.LightningModel.load_from_checkpoint(
                checkpoint_path,
                model_params=model_params,
            )
            self.model.eval()
            
            logger.debug(f"Enformer model loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.critical(f"Failed to load Enformer model from {checkpoint_path}: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def _predict_expression(self, dna_sequences: Tuple[str]) -> List[List[Optional[float]]]:
        """Predict the log-fold change measured from the DNA sequences for all the five cell types.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        tmp_sequences = []
        result = []
        for seq in dna_sequences:
            if self._is_valid_dna_sequence(seq):
                result.append(0)  # To be replaced by the actual expression level
                tmp_sequences.append(seq)
            else:
                result.append([None] * self.num_cell_types)
        
        if len(tmp_sequences) == 0:
            return result
        
        # Create dataset
        df_raw = pd.DataFrame({'seq': tmp_sequences})
        # print(df_raw)
        if len(df_raw)==0:
            return []
        test_dataset = grelu.data.dataset.DFSeqDataset(df_raw)
        
        if torch.cuda.is_available():
            pred_data = self.model.predict_on_dataset(test_dataset, devices=0)
        else:
            pred_data = self.model.predict_on_dataset(test_dataset, devices='cpu')
        pred_data = pred_data[:, :, 0]  # Shape: [n_sequences, n_cell_types]
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
        """Predict the log-fold change measured from the DNA sequences for the given cell type identified by cell index.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences, cell type index
           Returns:
               List of scores (-inf,inf)
        
        """
        dna_seqs = samples
        
        # Get expression prediction
        predictions = self._predict_expression(tuple(dna_seqs))
        
        expressions = [pred[cell_index] for pred in predictions]
        
        return expressions

    @scorer(
        name="dna_k562_promoter_logfoldchange",
        population_wise=False,  # Score individual candidates
        description="""K562 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the K562 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in K562 cells. High values indicate sequences that are predicted to drive strong K562-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in K562, implying limited promoter potential or possible cell-type specificity elsewhere.""",
    )
    def score_k562_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Predict the log-fold change measured from the DNA sequences for the K562 cell type.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 0)
    
    @scorer(
        name="dna_hepg2_promoter_logfoldchange",
        population_wise=False,  # Score individual candidates
        description="""HepG2 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the HepG2 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in HepG2 cells. High values indicate sequences that are predicted to drive strong HepG2-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in HepG2, implying limited promoter potential or possible cell-type specificity elsewhere.""",
    )
    def score_hepg2_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Predict the log-fold change measured from the DNA sequences for the HepG2 cell type.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 1)
    
    @scorer(
        name="dna_gm12878_promoter_logfoldchange",
        population_wise=False,  # Score individual candidates
        description="""GM12878 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the GM12878 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in GM12878 cells. High values indicate sequences that are predicted to drive strong GM12878-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in GM12878, implying limited promoter potential or possible cell-type specificity elsewhere.""",
    )
    def score_gm12878_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Predict the log-fold change measured from the DNA sequences for the GM12878 cell type.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 2)
    
    @scorer(
        name="dna_sknsh_promoter_logfoldchange",
        population_wise=False,  # Score individual candidates
        description="""SKNSH promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the SKNSH cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in SKNSH cells. High values indicate sequences that are predicted to drive strong SKNSH-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in SKNSH, implying limited promoter potential or possible cell-type specificity elsewhere.""",
    )
    def score_sknsh_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Predict the log-fold change measured from the DNA sequences for the SKNSH cell type.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 3)
    
    @scorer(
        name="dna_a549_promoter_logfoldchange",
        population_wise=False,  # Score individual candidates
        description="""A549 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the A549 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in A549 cells. High values indicate sequences that are predicted to drive strong A549-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in A549, implying limited promoter potential or possible cell-type specificity elsewhere.""",
    )
    def score_a549_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Predict the log-fold change measured from the DNA sequences for the A549 cell type.
           
           Return the list of log-fold change levels as scores.
           
           Args:
               DNA sequences
           Returns:
               List of scores (-inf,inf)
        
        """
        return self._score_expression(samples, 4)
    
    def _is_valid_dna_sequence(self, sequence: str) -> bool:
        """Check if a sequence is a valid DNA sequence."""
        if not sequence or not isinstance(sequence, str):
            return False
        
        # Check if all characters are valid DNA bases
        valid_bases = {'A', 'T', 'G', 'C'}
        sequence_upper = sequence.upper()
        
        for base in sequence_upper:
            if base not in valid_bases:
                return False
        return True
