import os
from typing import List, Optional, Dict, Any, Tuple, Set
from loguru import logger
from functools import lru_cache

# Disable wandb login prompts for programmatic execution
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import grelu.lightning
import grelu.data.dataset
import torch
import pandas as pd

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class Scorer(BaseScorer):
    """
    Enformer-based scorers for DNA sequence enhancer design.
    
    This class provides scoring functions for DNA sequence optimization using the Enformer model
    to predict expression levels across different cell types.
    """
    
    def __init__(self):
        super().__init__()

        """Initialize the Enformer scorers with default model."""
        self.cell_types = ["hepg2", "k562", "SKNSH"]
        self.num_cell_types = len(self.cell_types)
        
        # Load Enformer model with default checkpoint
        default_checkpoint = "epoch=13-step=34748.ckpt"
        self._load_model((os.path.join(CURRENT_FILE_DIR, "scorer_data", default_checkpoint)))
    
    def _load_model(self, checkpoint_path: str):
        """Load the Enformer model from checkpoint."""
        try:
            # Model configuration
            model_params = {
                'model_type': 'EnformerModel',
                'n_tasks': 3,  # Number of cell types to predict
                'crop_len': 0,  # No cropping of the model output
                'n_transformers': 1,  # Number of transformer layers
            }
            
            # Load from checkpoint with proper parameters
            self.model = grelu.lightning.LightningModel.load_from_checkpoint(
                checkpoint_path,
                model_params=model_params,
            )
            self.model.eval()
            
            logger.info(f"Enformer model loaded from {checkpoint_path}")
            
        except Exception as e:
            logger.error(f"Failed to load Enformer model from {checkpoint_path}: {e}")
            raise
    
    @lru_cache(maxsize=1000)
    def _predict_expression(self, dna_sequences: Tuple[str]) -> List[List[Optional[float]]]:
        """Predict the MPRA expression levels measured from the DNA sequences across three cell lines.
           
           Return the list of MPRA expression levels as scores.
           
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
        """Predict the MPRA expression levels measured from the DNA sequences for the given cell line based on cell index.
           
           Return the list of MPRA expression levels as scores.
           
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
        name="dna_hepg2_enhancer_MPRA_expression",
        population_wise=False,  # Score individual candidates
        description="HepG2 MPRA expression score (value range: -∞ to +∞). MPRA (Massively Parallel Reporter Assay) quantifies gene expression driven by DNA sequences using massively parallel sequencing to measure how well different sequences drive reporter gene expression. This score evaluates DNA enhancer sequences based on their predicted expression levels in HepG2 cells (human hepatocellular carcinoma cell line). Higher scores indicate stronger enhancer activity and greater gene expression in liver-like cellular contexts. Lower scores suggest weaker enhancer activity or potential silencing effects. This metric is essential for designing DNA sequences that need to function specifically in hepatic environments or liver-related therapeutic applications.",
    )
    def score_hepg2_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Calculate HepG2 MPRA expression score for DNA enhancer sequences.
        
        Uses an Enformer-based model to predict expression levels of DNA sequences in HepG2 cells.
        The model analyzes sequence patterns to predict enhancer activity specific to hepatic cellular contexts.
        
        - Higher scores (> 0): Strong enhancer activity, increased gene expression
        - Lower scores (< 0): Weak enhancer activity or silencing effects
        - Score = None: Invalid DNA sequences or computation errors
        
        This scoring is particularly valuable for liver-targeted therapeutic applications and 
        understanding hepatic gene regulation mechanisms.
           
        Args:
            samples: List of input samples, where each sample is a DNA sequence string
        
        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        
        """
        return self._score_expression(samples, 0)
    
    @scorer(
        name="dna_k562_enhancer_MPRA_expression",
        population_wise=False,  # Score individual candidates
        description="K562 MPRA expression score (value range: -∞ to +∞). MPRA (Massively Parallel Reporter Assay) quantifies gene expression driven by DNA sequences using massively parallel sequencing to measure how well different sequences drive reporter gene expression. This score evaluates DNA enhancer sequences based on their predicted expression levels in K562 cells (human erythroleukemic cell line). Higher scores indicate stronger enhancer activity and greater gene expression in hematopoietic cellular contexts. Lower scores suggest weaker enhancer activity or potential silencing effects. This metric is crucial for designing DNA sequences that need to function in blood cell lineages or hematological therapeutic applications.",
    )
    def score_k562_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Calculate K562 MPRA expression score for DNA enhancer sequences.
        
        Uses an Enformer-based model to predict expression levels of DNA sequences in K562 cells.
        The model analyzes sequence patterns to predict enhancer activity specific to erythroleukemic cellular contexts.
        
        - Higher scores (> 0): Strong enhancer activity, increased gene expression
        - Lower scores (< 0): Weak enhancer activity or silencing effects
        - Score = None: Invalid DNA sequences or computation errors
        
        This scoring is particularly valuable for hematological research and blood-related therapeutic applications
        where understanding gene regulation in blood cell lineages is essential.
           
        Args:
            samples: List of input samples, where each sample is a DNA sequence string
        
        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        
        """
        return self._score_expression(samples, 1)
    
    @scorer(
        name="dna_sknsh_enhancer_MPRA_expression",
        population_wise=False,  # Score individual candidates
        description="SKNSH MPRA expression score (value range: -∞ to +∞). MPRA (Massively Parallel Reporter Assay) quantifies gene expression driven by DNA sequences using massively parallel sequencing to measure how well different sequences drive reporter gene expression. This score evaluates DNA enhancer sequences based on their predicted expression levels in SKNSH cells (human neuroblastoma cell line). Higher scores indicate stronger enhancer activity and greater gene expression in neuronal cellular contexts. Lower scores suggest weaker enhancer activity or potential silencing effects. This metric is essential for designing DNA sequences that need to function in neural tissues or neurological therapeutic applications.",
    )
    def score_sknsh_expression(self, samples: List[str]) -> List[Optional[float]]:
        """Calculate SKNSH MPRA expression score for DNA enhancer sequences.
        
        Uses an Enformer-based model to predict expression levels of DNA sequences in SKNSH cells.
        The model analyzes sequence patterns to predict enhancer activity specific to neuroblastoma cellular contexts.
        
        - Higher scores (> 0): Strong enhancer activity, increased gene expression
        - Lower scores (< 0): Weak enhancer activity or silencing effects
        - Score = None: Invalid DNA sequences or computation errors
        
        This scoring is particularly valuable for neurological research and neural therapeutic applications
        where understanding gene regulation in neural cell lineages is critical.
           
        Args:
            samples: List of input samples, where each sample is a DNA sequence string
        
        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        
        """
        return self._score_expression(samples, 2)
    
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
