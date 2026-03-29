import os
from typing import List, Optional, Dict, Any, Tuple, Set
from pathlib import Path

import argparse
import torch
from rdkit import Chem
from packaging import version

import chemprop

try:
    # Try chemprop 1.6.1 import path
    from chemprop.data.scaler import StandardScaler, AtomBondScaler
except (ImportError, ModuleNotFoundError):
    # Fall back to older chemprop versions
    from chemprop.data import StandardScaler, AtomBondScaler

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class Scorer(BaseScorer):
    """Collection of Chemprop-based scoring functions.

    This class preloads two Chemprop models at initialization:
    - Primary cell toxicity model
    - Staphylococcus aureus activity model
    """

    # TODO: This is not complete yet: Lack the oracle checkpoints

    def __init__(self):
        # Call parent constructor to set up registry
        super().__init__()

        # Allowlist classes for PyTorch 2.6 safe loading
        self._safe_globals = [argparse.Namespace, StandardScaler, AtomBondScaler]
        # Only add safe globals if PyTorch version >= 2.6.0
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            if hasattr(torch.serialization, 'add_safe_globals'):
                torch.serialization.add_safe_globals(self._safe_globals)

        # Checkpoints (relative to module root)
        module_root = Path(__file__).resolve().parent
        models_dir = module_root / "scorer_data" / "antibiotics" / "models"

        print(f"Loading Chemprop models from: {models_dir}")

        # Collect all available checkpoint model.pt paths (1..20)
        tox_root = models_dir / "primary_cell_toxicity_model"
        staph_root = models_dir / "staph_aureus_model"
        self._tox_checkpoint_paths = self._collect_checkpoint_paths(
            tox_root, in_train_subdir=True
        )
        self._staph_checkpoint_paths = self._collect_checkpoint_paths(
            staph_root, in_train_subdir=False
        )

        if not self._tox_checkpoint_paths:
            raise FileNotFoundError("No primary_cell_toxicity_model checkpoints found.")
        if not self._staph_checkpoint_paths:
            raise FileNotFoundError("No staph_aureus_model checkpoints found.")

        # Preload a single ensemble per task (one load, internal averaging)
        self._tox_model_sets = [self._load_ensemble(self._tox_checkpoint_paths)]
        self._staph_model_sets = [self._load_ensemble(self._staph_checkpoint_paths)]

    def _build_predict_args(self, checkpoint_path: str):
        args_list = [
            "--test_path",
            "/dev/null",
            "--preds_path",
            "/dev/null",
            "--checkpoint_path",
            checkpoint_path,
        ]
        predict_args = chemprop.args.PredictArgs().parse_args(args_list)

        # Load training args and align critical prediction-time flags
        # PyTorch 2.6+ uses safe_globals as context manager, older versions use add_safe_globals
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            with torch.serialization.safe_globals(self._safe_globals):
                train_args = chemprop.utils.load_args(checkpoint_path)
        else:
            train_args = chemprop.utils.load_args(checkpoint_path)

        # Mirror relevant settings used during training
        for attr in [
            "features_scaling",
            "atom_descriptors",
            "bond_descriptors",
            "features_generator",
            "features_path",
            "atom_features_path",
            "bond_features_path",
        ]:
            if hasattr(train_args, attr):
                setattr(predict_args, attr, getattr(train_args, attr))

        return predict_args

    def _load_models(self, predict_args):
        # PyTorch 2.6+ uses safe_globals as context manager, older versions use add_safe_globals
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            with torch.serialization.safe_globals(self._safe_globals):
                return chemprop.train.load_model(args=predict_args)
        else:
            return chemprop.train.load_model(args=predict_args)

    def _build_ensemble_predict_args(self, checkpoint_paths: List[str]):
        # Build args with a single checkpoint to satisfy chemprop validation
        # Then override with the full list for true ensembling
        args_list = [
            "--test_path",
            "/dev/null",
            "--preds_path",
            "/dev/null",
            "--checkpoint_path",
            checkpoint_paths[0],
        ]
        predict_args = chemprop.args.PredictArgs().parse_args(args_list)

        # Mirror relevant settings from the first model's training config
        # PyTorch 2.6+ uses safe_globals as context manager, older versions use add_safe_globals
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            with torch.serialization.safe_globals(self._safe_globals):
                train_args = chemprop.utils.load_args(checkpoint_paths[0])
        else:
            train_args = chemprop.utils.load_args(checkpoint_paths[0])

        for attr in [
            "features_scaling",
            "atom_descriptors",
            "bond_descriptors",
            "features_generator",
            "features_path",
            "atom_features_path",
            "bond_features_path",
        ]:
            if hasattr(train_args, attr):
                setattr(predict_args, attr, getattr(train_args, attr))

        # Configure ensemble and performance-related flags
        predict_args.checkpoint_paths = checkpoint_paths
        predict_args.checkpoint_path = None
        predict_args.use_gpu = torch.cuda.is_available()
        # Larger batch size for faster inference (adjust if memory-limited)
        predict_args.batch_size = 1024 if torch.cuda.is_available() else 256
        return predict_args

    def _load_ensemble(self, checkpoint_paths: List[str]) -> Tuple:
        args = self._build_ensemble_predict_args(checkpoint_paths)
        # PyTorch 2.6+ uses safe_globals as context manager, older versions use add_safe_globals
        if version.parse(torch.__version__) >= version.parse("2.6.0"):
            with torch.serialization.safe_globals(self._safe_globals):
                model_objects = chemprop.train.load_model(args=args)
        else:
            model_objects = chemprop.train.load_model(args=args)
        return (args, model_objects)

    def _collect_checkpoint_paths(
        self, model_root: Path, in_train_subdir: bool, max_checkpoints: int = 20
    ) -> List[str]:
        paths: List[str] = []
        for i in range(1, 21):
            if len(paths) >= max_checkpoints:
                break
            base = model_root / "train" if in_train_subdir else model_root
            ckpt = base / f"checkpoints{i}" / "fold_0" / "model_0" / "model.pt"
            if ckpt.exists():
                paths.append(str(ckpt))
        return paths

    def _load_model_sets(self, checkpoint_paths: List[str]) -> List[Tuple]:
        model_sets: List[Tuple] = []
        for p in checkpoint_paths:
            args = self._build_predict_args(p)
            # PyTorch 2.6+ uses safe_globals as context manager, older versions use add_safe_globals
            if version.parse(torch.__version__) >= version.parse("2.6.0"):
                with torch.serialization.safe_globals(self._safe_globals):
                    model_objects = chemprop.train.load_model(args=args)
            else:
                model_objects = chemprop.train.load_model(args=args)
            model_sets.append((args, model_objects))
        return model_sets

    def _predict(self, smiles: List[str], model_sets: List[Tuple]) -> List[List[float]]:
        rows = [[s] for s in smiles]
        aggregated: Optional[List[List[float]]] = None
        num_sets = 0
        with torch.inference_mode():
            for predict_args, model_objects in model_sets:
                preds = chemprop.train.make_predictions(
                    args=predict_args, smiles=rows, model_objects=model_objects
                )
            aggregated = [[float(x) for x in row] for row in preds]

            num_sets += 1
        if aggregated is None or num_sets == 0:
            return []
        for i in range(len(aggregated)):
            for j in range(len(aggregated[i])):
                aggregated[i][j] /= float(num_sets)
        return aggregated

    def _predict_on_candidates(
        self, samples: List[str], model_sets: List[Tuple]
    ) -> List[Optional[float]]:
        valid_smiles = []
        valid_indices = []

        for i, sample in enumerate(samples):
            s = sample
            mol = Chem.MolFromSmiles(s)  # pylint: disable=no-member
            if mol is None:
                continue
            valid_smiles.append(s)
            valid_indices.append(i)

        if valid_smiles:
            preds = self._predict(valid_smiles, model_sets)
            # Take the first task prediction per molecule
            valid_scores = [float(p[0]) for p in preds]
        else:
            valid_scores = []

        results: List[Optional[float]] = [None] * len(samples)
        for idx, score in zip(valid_indices, valid_scores):
            results[idx] = score

        return results

    @scorer(
        name="toxicity_safety_chemprop",
        population_wise=False,
        description="Primary cell toxicity safety score (value range: 0.0 to 1.0). "
        "This score is computed as (1 - Primary cell toxicity probability) where the toxicity probability is predicted by a Chemprop ensemble model trained on primary cell toxicity data. "
        "The normalization inverts the toxicity prediction so higher scores indicate better safety profiles. "
        "High scores (>0.8) indicate excellent safety with low predicted toxicity to human primary cells, while low scores (<0.3) suggest high cytotoxicity that could lead to adverse effects in patients. "
        "This metric is crucial for drug safety assessment as primary cell toxicity often correlates with in vivo toxicity and can predict potential side effects in clinical development.",
    )
    def score_primary_cell_toxicity(self, samples: List[str]) -> List[Optional[float]]:
        # TODO: Make sure the docstring is correct
        """
        Compute the primary cell toxicity score for each sample, which is defined as 1 - primary cell toxicity probability.

        Args:
            samples: List of input samples, where each sample is a SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """

        raw = self._predict_on_candidates(samples, self._tox_model_sets)
        return [None if s is None else 1.0 - float(s) for s in raw]

    # @scorer(
    #     name="staph_aureus_chemprop",
    #     population_wise=False,
    #     description="Staphylococcus aureus antibacterial activity score (value range: 0.0 to 1.0). "
    #     "This score represents the predicted probability of inhibitory activity against S. aureus bacteria, as determined by a Chemprop ensemble model trained on experimental antibacterial screening data. "
    #     "The score directly reflects the likelihood of achieving meaningful minimum inhibitory concentration (MIC) values against this important pathogen. "
    #     "High scores (>0.7) indicate strong predicted antibacterial activity suitable for antibiotic development, while low scores (<0.3) suggest minimal or no antibacterial effect. "
    #     "S. aureus is a critical target for antibiotic discovery due to its clinical importance and propensity for developing resistance, making this score valuable for prioritizing compounds with therapeutic potential.",
    # )
    # def score_staph_aureus(self, samples: List[str]) -> List[Optional[float]]:
    #     # TODO: Make sure the docstring is correct
    #     """
    #     Compute the staphylococcus aureus antibacterial activity score for each sample, which is defined as the predicted probability of inhibitory activity against S. aureus bacteria.

    #     Args:
    #         samples: List of input samples, where each sample is a SMILES string of a molecule

    #     Returns:
    #         List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
    #     """
    #     return self._predict_on_candidates(samples, self._staph_model_sets)
