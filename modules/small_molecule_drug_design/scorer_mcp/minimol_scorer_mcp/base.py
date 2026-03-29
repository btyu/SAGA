import os
import threading
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAveragePrecision,
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
)
from rdkit import Chem
from minimol import Minimol

from .scorer_utils import BaseScorer, scorer

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


def _make_activation(name: str) -> nn.Module:
    """Create activation function module."""
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.01)
    if name == "gelu":
        return nn.GELU()
    raise ValueError(f"Unknown activation_function: {name}")


class MLPClassifier(pl.LightningModule):
    """Multi-layer perceptron classifier for antibacterial activity prediction."""

    def __init__(
        self,
        input_dim: int,
        num_tasks: int,
        dim_size: int = 512,
        shrinking_scale: float = 1.0,
        num_layers: int = 2,
        dropout_rate: float = 0.2,
        activation_function: str = "relu",
        use_batch_norm: bool = False,
        learning_rate: float = 1e-3,
        L1_weight_norm: float = 0.0,
        L2_weight_norm: float = 0.0,
        scheduler_step_size: int = 10,
        scheduler_gamma: float = 0.5,
        threshold: float = 0.5,
        fold_index: int = 0,
        optimized_thresholds: Optional[List[float]] = None,
        task_indices: Optional[List[int]] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.input_dim = input_dim
        self.num_tasks = num_tasks
        self.fold_index = fold_index
        # Store original task indices for logging (default to 0-based if not provided)
        self.task_indices = task_indices if task_indices is not None else list(
            range(num_tasks))

        # Set up thresholds - use optimized if provided, otherwise default
        if optimized_thresholds is not None:
            self.task_thresholds = optimized_thresholds
        else:
            self.task_thresholds = [threshold] * num_tasks

        layers: List[nn.Module] = []
        in_dim = input_dim
        hidden_dim = dim_size
        act = _make_activation(activation_function)

        for layer_idx in range(num_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            in_dim = hidden_dim
            hidden_dim = max(1, int(hidden_dim * shrinking_scale))

        self.backbone = nn.Sequential(*layers) if layers else nn.Identity()
        self.head = nn.Linear(in_dim, num_tasks)

        # Metrics per task
        self._build_metrics()

    def load_optimized_thresholds(self, thresholds: List[float]) -> None:
        """Load optimized thresholds for each task."""
        if len(thresholds) != self.num_tasks:
            raise ValueError(
                f"Expected {self.num_tasks} thresholds, got {len(thresholds)}")
        self.task_thresholds = thresholds

    def get_task_threshold(self, task_idx: int) -> float:
        """Get threshold for specific task."""
        return self.task_thresholds[task_idx]

    def _build_metrics(self) -> None:
        self.val_metrics = nn.ModuleDict({})
        self.test_metrics = nn.ModuleDict({})
        for split, container in (("val", self.val_metrics),
                                 ("test", self.test_metrics)):
            container["auprc"] = nn.ModuleList(
                [BinaryAveragePrecision() for _ in range(self.num_tasks)])
            container["auroc"] = nn.ModuleList(
                [BinaryAUROC() for _ in range(self.num_tasks)])
            container["acc"] = nn.ModuleList(
                [BinaryAccuracy() for _ in range(self.num_tasks)])
            container["precision"] = nn.ModuleList(
                [BinaryPrecision() for _ in range(self.num_tasks)])
            container["recall"] = nn.ModuleList(
                [BinaryRecall() for _ in range(self.num_tasks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone is not None:
            x = self.backbone(x)
        return self.head(x)

    def _compute_masked_bce_loss(self, logits: torch.Tensor,
                                 targets: torch.Tensor,
                                 mask: torch.Tensor) -> torch.Tensor:
        # mask: bool [B, T]
        if mask.numel() == 0 or mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        loss = F.binary_cross_entropy_with_logits(logits[mask], targets[mask])
        if self.hparams.L1_weight_norm and self.hparams.L1_weight_norm > 0:
            l1 = 0.0
            for p in self.parameters():
                l1 = l1 + p.abs().sum()
            loss = loss + self.hparams.L1_weight_norm * l1
        return loss

    def training_step(self, batch, batch_idx):
        x, y, mask, _ = batch
        logits = self(x)
        loss = self._compute_masked_bce_loss(logits, y, mask)
        self.log(f"train/loss_fold{self.fold_index}",
                 loss,
                 prog_bar=True,
                 on_step=False,
                 on_epoch=True,
                 batch_size=x.size(0))
        return loss

    def on_validation_epoch_start(self) -> None:
        """Reset validation metrics at start of each epoch."""
        for metric_list in self.val_metrics.values():
            for metric in metric_list:
                metric.reset()

    def on_test_epoch_start(self) -> None:
        """Reset test metrics at start of each epoch."""
        for metric_list in self.test_metrics.values():
            for metric in metric_list:
                metric.reset()

    @torch.no_grad()
    def _update_split_metrics(self, split: str, logits: torch.Tensor,
                              targets: torch.Tensor,
                              mask: torch.Tensor) -> None:
        """Update metrics for the given split. Call _compute_split_metrics to get values."""
        probs = torch.sigmoid(logits)
        metrics_container = self.val_metrics if split == "val" else self.test_metrics

        for task_idx in range(self.num_tasks):
            task_mask = mask[:, task_idx]
            if task_mask.sum() < 2:
                continue

            y_true = targets[task_mask, task_idx].detach()
            y_prob = probs[task_mask, task_idx].detach()

            # Check if we have both classes for AUROC/AUPRC
            unique_vals = torch.unique(y_true)
            if unique_vals.numel() >= 2:
                # Update AUROC and AUPRC metrics
                metrics_container["auprc"][task_idx].update(
                    y_prob, y_true.int())
                metrics_container["auroc"][task_idx].update(
                    y_prob, y_true.int())

            # Always update classification metrics using task-specific threshold
            task_threshold = self.get_task_threshold(task_idx)
            y_pred = (y_prob >= task_threshold).int()
            metrics_container["acc"][task_idx].update(y_pred, y_true.int())
            metrics_container["precision"][task_idx].update(
                y_pred, y_true.int())
            metrics_container["recall"][task_idx].update(y_pred, y_true.int())

    @torch.no_grad()
    def _compute_split_metrics(self, split: str) -> Dict[str, float]:
        """Compute and return all metrics for the given split."""
        per_task_values: Dict[str, List[float]] = {
            "auprc": [],
            "auroc": [],
            "acc": [],
            "precision": [],
            "recall": []
        }
        per_task_metrics: Dict[str, float] = {}
        metrics_container = self.val_metrics if split == "val" else self.test_metrics

        for task_idx in range(self.num_tasks):
            # Get the actual task index for logging (1-based)
            actual_task_idx = self.task_indices[task_idx] + 1

            # Compute AUPRC
            try:
                auprc_val = metrics_container["auprc"][task_idx].compute(
                ).item()
                per_task_values["auprc"].append(auprc_val)
                per_task_metrics[f"auprc_task{actual_task_idx}"] = auprc_val
            except (RuntimeError, ValueError):
                per_task_metrics[f"auprc_task{actual_task_idx}"] = float("nan")

            # Compute AUROC
            try:
                auroc_val = metrics_container["auroc"][task_idx].compute(
                ).item()
                per_task_values["auroc"].append(auroc_val)
                per_task_metrics[f"auroc_task{actual_task_idx}"] = auroc_val
            except (RuntimeError, ValueError):
                per_task_metrics[f"auroc_task{actual_task_idx}"] = float("nan")

            # Compute other metrics
            try:
                acc_val = metrics_container["acc"][task_idx].compute().item()
                per_task_values["acc"].append(acc_val)
                per_task_metrics[f"acc_task{actual_task_idx}"] = acc_val
            except (RuntimeError, ValueError):
                per_task_metrics[f"acc_task{actual_task_idx}"] = float("nan")

            try:
                prec_val = metrics_container["precision"][task_idx].compute(
                ).item()
                per_task_values["precision"].append(prec_val)
                per_task_metrics[f"precision_task{actual_task_idx}"] = prec_val
            except (RuntimeError, ValueError):
                per_task_metrics[f"precision_task{actual_task_idx}"] = float(
                    "nan")

            try:
                rec_val = metrics_container["recall"][task_idx].compute().item(
                )
                per_task_values["recall"].append(rec_val)
                per_task_metrics[f"recall_task{actual_task_idx}"] = rec_val
            except (RuntimeError, ValueError):
                per_task_metrics[f"recall_task{actual_task_idx}"] = float(
                    "nan")

        # Compute macro averages
        macro: Dict[str, float] = {}
        for k, vals in per_task_values.items():
            if len(vals) > 0:
                macro[k] = float(sum(vals) / len(vals))
            else:
                macro[k] = float("nan")

        # Combine macro and per-task metrics
        macro.update(per_task_metrics)
        return macro

    def validation_step(self, batch, batch_idx):
        x, y, mask, _ = batch
        logits = self(x)
        loss = self._compute_masked_bce_loss(logits, y, mask)
        self.log(f"val/loss_fold{self.fold_index}",
                 loss,
                 prog_bar=False,
                 on_step=False,
                 on_epoch=True,
                 batch_size=x.size(0))
        # Update metrics (accumulate)
        self._update_split_metrics("val", logits, y, mask)

    def on_validation_epoch_end(self) -> None:
        """Compute and log validation metrics at end of epoch."""
        macro = self._compute_split_metrics("val")
        for k, v in macro.items():
            if v == v:  # not NaN
                self.log(f"val/{k}_fold{self.fold_index}",
                         v,
                         prog_bar=(k == "auprc"),
                         on_step=False,
                         on_epoch=True)

        # Log the average AUPRC across tasks for early stopping
        if "auprc" in macro and macro["auprc"] == macro["auprc"]:  # not NaN
            self.log(f"val/auprc_avg_fold{self.fold_index}",
                     macro["auprc"],
                     prog_bar=True,
                     on_step=False,
                     on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, mask, _ = batch
        logits = self(x)
        loss = self._compute_masked_bce_loss(logits, y, mask)
        self.log(f"test/loss_fold{self.fold_index}",
                 loss,
                 prog_bar=False,
                 on_step=False,
                 on_epoch=True,
                 batch_size=x.size(0))
        # Update metrics (accumulate)
        self._update_split_metrics("test", logits, y, mask)

    def on_test_epoch_end(self) -> None:
        """Compute and log test metrics at end of epoch."""
        macro = self._compute_split_metrics("test")
        for k, v in macro.items():
            if v == v:  # not NaN
                self.log(f"test/{k}_fold{self.fold_index}",
                         v,
                         prog_bar=False,
                         on_step=False,
                         on_epoch=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(),
                                      lr=self.hparams.learning_rate,
                                      weight_decay=self.hparams.L2_weight_norm)
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
            "monitor": f"val/auprc_fold{self.fold_index}",
        }


class Scorer(BaseScorer):
    """Collection of Minimol-based scoring functions for antibacterial activity."""

    _resources_lock = threading.Lock()
    _gram_negative_models_cache: Optional[List[MLPClassifier]] = None
    _gonorrhea_models_cache: Optional[List[MLPClassifier]] = None
    _featurizer_cache: Optional[Minimol] = None
    _models_dir_cache: Optional[Path] = None

    def __init__(self):
        # Call parent constructor to set up registry
        super().__init__()

        # Model paths (relative to module root)
        models_dir = Path(CURRENT_FILE_DIR) / "scorer_data" / "minimol_antibiotics"

        self._ensure_resources_initialized(models_dir)

        self._gram_negative_models = self.__class__._gram_negative_models_cache
        self._gonorrhea_models = self.__class__._gonorrhea_models_cache

    @classmethod
    def _ensure_resources_initialized(cls, models_dir: Path) -> None:
        """
        Lazily load heavyweight resources (models + featurizer) once per process.
        Multiple MCP requests can reach this code concurrently, so guard with a lock.
        """
        if (
            cls._gram_negative_models_cache is not None
            and cls._gonorrhea_models_cache is not None
            and cls._featurizer_cache is not None
        ):
            return

        with cls._resources_lock:
            if cls._gram_negative_models_cache is not None and cls._gonorrhea_models_cache is not None:
                # Featurizer might still be missing if initialization previously failed
                if cls._featurizer_cache is None:
                    cls._featurizer_cache = Minimol(batch_size=64)
                return

            cls._models_dir_cache = models_dir

            gram_paths = cls._collect_gram_negative_paths(models_dir)
            if not gram_paths:
                raise FileNotFoundError(
                    "No gram_negative model checkpoints found."
                )

            gonorrhea_paths = cls._collect_gonorrhea_paths(models_dir)
            if not gonorrhea_paths:
                raise FileNotFoundError(
                    "No gonorrhea model checkpoints found.")

            cls._gram_negative_models_cache = cls._load_models(gram_paths)
            cls._gonorrhea_models_cache = cls._load_models(gonorrhea_paths)
            cls._featurizer_cache = Minimol(batch_size=64)

    @staticmethod
    def _collect_gram_negative_paths(models_dir: Path) -> List[str]:
        """Collect all gram_negative model paths."""
        paths = []
        for i in range(9):  # fold_0 to fold_8
            model_path = models_dir / f"gram_negative_model_fold_{i}.pt"
            if model_path.exists():
                paths.append(str(model_path))
        return paths

    @staticmethod
    def _collect_gonorrhea_paths(models_dir: Path) -> List[str]:
        """Collect all gonorrhea model paths."""
        paths = []
        # Check for all gonorrhea models (fold_0 to fold_8)
        for i in range(9):
            model_path = models_dir / f"gonorrhea_model_fold_{i}.pt"
            if model_path.exists():
                paths.append(str(model_path))
        return paths

    @staticmethod
    def _load_models(model_paths: List[str]) -> List[MLPClassifier]:
        """Load all models from the given paths."""
        models = []
        for path in model_paths:
            model = MLPClassifier.load_from_checkpoint(path,
                                                       map_location='cpu')
            model.eval()
            models.append(model)
        return models

    @classmethod
    def _get_shared_featurizer(cls):
        """Retrieve a process-wide Minimol featurizer instance."""
        if cls._featurizer_cache is None:
            with cls._resources_lock:
                if cls._featurizer_cache is None:
                    cls._featurizer_cache = Minimol(batch_size=64)
        return cls._featurizer_cache

    def _featurize_with_skips(
            self, smiles_list: List[str]
    ) -> Tuple[torch.Tensor, List[str], List[int]]:
        """Featurize SMILES with error handling."""

        def _recursive_featurize(smiles_sublist, positions):
            try:
                featurizer = self._get_shared_featurizer()
                feats = featurizer(smiles_sublist)
                return feats, smiles_sublist, positions
            except Exception:
                if len(smiles_sublist) == 1:
                    print(f"[precompute] dropping '{smiles_sublist[0]}'")
                    return [], [], []
                mid = len(smiles_sublist) // 2
                lf, ls, lp = _recursive_featurize(smiles_sublist[:mid],
                                                  positions[:mid])
                rf, rs, rp = _recursive_featurize(smiles_sublist[mid:],
                                                  positions[mid:])
                return lf + rf, ls + rs, lp + rp

        positions = list(range(len(smiles_list)))
        feats, kept_smiles, kept_positions = _recursive_featurize(
            smiles_list, positions)

        # Stack the list of tensors into a single tensor
        if feats:
            features = torch.stack(feats)
            return features, kept_smiles, kept_positions
        else:
            return torch.empty(0), [], []

    def _predict_ensemble(self, features: torch.Tensor,
                          models: List[MLPClassifier]) -> torch.Tensor:
        """Make ensemble predictions using all models."""
        if len(models) == 0:
            raise ValueError("No models provided for prediction")

        predictions = []
        with torch.inference_mode():
            for model in models:
                # Move features to model device
                device = next(model.parameters()).device
                feats_device = features.to(device)

                logits = model(feats_device)
                probs = torch.sigmoid(logits)
                predictions.append(probs.cpu())

        # Average predictions across all models
        ensemble_preds = torch.stack(predictions).mean(dim=0)
        return ensemble_preds

    def _predict_on_candidates(
            self, samples: List[str],
            models: List[MLPClassifier]) -> List[Optional[float]]:
        """Predict on SMILES strings using ensemble of models."""
        valid_smiles = []
        valid_indices = []

        # Filter valid SMILES
        for i, smiles in enumerate(samples):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            valid_smiles.append(smiles)
            valid_indices.append(i)

        if not valid_smiles:
            return [None] * len(samples)

        # Featurize valid SMILES
        features, kept_smiles, kept_positions = self._featurize_with_skips(
            valid_smiles)
        if features.numel() == 0 or len(kept_smiles) == 0:
            return [None] * len(samples)

        # Make ensemble predictions
        ensemble_preds = self._predict_ensemble(features, models)

        # Map predictions back to original samples
        results: List[Optional[float]] = [None] * len(samples)
        for i, (kept_idx,
                pred) in enumerate(zip(kept_positions, ensemble_preds)):
            original_idx = valid_indices[kept_idx]
            results[original_idx] = float(
                pred[0])  # Take first task for single-task models

        return results

    def _predict_on_candidates_multitask(
            self, samples: List[str], models: List[MLPClassifier],
            task_idx: int) -> List[Optional[float]]:
        """Predict on SMILES strings using ensemble of multi-task models for specific task."""
        valid_smiles = []
        valid_indices = []

        # Filter valid SMILES
        for i, smiles in enumerate(samples):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            valid_smiles.append(smiles)
            valid_indices.append(i)

        if not valid_smiles:
            return [None] * len(samples)

        # Featurize valid SMILES
        features, kept_smiles, kept_positions = self._featurize_with_skips(
            valid_smiles)
        if features.numel() == 0 or len(kept_smiles) == 0:
            return [None] * len(samples)

        # Make ensemble predictions
        ensemble_preds = self._predict_ensemble(features, models)

        # Map predictions back to original samples
        results: List[Optional[float]] = [None] * len(samples)
        for i, (kept_idx,
                pred) in enumerate(zip(kept_positions, ensemble_preds)):
            original_idx = valid_indices[kept_idx]
            results[original_idx] = float(pred[task_idx])  # Take specific task

        return results

    @scorer(
        name="acinetobacter_baumanii_minimol",
        population_wise=False,
        description=(
            "Acinetobacter baumannii antibacterial activity score (value range: 0.0 to 1.0). "
            "This score represents the predicted probability of inhibitory activity against A. baumannii bacteria, "
            "as determined by a Minimol ensemble model trained on experimental antibacterial screening data. "
            "For high-precision predictions: scores ≥0.57 achieve 50% precision, ≥0.63 achieve 60% precision, and ≥0.99 achieve 70% precision. "
            "The F1-maximizing threshold is 0.53 for optimal precision-recall balance. "
            "A. baumannii is a critical priority pathogen due to its multidrug resistance and clinical importance in hospital-acquired infections."
        ))
    def score_acinetobacter_baumanii(
            self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate Acinetobacter baumannii antibacterial activity score.

        Uses Minimol ensemble model to predict probability of antibacterial activity.
        Higher scores indicate stronger predicted inhibitory activity against A. baumannii.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._predict_on_candidates_multitask(
            samples, self._gram_negative_models, task_idx=0)

    @scorer(
        name="escherichia_coli_minimol",
        population_wise=False,
        description=(
            "Escherichia coli antibacterial activity score (value range: 0.0 to 1.0). "
            "This score represents the predicted probability of inhibitory activity against E. coli bacteria, "
            "as determined by a Minimol ensemble model trained on experimental antibacterial screening data. "
            "For high-precision predictions: scores ≥0.12 achieve 50% precision, ≥0.35 achieve 60% precision, and ≥0.75 achieve 70% precision. "
            "The F1-maximizing threshold is 0.15 for optimal precision-recall balance. "
            "E. coli is a key model organism and clinically important pathogen for antibiotic discovery."
        ))
    def score_escherichia_coli(
            self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate Escherichia coli antibacterial activity score.

        Uses Minimol ensemble model to predict probability of antibacterial activity.
        Higher scores indicate stronger predicted inhibitory activity against E. coli.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._predict_on_candidates_multitask(
            samples, self._gram_negative_models, task_idx=1)

    @scorer(
        name="klebsiella_pneumoniae_minimol",
        population_wise=False,
        description=(
            "Klebsiella pneumoniae antibacterial activity score (value range: 0.0 to 1.0). "
            "This score represents the predicted probability of inhibitory activity against K. pneumoniae bacteria, "
            "as determined by a Minimol ensemble model trained on experimental antibacterial screening data. "
            "For high-precision predictions: scores ≥0.09 achieve 50% precision, ≥0.16 achieve 60% precision, and ≥0.37 achieve 70% precision. "
            "The F1-maximizing threshold is 0.13 for optimal precision-recall balance. "
            "K. pneumoniae is a critical priority pathogen due to its carbapenem resistance and clinical importance."
        ))
    def score_klebsiella_pneumoniae(
            self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate Klebsiella pneumoniae antibacterial activity score.

        Uses Minimol ensemble model to predict probability of antibacterial activity.
        Higher scores indicate stronger predicted inhibitory activity against K. pneumoniae.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._predict_on_candidates_multitask(
            samples, self._gram_negative_models, task_idx=2)

    @scorer(
        name="pseudomonas_aeruginosa_minimol",
        population_wise=False,
        description=(
            "Pseudomonas aeruginosa antibacterial activity score (value range: 0.0 to 1.0). "
            "This score represents the predicted probability of inhibitory activity against P. aeruginosa bacteria, "
            "as determined by a Minimol ensemble model trained on experimental antibacterial screening data. "
            "For high-precision predictions: scores ≥0.06 achieve 50% precision, ≥0.10 achieve 60% precision, and ≥0.18 achieve 70% precision. "
            "The F1-maximizing threshold is 0.12 for optimal precision-recall balance. "
            "P. aeruginosa is a critical priority pathogen due to its intrinsic resistance mechanisms and clinical importance in cystic fibrosis and hospital-acquired infections."
        ))
    def score_pseudomonas_aeruginosa(
            self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate Pseudomonas aeruginosa antibacterial activity score.

        Uses Minimol ensemble model to predict probability of antibacterial activity.
        Higher scores indicate stronger predicted inhibitory activity against P. aeruginosa.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._predict_on_candidates_multitask(
            samples, self._gram_negative_models, task_idx=3)

    @scorer(
        name="neisseria_gonorrhoeae_minimol",
        population_wise=False,
        description=(
            "Neisseria gonorrhoeae antibacterial activity score (value range: 0.0 to 1.0). "
            "This score represents the predicted probability of inhibitory activity against N. gonorrhoeae bacteria, "
            "as determined by a Minimol ensemble model trained on experimental antibacterial screening data. "
            "For high-precision predictions: scores ≥0.33 achieve 50% precision, ≥0.46 achieve 60% precision, and ≥0.65 achieve 70% precision. "
            "The F1-maximizing threshold is 0.34 for optimal precision-recall balance. "
            "N. gonorrhoeae is a high priority pathogen due to the emergence of multidrug-resistant strains and the urgent need for new therapeutic options."
        ))
    def score_neisseria_gonorrhoeae(
            self, samples: List[str]) -> List[Optional[float]]:
        """
        Calculate Neisseria gonorrhoeae antibacterial activity score.

        Uses Minimol ensemble model to predict probability of antibacterial activity.
        Higher scores indicate stronger predicted inhibitory activity against N. gonorrhoeae.

        Args:
            samples: List of input samples, where each sample is the SMILES string of a molecule

        Returns:
            List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations
        """
        return self._predict_on_candidates(samples, self._gonorrhea_models)
