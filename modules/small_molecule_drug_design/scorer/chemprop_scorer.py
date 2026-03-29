from typing import List, Optional, Tuple
from pathlib import Path

import argparse
import torch
from rdkit import Chem

import chemprop
from chemprop.data import StandardScaler, AtomBondScaler

from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate


@register_scorer_class
class ChempropScorers:
    """Collection of Chemprop-based scoring functions.

    This class preloads two Chemprop models at initialization:
    - Primary cell toxicity model
    - Staphylococcus aureus activity model
    """

    def __init__(self):
        # Allowlist classes for PyTorch 2.6 safe loading
        self._safe_globals = [
            argparse.Namespace, StandardScaler, AtomBondScaler
        ]
        torch.serialization.add_safe_globals(self._safe_globals)

        # Checkpoints (relative to module root)
        module_root = Path(__file__).resolve().parent.parent
        models_dir = module_root / "oracles" / "antibiotics" / "models"

        # Collect all available checkpoint model.pt paths (1..20)
        tox_root = models_dir / "primary_cell_toxicity_model"
        staph_root = models_dir / "staph_aureus_model"
        self._tox_checkpoint_paths = self._collect_checkpoint_paths(
            tox_root, in_train_subdir=True)
        self._staph_checkpoint_paths = self._collect_checkpoint_paths(
            staph_root, in_train_subdir=False)

        if not self._tox_checkpoint_paths:
            raise FileNotFoundError(
                "No primary_cell_toxicity_model checkpoints found.")
        if not self._staph_checkpoint_paths:
            raise FileNotFoundError("No staph_aureus_model checkpoints found.")

        # Preload a single ensemble per task (one load, internal averaging)
        self._tox_model_sets = [
            self._load_ensemble(self._tox_checkpoint_paths)
        ]
        self._staph_model_sets = [
            self._load_ensemble(self._staph_checkpoint_paths)
        ]

    @contextmanager
    def _torch_safe_globals(self):
        """Compatibility context manager for torch safe globals.

        Uses torch.serialization.safe_globals if available; otherwise falls back
        to add_safe_globals/remove_safe_globals if present; otherwise no-op.
        """
        safe_cm = getattr(torch.serialization, "safe_globals", None)
        if safe_cm is not None:
            with safe_cm(self._safe_globals):
                yield
            return
        add_fn = getattr(torch.serialization, "add_safe_globals", None)
        remove_fn = getattr(torch.serialization, "remove_safe_globals", None)
        if add_fn is not None and remove_fn is not None:
            add_fn(self._safe_globals)
            try:
                yield
            finally:
                try:
                    remove_fn(self._safe_globals)
                except Exception:
                    pass
        else:
            with nullcontext():
                yield

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
        with torch.serialization.safe_globals(self._safe_globals):
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
        with torch.serialization.safe_globals(self._safe_globals):
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
        with torch.serialization.safe_globals(self._safe_globals):
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
        with torch.serialization.safe_globals(self._safe_globals):
            model_objects = chemprop.train.load_model(args=args)
        return (args, model_objects)

    def _collect_checkpoint_paths(self, model_root: Path,
                                  in_train_subdir: bool) -> List[str]:
        paths: List[str] = []
        for i in range(1, 21):
            base = model_root / "train" if in_train_subdir else model_root
            ckpt = base / f"checkpoints{i}" / "fold_0" / "model_0" / "model.pt"
            if ckpt.exists():
                paths.append(str(ckpt))
        return paths

    def _load_model_sets(self, checkpoint_paths: List[str]) -> List[Tuple]:
        model_sets: List[Tuple] = []
        for p in checkpoint_paths:
            args = self._build_predict_args(p)
            with torch.serialization.safe_globals(self._safe_globals):
                model_objects = chemprop.train.load_model(args=args)
            model_sets.append((args, model_objects))
        return model_sets

    def _predict(self, smiles: List[str],
                 model_sets: List[Tuple]) -> List[List[float]]:
        rows = [[s] for s in smiles]
        aggregated: Optional[List[List[float]]] = None
        num_sets = 0
        with torch.inference_mode():
            for predict_args, model_objects in model_sets:
                preds = chemprop.train.make_predictions(
                    args=predict_args,
                    smiles=rows,
                    model_objects=model_objects)
            aggregated = [[float(x) for x in row] for row in preds]

            num_sets += 1
        if aggregated is None or num_sets == 0:
            return []
        for i in range(len(aggregated)):
            for j in range(len(aggregated[i])):
                aggregated[i][j] /= float(num_sets)
        return aggregated

    def _predict_on_candidates(
            self, candidates: List[Candidate],
            model_sets: List[Tuple]) -> List[float]:
        valid_smiles = []
        valid_indices = []

        for i, candidate in enumerate(candidates):
            s = candidate.representation  # type: ignore[attr-defined]
            if not isinstance(s, str) or not s or s.strip() == "":
                continue
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

        results: List[float] = [0.0] * len(candidates)
        for idx, score in zip(valid_indices, valid_scores):
            results[idx] = score

        return results

    @register_scorer(
        name="toxicity_safety_chemprop",
        population_wise=False,
        description=

        "Primary cell toxicity safety score (value range: 0.0 to 1.0). "
        "This score is computed as (1 - Primary cell toxicity probability) where the toxicity probability is predicted by a Chemprop ensemble model trained on primary cell toxicity data. "
        "The normalization inverts the toxicity prediction so higher scores indicate better safety profiles. "
        "High scores (>0.8) indicate excellent safety with low predicted toxicity to human primary cells, while low scores (<0.3) suggest high cytotoxicity that could lead to adverse effects in patients. "
        "This metric is crucial for drug safety assessment as primary cell toxicity often correlates with in vivo toxicity and can predict potential side effects in clinical development.",
    )
    def score_primary_cell_toxicity(
            self, candidates: List[Candidate]) -> List[float]:
        raw = self._predict_on_candidates(candidates, self._tox_model_sets)
        return [1.0 - float(s) for s in raw]

    @register_scorer(
        name="staph_aureus_chemprop",
        population_wise=False,
        description=
        "Staphylococcus aureus antibacterial activity score (value range: 0.0 to 1.0). "
        "This score represents the predicted probability of inhibitory activity against S. aureus bacteria, as determined by a Chemprop ensemble model trained on experimental antibacterial screening data. "
        "The score directly reflects the likelihood of achieving meaningful minimum inhibitory concentration (MIC) values against this important pathogen. "
        "High scores (>0.7) indicate strong predicted antibacterial activity suitable for antibiotic development, while low scores (<0.3) suggest minimal or no antibacterial effect. "
        "S. aureus is a critical target for antibiotic discovery due to its clinical importance and propensity for developing resistance, making this score valuable for prioritizing compounds with therapeutic potential.",
    )
    def score_staph_aureus(self,
                           candidates: List[Candidate]) -> List[float]:
        return self._predict_on_candidates(candidates, self._staph_model_sets)


if __name__ == "__main__":
    # Simple smoke test (optional manual run)
    from scileo_agent.core.registry import get_scorer, list_scorers

    print("Available scorers:\n", list_scorers())
    scorer1 = get_scorer("toxicity_safety_chemprop")
    scorer2 = get_scorer("staph_aureus_chemprop")
    test = [
        Candidate(
            representation=
            "CN1CCN(c2c(F)cc3c(c2F)-n2c(c(CC(=O)O)c4ccc(Cl)cc42)CC3)CC1"),
        Candidate(
            representation=
            "O=C(CCc1ccc(F)cc1Cl)NC1C(=O)N2C(C(=O)O)=C(C=CCN3CCOCC3)CSC12"),
        Candidate(
            representation=
            "CC1(C)CN(c2nc3c(cc2Cl)c(=O)c(C(=O)O)cn3-c2ccc(C(N)=O)cc2Cl)CCN1"),
        Candidate(
            representation=
            "O=C(CCc1ccc(Cl)cc1)NC1C(=O)N2C(C(=O)O)=C(C=CCNC(=O)c3cccnc3)CSC12"
        ),
        Candidate(
            representation=
            "O=C(Cc1ccc(F)cc1Cl)NC1C(=O)N2C(C(=O)O)=C(c3ccc(-c4ccncc4)cc3)CSC12"
        ),
    ]
    print(scorer1(test))
    print(scorer2(test))
