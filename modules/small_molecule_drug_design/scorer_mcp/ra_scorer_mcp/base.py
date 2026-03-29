import os
from loguru import logger
from typing import List, Optional

from rdkit import Chem

from .scorer_utils import BaseScorer, scorer

from RAscore import RAscore_XGB  # type: ignore

# ==================================================================================================== #
#                                             NOTE
# This file will be executed in docker, and should be self-contained
# So, you should NOT import any uninstalled modules other than the ones in this scorer directory
# For example, you CANNOT import `scileo_agent` and other modules outside this scorer directory
# ==================================================================================================== #

CURRENT_FILE_DIR = os.path.dirname(os.path.abspath(__file__))


class Scorer(BaseScorer):
    """Collection of RA-based scoring functions.

    Loads the XGB model from the local RA models directory prepared per README.
    """

    # TODO: This is not complete yet: Lack the oracle checkpoints

    def __init__(self):
        # Call parent constructor to set up registry
        super().__init__()

        self._xgb_model_path = os.path.join(CURRENT_FILE_DIR, "scorer_data", "model.pkl")

        # Initialize scorer
        self._xgb_scorer = RAscore_XGB.RAScorerXGB(
            model_path=self._xgb_model_path)

    def _predict_valid_smiles(self,
                              smiles: List[str]) -> List[Optional[float]]:
        results: List[Optional[float]] = []
        for s in smiles:
            try:
                score = float(self._xgb_scorer.predict(s))
            except Exception:
                score = None
            results.append(score)
        return results

    @scorer(
        name="ra_score_xgb",
        population_wise=False,
        description=
        "RAscore XGB synthesizability score (value range: 0.0 to 1.0). "
        "RAscore (Retrosynthetic Accessibility score) predicts the synthesizability of molecules using machine learning trained on synthetic route data from chemical literature. "
        "The XGB model uses molecular fingerprints and chemical knowledge to estimate synthetic accessibility as a probability score. "
        "High scores (>0.7) indicate highly synthesizable molecules with well-established synthetic routes and readily available starting materials, while low scores (<0.3) suggest challenging synthesis requiring novel chemistry or exotic reagents. "
        "Unlike rule-based approaches, RAscore learns from actual synthetic procedures, making it particularly valuable for assessing real-world synthetic feasibility in drug discovery campaigns.",
    )
    def score_ra_xgb(self,
                     samples: List[str]) -> List[float]:
        valid_smiles: List[str] = []
        valid_indices: List[int] = []

        for i, sample in enumerate(samples):
            s = sample
            if not isinstance(s, str) or not s or s.strip() == "":
                continue
            mol = Chem.MolFromSmiles(s)
            if mol is None:
                continue
            valid_smiles.append(s)
            valid_indices.append(i)

        if valid_smiles:
            valid_scores = self._predict_valid_smiles(valid_smiles)
        else:
            valid_scores = []

        results: List[float] = [0.0] * len(samples)
        for idx, score in zip(valid_indices, valid_scores):
            results[idx] = score if score is not None else 0.0

        return results
