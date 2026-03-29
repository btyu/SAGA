from typing import List, Optional
from pathlib import Path

from rdkit import Chem

from scileo_agent.core.registry import register_scorer_class, register_scorer
from scileo_agent.core.data_models import Candidate

try:
    from RAscore import RAscore_XGB  # type: ignore
    _RA_IMPORT_ERROR: Optional[Exception] = None
except Exception as e:  # pragma: no cover - optional dependency
    RAscore_XGB = None
    _RA_IMPORT_ERROR = e


@register_scorer_class
class RAScorers:
    """Collection of RA-based scoring functions.

    Loads the XGB model from the local RA models directory prepared per README.
    """

    def __init__(self):
        if RAscore_XGB is None:
            raise ImportError(
                f"RAscore package not available. Ensure it is installed as per README. Error: {_RA_IMPORT_ERROR}"
            )

        # Resolve model path relative to module root
        module_root = Path(__file__).resolve().parent.parent
        models_dir = module_root / "oracles" / "ra_score" / "models"
        self._xgb_model_path = str(models_dir / "XGB_chembl_ecfp_counts" /
                                   "model.pkl")

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

    @register_scorer(
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
                     candidates: List[Candidate]) -> List[float]:
        valid_smiles: List[str] = []
        valid_indices: List[int] = []

        for i, candidate in enumerate(candidates):
            s = candidate.representation  # type: ignore[attr-defined]
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

        results: List[float] = [0.0] * len(candidates)
        for idx, score in zip(valid_indices, valid_scores):
            results[idx] = score if score is not None else 0.0

        return results


if __name__ == "__main__":
    from scileo_agent.core.registry import get_scorer, list_scorers

    print("Available scorers:\n", list_scorers())
    scorer = get_scorer("ra_score_xgb")
    test = [
        Candidate(representation="CC(=O)NC1=CC=C(O)C=C1"),
        Candidate(representation="COC(=O)C12C3C4C1C1C2C2C3C4C12C(=O)OC")
    ]
    print(scorer(test))
