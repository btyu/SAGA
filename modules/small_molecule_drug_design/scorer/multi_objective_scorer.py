"""Multi-objective scorer for antibiotic drug design."""
from typing import List

from scileo_agent.core.registry import get_scorer
from scileo_agent.core.data_models import Candidate


def score_antibiotics_multi_objective(smiles_list: List[str]) -> List[float]:
    """
    Score a list of SMILES by multiplying 5 antibiotic objectives together.

    Objectives:
    - klebsiella_pneumoniae_minimol: K. pneumoniae antibacterial activity
    - antibiotics_novelty: Novelty vs known antibiotics
    - toxicity_safety_chemprop: Safety (1 - toxicity)
    - antibiotics_motifs_filter: Binary filter for known antibiotic motifs
    - arthor_similarity: Similarity to Enamine REAL compounds

    Args:
        smiles_list: List of SMILES strings to score

    Returns:
        List of scores (product of all objectives). Returns 0.0 for invalid
        molecules or if any objective returns None.
    """
    # Load all required scorers
    kp_scorer = get_scorer("klebsiella_pneumoniae_minimol")
    novelty_scorer = get_scorer("antibiotics_novelty")
    toxicity_scorer = get_scorer("toxicity_safety_chemprop")
    motifs_scorer = get_scorer("antibiotics_motifs_filter")
    similarity_scorer = get_scorer("arthor_similarity")

    # Convert SMILES to Candidate objects
    candidates = [Candidate(representation=smiles) for smiles in smiles_list]

    # Score each objective
    kp_scores = kp_scorer(candidates)
    novelty_scores = novelty_scorer(candidates)
    toxicity_scores = toxicity_scorer(candidates)
    motifs_scores = motifs_scorer(candidates)
    similarity_scores = similarity_scorer(candidates)

    # Multiply all scores together
    results = []
    for i in range(len(smiles_list)):
        # Get each score, treating None as 0.0
        kp = kp_scores[i] if kp_scores[i] is not None else 0.0
        novelty = novelty_scores[i] if novelty_scores[i] is not None else 0.0
        toxicity = toxicity_scores[i] if toxicity_scores[i] is not None else 0.0
        motifs = motifs_scores[i] if motifs_scores[i] is not None else 0.0
        similarity = similarity_scores[i] if similarity_scores[i] is not None else 0.0

        # Multiply all objectives
        product = kp * novelty * toxicity * motifs * similarity
        results.append(product)

    return results


if __name__ == "__main__":
    # Example usage
    test_smiles = [
        "CC(C)C(=O)O",  # Simple molecule
        "CN1CCN(c2c(F)cc3c(c2F)-n2c(c(CC(=O)O)c4ccc(Cl)cc42)CC3)CC1",  # Complex antibiotic
        "INVALID_SMILES",  # Invalid SMILES
    ]

    print("Testing multi-objective scorer:")
    scores = score_antibiotics_multi_objective(test_smiles)
    for smiles, score in zip(test_smiles, scores):
        print(f"  {smiles[:50]:50s} -> {score:.6f}")





