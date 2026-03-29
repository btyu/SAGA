# Barebone prompts with no requirements - just maximize reward

BAREBONE_CROSSOVER_TEMPLATE = """
I have two molecules (Parent A and Parent B) as SMILES strings.

Objectives to optimize:
{OBJECTIVES_SUMMARY}

Aggregation equation:
{AGGREGATION_EQUATION}

Parent A: {SMILE_A}
Scores: {SMILE_A_SCORES}
Aggregated score: {SMILE_A_AGG_SCORE}

Parent B: {SMILE_B}
Scores: {SMILE_B_SCORES}
Aggregated score: {SMILE_B_AGG_SCORE}

Task: Design a new molecule by combining features from both parents to maximize the aggregated score.

Output ONLY the SMILES string of the new molecule, nothing else.
"""

BAREBONE_MUTATION_TEMPLATE = """
I have a molecule represented by SMILES.

Objectives to optimize:
{OBJECTIVES_SUMMARY}

Aggregation equation:
{AGGREGATION_EQUATION}

Candidate: {SMILE}
Scores: {SMILE_SCORES}
Aggregated score: {SMILE_AGG_SCORE}

Task: Mutate this molecule to improve the aggregated score.

Output ONLY the SMILES string of the mutated molecule, nothing else.
"""


def build_barebone_crossover_prompt(
    candidate_a,
    candidate_b,
    objectives,
    aggregation_equation,
    agg_score_a,
    agg_score_b,
):
    """Build barebone crossover prompt with no requirements."""
    from modules.small_molecule_drug_design.prompting.multiobj_prompts import (
        _build_objectives_summary,
        _format_candidate_scores,
    )

    return BAREBONE_CROSSOVER_TEMPLATE.format(
        OBJECTIVES_SUMMARY=_build_objectives_summary(objectives),
        AGGREGATION_EQUATION=str(aggregation_equation),
        SMILE_A=candidate_a.representation,
        SMILE_A_SCORES=_format_candidate_scores(candidate_a, objectives),
        SMILE_A_AGG_SCORE=f"{agg_score_a:.6f}",
        SMILE_B=candidate_b.representation,
        SMILE_B_SCORES=_format_candidate_scores(candidate_b, objectives),
        SMILE_B_AGG_SCORE=f"{agg_score_b:.6f}",
    )


def build_barebone_mutation_prompt(
    candidate,
    objectives,
    aggregation_equation,
    agg_score,
):
    """Build barebone mutation prompt with no requirements."""
    from modules.small_molecule_drug_design.prompting.multiobj_prompts import (
        _build_objectives_summary,
        _format_candidate_scores,
    )

    return BAREBONE_MUTATION_TEMPLATE.format(
        OBJECTIVES_SUMMARY=_build_objectives_summary(objectives),
        AGGREGATION_EQUATION=str(aggregation_equation),
        SMILE=candidate.representation,
        SMILE_SCORES=_format_candidate_scores(candidate, objectives),
        SMILE_AGG_SCORE=f"{agg_score:.6f}",
    )
