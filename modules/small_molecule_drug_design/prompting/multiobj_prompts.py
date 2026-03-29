from typing import List, Optional

MULTIOBJ_CROSSOVER_TEMPLATE = """
I have two molecules (Parent A and Parent B), each represented by SMILES.
We are optimizing a multi-objective with the following objectives:

{OBJECTIVES_SUMMARY}

Aggregation equation:
{AGGREGATION_EQUATION}

Parent A: {SMILE_A}
Scores per objective:
{SMILE_A_SCORES}
Aggregated weighted score: {SMILE_A_AGG_SCORE}

Parent B: {SMILE_B}
Scores per objective:
{SMILE_B_SCORES}
Aggregated weighted score: {SMILE_B_AGG_SCORE}

Role: You are a medicinal chemistry expert. Propose a new molecule that improves the multi-objective score given the aggregation equation and objective directions. You may use crossover and/or small mutations, or propose a new molecule.

Guidance:
- Prioritize objectives explicitly based on (a) weights and (b) current per-objective scores of the parents (identify which properties most limit the weighted score).
- Describe concrete chemistry operations (ring substitutions, heteroatom swaps, bioisosteres, linker edits, conformational constraints, charge tuning, etc.).
- Ensure chemical validity: correct valence, no highly reactive/unstable motifs without justification, reasonable heteroatom placement, plausible tautomer/protonation at physiological pH.
- Medicinal chemistry checks: synthetic accessibility (brief rationale), avoid obvious PAINS/reactive warheads unless justified, consider permeability/solubility trade-offs, control lipophilicity.
- Diversity: propose exactly 5 distinct candidate SMILES, explain their intended improvements, then critically evaluate and discard weaker ones. Select only ONE final design to output as the best balance for the given objectives.
- Stereochemistry: include stereochemical specification in SMILES when introducing or affecting chiral centers.

CRITICAL: Respond with ONLY a YAML mapping containing exactly two keys and no extra text.

Format (YAML):
explanation: |
  Step 1 (Objective Prioritization):
    - State which objectives to focus on and why (weights vs current scores)
  Step 2 (Brainstorm Candidates):
    - Propose 5 distinct candidate SMILES
    - For each, first describe the structural edit and expected effect on each objective, then output the SMILES.
  Step 3 (Safety & Self-Critique Selection):
    - For each of the 5 candidates, assess:
        * Alignment with objectives
        * Safety/feasibility (validity, synthetic tractability, ADMET, PAINS/reactivity, permeability/solubility)
    - Explicitly discard weaker ones (explain why they fail trade-offs or safety checks)
    - Justify why the chosen candidate best balances the objectives and passes feasibility checks
molecule: SMILES_string_here
"""

MULTIOBJ_3D_POSE_CROSSOVER_TEMPLATE = """
I have two molecules (Parent A and Parent B), each represented by SMILES.
We are optimizing a multi-objective with the following objectives:

{OBJECTIVES_SUMMARY}

Aggregation equation:
{AGGREGATION_EQUATION}

Parent A: {SMILE_A}
Scores per objective:
{SMILE_A_SCORES}
Aggregated weighted score: {SMILE_A_AGG_SCORE}
3D docked pose pocket residue map: {SMILE_A_RESIDUE_MAP}

Parent B: {SMILE_B}
Scores per objective:
{SMILE_B_SCORES}
Aggregated weighted score: {SMILE_B_AGG_SCORE}
3D docked pose pocket residue map: {SMILE_B_RESIDUE_MAP}

Role: You are a medicinal chemistry expert. Propose a new molecule that improves the multi-objective score given the aggregation equation and objective directions. You may use crossover and/or small mutations, or propose a new molecule.

Guidance:
- Prioritize objectives explicitly based on (a) weights and (b) current per-objective scores of the parents (identify which properties most limit the weighted score).
- Describe concrete chemistry operations (ring substitutions, heteroatom swaps, bioisosteres, linker edits, conformational constraints, charge tuning, etc.).
- Ensure chemical validity: correct valence, no highly reactive/unstable motifs without justification, reasonable heteroatom placement, plausible tautomer/protonation at physiological pH.
- Medicinal chemistry checks: synthetic accessibility (brief rationale), avoid obvious PAINS/reactive warheads unless justified, consider permeability/solubility trade-offs, control lipophilicity.
- Diversity: propose exactly 5 distinct candidate SMILES, explain their intended improvements, then critically evaluate and discard weaker ones. Select only ONE final design to output as the best balance for the given objectives.
- Stereochemistry: include stereochemical specification in SMILES when introducing or affecting chiral centers.

CRITICAL: Respond with ONLY a YAML mapping containing exactly two keys and no extra text.

Format (YAML):
explanation: |
  Step 1 (Objective Prioritization):
    - State which objectives to focus on and why (weights vs current scores)
  Step 2 (Brainstorm Candidates):
    - Propose 5 distinct candidate SMILES
    - For each, describe the structural edit and expected effect on each objective
  Step 3 (Safety & Self-Critique Selection):
    - For each of the 5 candidates, assess:
        * Alignment with objectives
        * Safety/feasibility (validity, synthetic tractability, ADMET, PAINS/reactivity, permeability/solubility)
    - Explicitly discard weaker ones (explain why they fail trade-offs or safety checks)
    - Justify why the chosen candidate best balances the objectives and passes feasibility checks
molecule: SMILES_string_here
"""

MULTIOBJ_MUTATION_TEMPLATE = """
I have a molecule represented by SMILES.
We are optimizing a multi-objective with the following objectives:

{OBJECTIVES_SUMMARY}

Aggregation equation:
{AGGREGATION_EQUATION}

Candidate: {SMILE}
Scores per objective:
{SMILE_SCORES}
Aggregated weighted score: {SMILE_AGG_SCORE}

Role: You are a medicinal chemistry expert. Propose a mutated version of this molecule that improves the multi-objective score.

Allowed operations (choose those that best fit your plan):
- Bond insertion or deletion
- Atom insertion or deletion
- Functional group insertion or deletion or substitution
- Bond order changes (single/double/triple)
- Atom identity changes (e.g., C→N, O→S)

Guidance:
- Retain the core scaffold unless a clear, justified benefit outweighs similarity loss; prefer modest, property-targeted edits.
- Explicitly prioritize objectives based on weights and the candidate’s current scores (what property is currently limiting?).
- Describe precise atomic/functional group edits and how they improve prioritized properties.
- Maintain chemical validity and plausible protonation; specify stereochemistry if new stereocenters are created.
- Similarity target: aim for a moderate change (conceptually ~0.4–0.8 Tanimoto) unless strong justification is provided.
- Diversity: list 1–2 alternative mutation strategies in the explanation even if you output only one molecule.

CRITICAL: Respond with ONLY a YAML mapping containing exactly two keys and no extra text.

Format (YAML):
explanation: |
  Priority:
    - Which objectives to improve first and why
  Planned_changes:
    - Exact mutations (atom/bond/group) and expected property impact
  Alternatives:
    - 1–2 different mutation strategies to explore diversity
  Safety_checks:
    - Validity/synthetic feasibility/ADMET risk notes and mitigations
molecule: SMILES_string_here

Requirements:
- Include exactly two keys: explanation and molecule
- No text before or after the YAML mapping
- The explanation may be multi-line; prefer a YAML block scalar as shown
- The molecule must be a valid single-line SMILES string
- The molecule value must be PLAIN SMILES: no quotes, no spaces, no CXSMILES annotations (no '|' characters), no comments
- The molecule must appear on the same line as the 'molecule:' key with no trailing content
- The molecule must be chemically valid and reasonably similar to the original (modest mutations; scaffold retained) unless explicitly justified
- Do not include any markdown formatting or additional text
"""

MULTIOBJECTIVES_WEIGHTAGE_TEMPLATE = """
I have a set of {NUM_OBJECTIVES} objectives in the format of '<OBJECTIVE_[INDEX_NUMBER]> - <OBJECTIVE> - Description: <OBJECTIVE_DESCRIPTION> \n' :

{OBJECTIVES_INFO}

Based on your expert knowledge of computational chemistry and drug discovery, propose a set of weights for these objectives,
with each weight ranging from 0.0 to 1.0, where 0.0 indicates no weightage (ie. lowest importance) and 1.0 indicates full weightage (ie. highest importance).
The weights do not have to sum up to 1.0.

CRITICAL: Respond with ONLY a YAML mapping containing exactly two keys and no extra text.
You MUST show your work for each gate check - no shortcuts or summary statements allowed.
If you skip any required steps or provide incomplete gate checking, your response will be rejected.

Format (YAML):
explanation: |
  Rationale:
    - Describe the trade-offs and why certain properties should be prioritized
  Risk_management:
    - Note any constraints (e.g., ADMET, SA) that influence weighting
weights: [<weight_1>, <weight_2>, ..., <weight_N>]

Requirements:
- There should be the same number of weights as the number of objectives, ie. one weight per objective
- The order of weights should correspond to the order of objectives ie. [<weight_1>, <weight_2>] should correspond to <OBJECTIVE_1> and <OBJECTIVE_2>
- If there is only one objective, the weight should be 1.0
"""


def build_multiobjective_weightage_prompt(objectives) -> str:
    objectives_info = "\n".join([
        f"<OBJECTIVE_{idx + 1}> - {getattr(o, 'name', 'objective')} - Description: {getattr(o, 'description', '')}"
        for idx, o in enumerate(objectives)
    ])
    return MULTIOBJECTIVES_WEIGHTAGE_TEMPLATE.format(
        NUM_OBJECTIVES=len(objectives), OBJECTIVES_INFO=objectives_info)


def _build_objectives_summary(objectives) -> str:
    lines = []
    for obj in objectives:
        is_max = getattr(obj, "is_maximization", None)
        if is_max is None:
            is_max = getattr(obj, "optimization_direction",
                             "maximize") == "maximize"
        direction = "maximize" if is_max else "minimize"
        description = getattr(obj, "description", "") or ""
        name = getattr(obj, "name", None) or "objective"
        lines.append(f"- {name} ({direction}): {description}")
    return "\n".join(lines)


def _format_candidate_scores(candidate, objectives) -> str:
    lines = []
    for obj in objectives:
        name = getattr(obj, "name", "objective")
        val = getattr(candidate, "scores", {}).get(name, "unknown")
        lines.append(f"  - {name}: {val}")
    return "\n".join(lines)


def build_multiobj_crossover_prompt(parents, objectives,
                                    aggregation_equation: str,
                                    agg_score_a: float,
                                    agg_score_b: float,
                                    add_3d_docked_pose_info: bool = False,
                                    residue_map_a: Optional[str] = None,
                                    residue_map_b: Optional[str] = None,
                                    ) -> str:
    # Ensure objectives align with weights length
    objs = objectives

    if add_3d_docked_pose_info:
      prompt = MULTIOBJ_3D_POSE_CROSSOVER_TEMPLATE.format(
        OBJECTIVES_SUMMARY=_build_objectives_summary(objs),
        AGGREGATION_EQUATION=str(aggregation_equation),
        SMILE_A=parents[0].representation,
        SMILE_A_SCORES=_format_candidate_scores(parents[0], objs),
        SMILE_A_AGG_SCORE=f"{agg_score_a:.6f}",
        SMILE_B=parents[1].representation,
        SMILE_B_SCORES=_format_candidate_scores(parents[1], objs),
        SMILE_B_AGG_SCORE=f"{agg_score_b:.6f}",
        SMILE_A_RESIDUE_MAP=residue_map_a,
        SMILE_B_RESIDUE_MAP=residue_map_b,
      )
    else:  
      prompt = MULTIOBJ_CROSSOVER_TEMPLATE.format(
        OBJECTIVES_SUMMARY=_build_objectives_summary(objs),
        AGGREGATION_EQUATION=str(aggregation_equation),
        SMILE_A=parents[0].representation,
        SMILE_A_SCORES=_format_candidate_scores(parents[0], objs),
        SMILE_A_AGG_SCORE=f"{agg_score_a:.6f}",
        SMILE_B=parents[1].representation,
        SMILE_B_SCORES=_format_candidate_scores(parents[1], objs),
        SMILE_B_AGG_SCORE=f"{agg_score_b:.6f}",
      )

    return prompt


def build_multiobj_mutation_prompt(candidate, objectives,
                                   aggregation_equation: str,
                                   agg_score: float) -> str:
    objs = objectives
    return MULTIOBJ_MUTATION_TEMPLATE.format(
        OBJECTIVES_SUMMARY=_build_objectives_summary(objs),
        AGGREGATION_EQUATION=str(aggregation_equation),
        SMILE=candidate.representation,
        SMILE_SCORES=_format_candidate_scores(candidate, objs),
        SMILE_AGG_SCORE=f"{agg_score:.6f}",
    )
