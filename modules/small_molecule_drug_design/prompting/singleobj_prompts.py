from typing import List

SINGLEOBJ_CROSSOVER_TEMPLATE = """
I have two molecules each represented by SMILES, and their associated {PROPERTY_NAME} value. {OBJECTIVE_DESCRIPTION}
({SMILE_A}, {SMILE_A_SCORE}) ({SMILE_B}, {SMILE_B_SCORE})

Please propose a new molecule that {OBJECTIVE}. You can either make crossover and mutations based on the given molecules or just propose a new molecule based on your knowledge.

CRITICAL: Your response must be ONLY a valid JSON object with exactly these two keys. No additional text, no markdown, no explanations outside the JSON.

Format:
{{"explanation": "your reasoning here", "molecule": "SMILES_string_here"}}

Requirements:
- Use double quotes for keys and string values
- Include exactly two keys: "explanation" and "molecule"
- No trailing commas
- No text before or after the JSON object
- "explanation" must be a single string describing your reasoning
- "molecule" must be a valid SMILES string
"""

SINGLEOBJ_MUTATION_TEMPLATE = """
I have a molecule represented by SMILES and its associated {PROPERTY_NAME}: ({SMILE}, {SCORE}). {OBJECTIVE_DESCRIPTION}

Please propose a mutated version of this molecule that {OBJECTIVE} by applying one or more of the following operations:
- Bond insertion or deletion
- Atom insertion or deletion
- Functional group insertion or deletion or substitution
- Bond order changes (single/double/triple)
- Atom identity changes (e.g., C→N, O→S)

CRITICAL: Your response must be ONLY a valid JSON object with exactly these two keys. No additional text, no markdown, no explanations outside the JSON.

Format:
{{"explanation": "your reasoning here", "molecule": "SMILES_string_here"}}

Requirements:
- Use double quotes for keys and string values
- Include exactly two keys: "explanation" and "molecule"
- No trailing commas
- No text before or after the JSON object
- "explanation" describes the specific mutations and why they improve {PROPERTY_NAME}
- "molecule" must be a valid SMILES string of the mutated molecule
- The molecule must be chemically valid and reasonably similar to the original (modest mutations; scaffold retained)
"""


def build_singleobj_crossover_prompt(parents, objective) -> str:
    return SINGLEOBJ_CROSSOVER_TEMPLATE.format(
        PROPERTY_NAME=objective.name,
        OBJECTIVE_DESCRIPTION=objective.description,
        OBJECTIVE=(f"has a higher {objective.name} score" if getattr(
            objective, "is_maximization", True) else
                   f"has a lower {objective.name} score"),
        SMILE_A=parents[0].representation,
        SMILE_A_SCORE=parents[0].scores.get(objective.name, "unknown"),
        SMILE_B=parents[1].representation,
        SMILE_B_SCORE=parents[1].scores.get(objective.name, "unknown"),
    )


def build_singleobj_mutation_prompt(candidate, objective) -> str:
    return SINGLEOBJ_MUTATION_TEMPLATE.format(
        PROPERTY_NAME=objective.name,
        OBJECTIVE_DESCRIPTION=objective.description,
        OBJECTIVE=(f"has a higher {objective.name} score" if getattr(
            objective, "is_maximization", True) else
                   f"has a lower {objective.name} score"),
        SMILE=candidate.representation,
        SCORE=candidate.scores.get(objective.name, "unknown"),
    )
