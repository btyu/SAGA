# SmallWorld similarity scorer MCP module

# --- auto-generated scorers start ---
scorers: dict = {
    "small_world_similarity": {
        "function_name": "score_small_world_similarity",
        "type": "candidate-wise",
        "description": "Similarity to Enamine REAL via SmallWorld API (0-1). Scores computed via RDKit Tanimoto similarity against closest SmallWorld hits.",
        "tool_description": "Compute SmallWorld similarity scores for the provided SMILES strings.\n\nArgs:\n    samples: List of SMILES strings to evaluate.\n\nReturns:\n    List of float scores in [0, 1]; scores can be None when SMILES parsing fails or inputs are invalid.",
    },
}
# --- auto-generated scorers end ---

