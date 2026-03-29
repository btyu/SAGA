# --- auto-generated scorers start ---
scorers: dict = {
    'arthor_similarity': {
        'function_name': 'score_arthor_similarity',
        'population_wise': False,
        'description': 'Similarity to Enamine REAL via Arthor API (0-1). Calculates RDKit Tanimoto vs the closest REAL hit; scores below 0.5 similarity map to 0, 1.0 similarity maps to 1.0.',
        'tool_description': 'Compute Arthor similarity scores for the provided SMILES strings.\n\nArgs:\n    samples: List of SMILES strings to evaluate\n\nReturns:\n    List of float scores in [0, 1]; scores can be None when SMILES parsing fails',
    },
}
# --- auto-generated scorers end ---





