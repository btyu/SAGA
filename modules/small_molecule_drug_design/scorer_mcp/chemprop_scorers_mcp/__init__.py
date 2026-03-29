# --- auto-generated scorers start ---
scorers: dict = {
    'toxicity_safety_chemprop': {
        'function_name': 'score_primary_cell_toxicity',
        'type': 'candidate-wise',
        'description': 'Primary cell toxicity safety score (value range: 0.0 to 1.0). This score is computed as (1 - Primary cell toxicity probability) where the toxicity probability is predicted by a Chemprop ensemble model trained on primary cell toxicity data. The normalization inverts the toxicity prediction so higher scores indicate better safety profiles. High scores (>0.8) indicate excellent safety with low predicted toxicity to human primary cells, while low scores (<0.3) suggest high cytotoxicity that could lead to adverse effects in patients. This metric is crucial for drug safety assessment as primary cell toxicity often correlates with in vivo toxicity and can predict potential side effects in clinical development.',
        'tool_description': 'Compute the primary cell toxicity score for each sample, which is defined as 1 - primary cell toxicity probability.\n\nArgs:\n    samples: List of input samples, where each sample is a SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
}
# --- auto-generated scorers end ---
