# --- auto-generated scorers start ---
scorers: dict = {
    'ra_score_xgb': {
        'function_name': 'score_ra_xgb',
        'population_wise': False,
        'description': 'RAscore XGB synthesizability score (value range: 0.0 to 1.0). RAscore (Retrosynthetic Accessibility score) predicts the synthesizability of molecules using machine learning trained on synthetic route data from chemical literature. The XGB model uses molecular fingerprints and chemical knowledge to estimate synthetic accessibility as a probability score. High scores (>0.7) indicate highly synthesizable molecules with well-established synthetic routes and readily available starting materials, while low scores (<0.3) suggest challenging synthesis requiring novel chemistry or exotic reagents. Unlike rule-based approaches, RAscore learns from actual synthetic procedures, making it particularly valuable for assessing real-world synthetic feasibility in drug discovery campaigns.',
        'tool_description': 'Compute the RAscore XGB synthesizability score for each sample.\n\nArgs:\n    samples: List of input samples, where each sample is a SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
}
# --- auto-generated scorers end ---