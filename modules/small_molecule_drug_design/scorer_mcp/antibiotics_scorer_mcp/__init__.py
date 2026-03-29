# --- auto-generated scorers start ---
scorers: dict = {
    'antibiotics_novelty': {
        'function_name': 'score_antibiotics_novelty',
        'population_wise': False,
        'description': 'Antibiotics novelty score (value range: 0.0 to 1.0). This score is computed as (1 - maximum Tanimoto similarity) using Morgan fingerprints (radius=2, 2048 bits) against a reference set of existing marketed antibiotics. The normalization ensures that completely novel structures score 1.0 while identical matches to known antibiotics score 0.0. High scores (>0.8) indicate high structural novelty that may circumvent existing resistance mechanisms and provide new modes of action, while low scores (<0.4) suggest close similarity to known antibiotics that may face cross-resistance issues. Novel antibiotics are crucial for combating antimicrobial resistance, as structurally distinct compounds are more likely to retain activity against resistant bacterial strains and offer new therapeutic options.',
        'tool_description': 'Compute the antibiotics novelty score for each sample, which is defined as 1 - max Tanimoto similarity to any known antibiotic.\n\nArgs:\n    samples: List of input samples, where each sample is a SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
    'antibiotics_motifs_filter': {
        'function_name': 'score_antibiotics_motifs_filter',
        'population_wise': False,
        'type': 'filter',
        'description': 'Binary filter for known antibiotic structural motifs (value: 0.0 or 1.0). This scorer identifies molecules containing structural patterns commonly found in existing antibiotics, including sulfonamides, aminoglycosides, beta-lactams, tetracyclines, quinolones, and pyrimidine derivatives. It also flags molecules matching PAINS (Pan-Assay Interference Compounds) alerts. A score of 1.0 indicates the molecule does NOT contain any known antibiotic motifs or PAINS alerts, suggesting structural novelty and reduced risk of assay interference. A score of 0.0 indicates the molecule contains one or more known antibiotic motifs or PAINS alerts, which may indicate similarity to existing antibiotics or potential assay interference issues. This filter is useful for identifying structurally novel candidates that escape known antibiotic classes while avoiding problematic structural patterns.',
        'tool_description': 'Compute the antibiotics motifs filter for each sample, which returns 1.0 if the molecule does NOT contain any known antibiotic motifs or PAINS alerts, and 0.0 if it contains one or more.\n\nArgs:\n    samples: List of input samples, where each sample is a SMILES string of a molecule\n\nReturns:\n    List of float scores (0.0 or 1.0), each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
}
# --- auto-generated scorers end ---
