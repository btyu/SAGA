# --- auto-generated scorers start ---
scorers: dict = {
    'acinetobacter_baumanii_minimol': {
        'function_name': 'score_acinetobacter_baumanii',
        'population_wise': False,
        'description': 'Acinetobacter baumannii antibacterial activity score (value range: 0.0 to 1.0). This score represents the predicted probability of inhibitory activity against A. baumannii bacteria, as determined by a Minimol ensemble model trained on experimental antibacterial screening data. For high-precision predictions: scores ≥0.57 achieve 50% precision, ≥0.63 achieve 60% precision, and ≥0.99 achieve 70% precision. The F1-maximizing threshold is 0.53 for optimal precision-recall balance. A. baumannii is a critical priority pathogen due to its multidrug resistance and clinical importance in hospital-acquired infections.',
        'tool_description': 'Calculate Acinetobacter baumannii antibacterial activity score.\n\nUses Minimol ensemble model to predict probability of antibacterial activity.\nHigher scores indicate stronger predicted inhibitory activity against A. baumannii.\n\nArgs:\n    samples: List of input samples, where each sample is the SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
    'escherichia_coli_minimol': {
        'function_name': 'score_escherichia_coli',
        'population_wise': False,
        'description': 'Escherichia coli antibacterial activity score (value range: 0.0 to 1.0). This score represents the predicted probability of inhibitory activity against E. coli bacteria, as determined by a Minimol ensemble model trained on experimental antibacterial screening data. For high-precision predictions: scores ≥0.12 achieve 50% precision, ≥0.35 achieve 60% precision, and ≥0.75 achieve 70% precision. The F1-maximizing threshold is 0.15 for optimal precision-recall balance. E. coli is a key model organism and clinically important pathogen for antibiotic discovery.',
        'tool_description': 'Calculate Escherichia coli antibacterial activity score.\n\nUses Minimol ensemble model to predict probability of antibacterial activity.\nHigher scores indicate stronger predicted inhibitory activity against E. coli.\n\nArgs:\n    samples: List of input samples, where each sample is the SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
    'klebsiella_pneumoniae_minimol': {
        'function_name': 'score_klebsiella_pneumoniae',
        'population_wise': False,
        'description': 'Klebsiella pneumoniae antibacterial activity score (value range: 0.0 to 1.0). This score represents the predicted probability of inhibitory activity against K. pneumoniae bacteria, as determined by a Minimol ensemble model trained on experimental antibacterial screening data. For high-precision predictions: scores ≥0.09 achieve 50% precision, ≥0.16 achieve 60% precision, and ≥0.37 achieve 70% precision. The F1-maximizing threshold is 0.13 for optimal precision-recall balance. K. pneumoniae is a critical priority pathogen due to its carbapenem resistance and clinical importance.',
        'tool_description': 'Calculate Klebsiella pneumoniae antibacterial activity score.\n\nUses Minimol ensemble model to predict probability of antibacterial activity.\nHigher scores indicate stronger predicted inhibitory activity against K. pneumoniae.\n\nArgs:\n    samples: List of input samples, where each sample is the SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
    'pseudomonas_aeruginosa_minimol': {
        'function_name': 'score_pseudomonas_aeruginosa',
        'population_wise': False,
        'description': 'Pseudomonas aeruginosa antibacterial activity score (value range: 0.0 to 1.0). This score represents the predicted probability of inhibitory activity against P. aeruginosa bacteria, as determined by a Minimol ensemble model trained on experimental antibacterial screening data. For high-precision predictions: scores ≥0.06 achieve 50% precision, ≥0.10 achieve 60% precision, and ≥0.18 achieve 70% precision. The F1-maximizing threshold is 0.12 for optimal precision-recall balance. P. aeruginosa is a critical priority pathogen due to its intrinsic resistance mechanisms and clinical importance in cystic fibrosis and hospital-acquired infections.',
        'tool_description': 'Calculate Pseudomonas aeruginosa antibacterial activity score.\n\nUses Minimol ensemble model to predict probability of antibacterial activity.\nHigher scores indicate stronger predicted inhibitory activity against P. aeruginosa.\n\nArgs:\n    samples: List of input samples, where each sample is the SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
    'neisseria_gonorrhoeae_minimol': {
        'function_name': 'score_neisseria_gonorrhoeae',
        'population_wise': False,
        'description': 'Neisseria gonorrhoeae antibacterial activity score (value range: 0.0 to 1.0). This score represents the predicted probability of inhibitory activity against N. gonorrhoeae bacteria, as determined by a Minimol ensemble model trained on experimental antibacterial screening data. For high-precision predictions: scores ≥0.33 achieve 50% precision, ≥0.46 achieve 60% precision, and ≥0.65 achieve 70% precision. The F1-maximizing threshold is 0.34 for optimal precision-recall balance. N. gonorrhoeae is a high priority pathogen due to the emergence of multidrug-resistant strains and the urgent need for new therapeutic options.',
        'tool_description': 'Calculate Neisseria gonorrhoeae antibacterial activity score.\n\nUses Minimol ensemble model to predict probability of antibacterial activity.\nHigher scores indicate stronger predicted inhibitory activity against N. gonorrhoeae.\n\nArgs:\n    samples: List of input samples, where each sample is the SMILES string of a molecule\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
}
# --- auto-generated scorers end ---
