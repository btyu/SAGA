# --- auto-generated scorers start ---
scorers: dict = {
    'dna_stability': {
        'function_name': 'score_stability_expression',
        'population_wise': False,
        'description': 'DNA stability score (value range: 0.0 to 1.0). This objective measures the GC content of a DNA sequence, calculated as the proportion of guanine (G) and cytosine (C) bases relative to total sequence length. GC content is a key determinant of DNA stability because G-C pairs form three hydrogen bonds, compared to two for A-T pairs, leading to stronger and more thermally stable double-stranded structures. High values (>0.5) indicate GC-rich sequences with greater thermodynamic stability, which may enhance structural rigidity but also increase synthesis difficulty. Low values (<0.5) indicate AT-rich sequences with lower stability, which may be easier to unwind for transcription but more prone to structural fragility.',
        'tool_description': 'Score the stability of input DNA sequences, which is the proportion of guanine (G) and cytosine (C) bases relative to total sequence length.\n\nArgs:\n    samples: List of DNA sequences, where each sequence is a string of DNA bases\n\nReturns:\n    List of float scores, each calculated for each sample; scores can be None for invalid samples or invalid computations',
    },
}
# --- auto-generated scorers end ---
