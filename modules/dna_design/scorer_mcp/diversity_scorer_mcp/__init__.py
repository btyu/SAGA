# --- auto-generated scorers start ---
scorers: dict = {
    'dna_diversity': {
        'function_name': 'score_diversity_expression',
        'population_wise': True,
        'description': 'Sequence diversity score (value range: 0 to +∞). This objective quantifies the average Hamming distance among a group of DNA sequences, providing a measure of how different the sequences are from one another. Diversity is important in DNA design tasks because it reduces redundancy, increases exploration of sequence space, and can improve the chances of identifying functional variants. High values indicate that the sequences are highly diverse, suggesting broad coverage of sequence space and reduced redundancy. Low values suggest that the sequences are very similar or nearly identical, implying limited diversity and potentially reduced effectiveness in exploring novel regulatory patterns.',
        'tool_description': 'Score the diversity of input DNA sequences, which is the average Hamming distance among a group of DNA sequences.\n\nArgs:\n    samples: List of DNA sequences\nReturns:\n    A score [0, inf), which is the average Hamming distance among a group of DNA sequences.',
    },
}
# --- auto-generated scorers end ---
