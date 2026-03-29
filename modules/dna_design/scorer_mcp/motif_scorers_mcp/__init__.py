# --- auto-generated scorers start ---
scorers: dict = {
    'dna_motif_num': {
        'function_name': 'score_motif_allspec',
        'population_wise': False,
        'description': 'Number of discovered DNA motifs (value range: 0 to +∞). We utilize a database known as JASPAR for matching, and this function contains motifs from all species. This objective quantifies how many transcription factor binding motifs are present in a DNA sequence. Motif counts reflect potential regulatory activity, since motifs are short sequence patterns recognized by transcription factors that control gene expression. A higher motif count suggests richer regulatory potential, while a lower count suggests fewer recognizable regulatory signals. High values indicate that the sequence contains many recognizable regulatory patterns, which may correspond to strong or complex transcriptional regulation. Low values (close to 0) suggest the sequence has few known motifs, implying limited transcription factor binding potential or non-regulatory sequence regions.',
        'tool_description': 'Score based on the enrichment of all motifs from the input DNA sequences.\n\nReturn a list of numbers representing the total number of discovered motifs for each DNA sequence/\n\nArgs:\n    DNA sequences\nReturn:\n    List of scores (0,+inf)',
    },
    'dna_motif_num_human': {
        'function_name': 'score_motif_human',
        'population_wise': False,
        'description': 'Number of discovered DNA motifs (value range: 0 to +∞). We utilize a database known as JASPAR for matching, and this function contains motifs from the human. This objective quantifies how many transcription factor binding motifs are present in a DNA sequence. Motif counts reflect potential regulatory activity, since motifs are short sequence patterns recognized by transcription factors that control gene expression. A higher motif count suggests richer regulatory potential, while a lower count suggests fewer recognizable regulatory signals. High values indicate that the sequence contains many recognizable regulatory patterns, which may correspond to strong or complex transcriptional regulation. Low values (close to 0) suggest the sequence has few known motifs, implying limited transcription factor binding potential or non-regulatory sequence regions.',
        'tool_description': 'Score based on the enrichment of human motifs from the input DNA sequences.\n\nReturn a list of numbers representing the total number of discovered motifs for each DNA sequence/\n\nArgs:\n    DNA sequences\nReturn:\n    List of scores (0,+inf)',
    },
}
# --- auto-generated scorers end ---
