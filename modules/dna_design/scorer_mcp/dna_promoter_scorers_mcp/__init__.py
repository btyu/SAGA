# --- auto-generated scorers start ---
scorers: dict = {
    'dna_k562_promoter_logfoldchange': {
        'function_name': 'score_k562_expression',
        'population_wise': False,
        'description': 'K562 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the K562 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in K562 cells. High values indicate sequences that are predicted to drive strong K562-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in K562, implying limited promoter potential or possible cell-type specificity elsewhere.',
        'tool_description': 'Predict the log-fold change measured from the DNA sequences for the K562 cell type.\n\nReturn the list of log-fold change levels as scores.\n\nArgs:\n    DNA sequences\nReturns:\n    List of scores (-inf,inf)',
    },
    'dna_hepg2_promoter_logfoldchange': {
        'function_name': 'score_hepg2_expression',
        'population_wise': False,
        'description': 'HepG2 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the HepG2 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in HepG2 cells. High values indicate sequences that are predicted to drive strong HepG2-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in HepG2, implying limited promoter potential or possible cell-type specificity elsewhere.',
        'tool_description': 'Predict the log-fold change measured from the DNA sequences for the HepG2 cell type.\n\nReturn the list of log-fold change levels as scores.\n\nArgs:\n    DNA sequences\nReturns:\n    List of scores (-inf,inf)',
    },
    'dna_gm12878_promoter_logfoldchange': {
        'function_name': 'score_gm12878_expression',
        'population_wise': False,
        'description': 'GM12878 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the GM12878 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in GM12878 cells. High values indicate sequences that are predicted to drive strong GM12878-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in GM12878, implying limited promoter potential or possible cell-type specificity elsewhere.',
        'tool_description': 'Predict the log-fold change measured from the DNA sequences for the GM12878 cell type.\n\nReturn the list of log-fold change levels as scores.\n\nArgs:\n    DNA sequences\nReturns:\n    List of scores (-inf,inf)',
    },
    'dna_sknsh_promoter_logfoldchange': {
        'function_name': 'score_sknsh_expression',
        'population_wise': False,
        'description': 'SKNSH promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the SKNSH cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in SKNSH cells. High values indicate sequences that are predicted to drive strong SKNSH-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in SKNSH, implying limited promoter potential or possible cell-type specificity elsewhere.',
        'tool_description': 'Predict the log-fold change measured from the DNA sequences for the SKNSH cell type.\n\nReturn the list of log-fold change levels as scores.\n\nArgs:\n    DNA sequences\nReturns:\n    List of scores (-inf,inf)',
    },
    'dna_a549_promoter_logfoldchange': {
        'function_name': 'score_a549_expression',
        'population_wise': False,
        'description': 'A549 promoter log-fold change (logFC) Massively Parallel Reporter Assay (MPRA) expression score (value range: -∞ to ∞). We fine-tuned Enformer for prediction. This objective measures the predicted transcriptional activity of DNA promoter sequences in the A549 cell line using the fine-tuned Enformer model. Log-fold change (logFC) quantifies the relative difference in expression levels, capturing how much more (or less) a sequence drives expression compared to baseline. This provides insight into whether a designed sequence acts as a strong or weak promoter in A549 cells. High values indicate sequences that are predicted to drive strong A549-specific expression, suggesting they function as effective promoters in this hematopoietic lineage. Low or negative values suggest weak or repressive activity in A549, implying limited promoter potential or possible cell-type specificity elsewhere.',
        'tool_description': 'Predict the log-fold change measured from the DNA sequences for the A549 cell type.\n\nReturn the list of log-fold change levels as scores.\n\nArgs:\n    DNA sequences\nReturns:\n    List of scores (-inf,inf)',
    },
}
# --- auto-generated scorers end ---
