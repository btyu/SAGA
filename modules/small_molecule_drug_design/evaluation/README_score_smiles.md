# Score SMILES Script

## Overview

The `score_smiles.py` script allows you to score a list of SMILES strings with multiple objectives and output the results to a CSV file. It also includes filtering information (PAINS, motif filters) to indicate which molecules would be filtered out.

## Features

- Score SMILES with any available objective from the SciLeoAgent framework
- Support for both command-line SMILES input and file input (text or CSV)
- Automatic filtering analysis using `filter_smiles_preserves_existing_hits()`
- Comprehensive output with:
  - All objective scores
  - `motif_filtered` column: boolean indicating if molecule would be filtered
  - `filter_reason` column: detailed reason for filtering (PAINS, motif matches)
- Summary statistics for all objectives

## Installation

This script requires the same environment as the SciLeoAgent framework. Ensure you have:

1. Activated the `genesis` conda environment
2. All dependencies installed (see main repository README)

## Usage

### Basic Usage

```bash
# Score SMILES from command line
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles "CCO" "c1ccccc1" "CC(C)O" \
    --objectives qed logp mw \
    --output results.csv
```

### Score from File

```bash
# From text file (one SMILES per line)
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles-file molecules.txt \
    --objectives qed logp mw \
    --output results.csv

# From CSV file (looks for 'smiles' column)
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles-file molecules.csv \
    --objectives qed logp mw \
    --output results.csv
```

### Antibiotics Optimization Example

```bash
# Score with Klebsiella pneumoniae and related objectives
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles-file candidates.csv \
    --objectives klebsiella_pneumoniae toxicity pains brenk antibiotics_novelty deepdl mw \
    --output klebsiella_scored.csv

# Score with E. coli
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles-file candidates.txt \
    --objectives escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
    --output escherichia_coli_scored.csv
```

### Skip Filtering Information

If you don't need the filtering columns:

```bash
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles "CCO" "c1ccccc1" \
    --objectives qed logp \
    --output results.csv \
    --no-filter-info
```

## Available Objectives

The script supports all objectives available in `run_optimization.py`:

### Protein-Ligand Docking
- `ampcclean` - AMPcclean UniDock docking
- `muopioidclean` - MuOpioid UniDock docking
- `mpro` - Main protease (SARS-CoV-2) UniDock docking
- `mars1` - MARS1 UniDock docking

### Molecular Properties
- `qed` - Quantitative Estimate of Drug-likeness
- `logp` - Lipophilicity score (LogP)
- `mw` - Molecular weight score

### Drug-likeness and Toxicity
- `deepdl` - DeepDL druglikeness score
- `toxicity` - Cell toxicity (ChemProp model)
- `staph_aureus` - S. aureus activity (ChemProp)

### Filters
- `pains` - PAINS filter (1.0 = pass, 0.0 = fail)
- `brenk` - Brenk filter (1.0 = pass, 0.0 = fail)

### Antibiotics-Specific
- `antibiotics_novelty` - Novelty score for antibiotics
- `acinetobacter_baumanii` - A. baumannii activity (MiniMol)
- `escherichia_coli` - E. coli activity (MiniMol)
- `klebsiella_pneumoniae` - K. pneumoniae activity (MiniMol)
- `pseudomonas_aeruginosa` - P. aeruginosa activity (MiniMol)
- `neisseria_gonorrhoeae` - N. gonorrhoeae activity (MiniMol)

### COVID-19 Specific
- `mpro_his161_a` - MPro His161 interaction
- `mpro_glu164_a` - MPro Glu164 interaction
- `mpro_his39_a` - MPro His39 interaction

### Other
- `ra` - RA score (XGBoost model)

## Output Format

The script generates a CSV file with the following columns:

1. `smiles` - Input SMILES string
2. `<objective_1>` - Score for first objective
3. `<objective_2>` - Score for second objective
4. ... (one column per objective)
5. `motif_filtered` - Boolean indicating if molecule would be filtered
6. `filter_reason` - Detailed reason for filtering (empty if not filtered)

### Example Output

```csv
smiles,qed,logp,mw,motif_filtered,filter_reason
CCO,0.407,1.0,0.0,False,
c1ccccc1,0.443,1.0,0.0,False,
CC(C)Cc1ccc(C(C)C(=O)O)cc1,0.822,0.463,1.0,False,
```

## Filtering Information

The script uses `filter_smiles_preserves_existing_hits()` from `rdkit_utils.py` to check molecules against:

1. **PAINS filters** - Pan-assay interference compounds
2. **Motif filters** - Unwanted substructures including:
   - Sulfonamides
   - Aminoglycosides
   - Tetracyclic skeletons
   - Beta-lactams
   - Quinolones
   - And other antibiotic-specific motifs

Molecules are marked as `motif_filtered=True` if they match any of these filters. The `filter_reason` column provides details about which filter(s) matched.

**Note**: The filtering is informational only - all molecules are scored regardless of filter status. This allows you to see which molecules would be filtered and decide how to handle them.

## Summary Statistics

The script automatically prints summary statistics for all objectives:

```
================================================================================
SUMMARY STATISTICS
================================================================================
Total molecules: 4
Filtered by motif/PAINS: 0 (0.0%)

klebsiella_pneumoniae:
  Valid scores: 4/4 (100.0%)
  Mean: 0.0013
  Std: 0.0006
  Min: 0.0006
  Max: 0.0019
  Median: 0.0014

...
================================================================================
```

## Performance Considerations

- **Loading time**: The first run loads ML models (ChemProp, DeepDL, MiniMol), which takes ~10-30 seconds
- **Scoring speed**:
  - Simple objectives (QED, LogP, MW): Very fast (~1000s molecules/second)
  - ML-based objectives (ChemProp, DeepDL, MiniMol): Medium speed (~100s molecules/second)
  - UniDock docking: Slow (~1-10 molecules/second depending on protein size)

## Error Handling

- Invalid SMILES are skipped with a warning
- Failed scorers return `None` for that objective
- The script continues even if some objectives fail
- All errors are printed with traceback for debugging

## Tips

1. **Start small**: Test with a few molecules first to verify objectives work as expected
2. **Avoid UniDock for large datasets**: Docking is very slow; use it only when necessary
3. **Check filtering**: Review `motif_filtered` and `filter_reason` columns to understand why molecules are filtered
4. **Combine with other tools**: Use the output CSV with pandas for further analysis

## Example Workflow

```bash
# 1. Score a dataset with antibiotics objectives
python -m modules.small_molecule_drug_design.evaluation.score_smiles \
    --smiles-file generated_molecules.csv \
    --objectives klebsiella_pneumoniae toxicity pains brenk antibiotics_novelty deepdl mw \
    --output scored_molecules.csv

# 2. Analyze results with pandas
python
>>> import pandas as pd
>>> df = pd.read_csv('scored_molecules.csv')
>>>
>>> # Filter to non-filtered molecules
>>> clean = df[~df['motif_filtered']]
>>>
>>> # Find top molecules by activity and druglikeness
>>> top = clean[(clean['klebsiella_pneumoniae'] > 0.002) & (clean['deepdl'] > 0.7)]
>>> print(top.head())
```

## Troubleshooting

### ModuleNotFoundError
Ensure you're running from the SciLeoAgent root directory and have activated the genesis environment.

### Scorer not found
Check the available objectives list above. The objective name must match exactly (case-sensitive).

### CUDA/GPU errors
Some models (ChemProp, MiniMol) use GPU if available. If you encounter GPU errors, the models will fall back to CPU.

### Memory issues
For large datasets (>10,000 molecules), consider:
- Scoring in batches
- Using only necessary objectives
- Avoiding multiple heavy ML models simultaneously

## See Also

- `antibiotics_multiple_seed_analyzer.py` - Multi-seed results analysis
- `antibiotics_results_analyzer.py` - Single-seed results analysis
- `run_optimization.py` - Main optimization script
- `../../utils/rdkit_utils.py` - Filtering functions
