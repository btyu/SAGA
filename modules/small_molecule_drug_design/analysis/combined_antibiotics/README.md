# Combined Antibiotics Fragment Analysis

Artifacts produced by the workflow:

- `filtered_smiles.smi` – SMILES from `data/molecules/combined_antibiotics.txt` whose heavy-atom count is ≤ 80.
- `fragments.smi` – unique AnyBreak fragments generated from the filtered molecules (one fragment per line).
- `kp_scores.csv` – KP predictor outputs for every fragment, including the SMILES and score columns.
- `summary.md` – quick stats (counts, min/max/mean KP) plus references to the files above.

All paths in this folder are relative to `modules/small_molecule_drug_design/analysis/combined_antibiotics/`.






