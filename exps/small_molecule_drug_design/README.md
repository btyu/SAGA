MolGA command: python exp_kp_manual_level_0.py --level 0 --no-llm  --run-name kp_molga_level_0 --seed 7
Level 0 command: python exp_kp_manual_level_0.py --level 0 --run-name kp_manual_level_0 --seed 7
Level 1 command: python exp_kp.py --level 1 --run-name kp_level_1 --seed 7
Level 2 command: python exp_kp.py --level 2 --run-name kp_level_2 --seed 7
Level 3 command: python exp_kp.py --level 3 --run-name kp_level_3 --seed 7

## Results Location

Run results are saved in `runs/{run_id}/logs/iteration_X/` directories:
- CSV files with molecules are in `runs/{run_id}/logs/iteration_X/per_run/`
- Files end with `_original.csv`, `_mutation.csv`, `_selected.csv`, or `_crossover.csv`

Example: `runs/kp_antibiotics_001_level1_seed42-20251120115923/logs/iteration_1/`

## Evaluation

To evaluate a run iteration (aggregate SMILES, select top 100 diverse, compute type 2 properties, and get pass rates):

```bash
conda activate genesis
python modules/small_molecule_drug_design/ablations/evaluate_iteration.py runs/{run_id}/logs/iteration_X
```

Example:
```bash
python modules/small_molecule_drug_design/ablations/evaluate_iteration.py runs/kp_antibiotics_001_level1_seed42-20251120115923/logs/iteration_1
```

The script will:
1. Find all CSV files ending with `_original`, `_mutation`, `_selected`, or `_crossover`
2. Aggregate SMILES and compute aggregate score (multiplication of objectives)
3. Select top 100 diverse molecules (Tanimoto < 0.6)
4. Compute all type 2 evaluation properties (QED, SA, MW, PAINS, BRENK, DeepDL, etc.)
5. Calculate pass rates for each metric and total pass rate
6. Print summary to console and save results to `{iteration_X}_type2_results.csv`
