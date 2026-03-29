# SciLeo Ablation Study

Comparison of SciLeo vs baselines (REINVENT4, NatureLM, MolT5, TextGrad) for antibiotic molecule generation.

## Quick Start

```bash
# Step 1: Process all CSV files and compute pass rates on top 100 diverse molecules
conda run -n genesis python process_scileo_csvs.py --recompute-held-out

# Step 2: Run Type 2 evaluation (held-out filters analysis)
conda run -n genesis python evaluation_type2.py
```

## Critical Requirements

### ⚠️ PASS RATES MUST BE COMPUTED ON TOP 100 DIVERSE MOLECULES

**CRITICAL**: All pass rate calculations in `process_scileo_csvs.py` MUST be computed on the **top 100 diverse molecules**, NOT on all molecules in the dataset.

- The script selects top 100 diverse molecules using `select_diverse_smiles()` based on `aggregate_score`
- Pass rates are then computed ONLY on these top 100 diverse molecules
- This matches the evaluation in `evaluation_type2.py` which also uses top 100 diverse

### ⚠️ THRESHOLDS MUST MATCH evaluation_type2.py EXACTLY

**CRITICAL**: The thresholds in `process_scileo_csvs.py` MUST match `evaluation_type2.py` exactly:

**In `process_scileo_csvs.py` (PRIMARY_THRESHOLDS):**
```python
PRIMARY_THRESHOLDS = {
    "kp": 0.05,      # MUST match ACTIVITY_THRESHOLDS[0] = 0.05 from evaluation_type2.py
    "novelty": 0.6,  # MUST match FILTERS["antibiotics_novelty"] = 0.6
    "toxicity": 0.5, # MUST match FILTERS["toxicity"] = 0.5
    "motifs": 1.0,   # MUST match FILTERS["antibiotics_motifs_filter"] = 1.0
    "similarity": 0.5,
}
```

**In `evaluation_type2.py`:**
- `ACTIVITY_THRESHOLDS = [0.05, 0.1]` - KP threshold is 0.05 (lower threshold)
- `FILTERS["antibiotics_novelty"] = 0.6`
- `FILTERS["toxicity"] = 0.5`
- `FILTERS["antibiotics_motifs_filter"] = 1.0`

**If you change thresholds in `evaluation_type2.py`, you MUST update `process_scileo_csvs.py` and recompute!**

## What Each Evaluation Does

**Type 2**: Tests how many molecules pass held-out filters:
- **Top 100 Diverse Selection**: Ranked by `aggregate_score` (geometric mean of kp, novelty, toxicity, motifs, similarity), then diversity filtered (Tanimoto < 0.6)
- **Activity Thresholds**: 0.05 and 0.1 (from `ACTIVITY_THRESHOLDS`)
- **Held-out Filters**: QED, SA, DeepDL, MW, PAINS, BRENK, antibiotics_motifs_filter, antibiotics_novelty, toxicity, ring_score
- **Pass rates computed on top 100 diverse molecules** (selected BEFORE computing pass rates)

## Input Data

All scored molecules in `scileo_ablations/` directory:
- **SciLeo Level 1**: Iterations 1, 2, and 3 for E. coli
- **Baseline methods**: REINVENT4, NatureLM, MolT5, TextGrad

## Results

Results are saved in `type2_results/`:
- **CSV Files**: `passing_counts_{target}.csv`, `per_filter_pass_rates_{target}.csv`
- **Plots**: `passing_curves_*.png`, `filter_cascade_*.png`, `per_filter_pass_rates_*.png`
- **Passing Molecules**: `passing_molecules/{method}_{target}_activity{threshold}.csv`

## Output Files

Top 100 diverse molecules saved in `top100_diverse_molecules/`:
- `ecoli_level1_iter{1,2,3}_*_top100_diverse.csv` - Top 100 diverse molecules ranked by aggregate score

## Documentation

- `DATA_LOCATIONS.md` - Where to find all data files for each method
- `EVALUATION_OUTPUTS.md` - Complete list of files created by evaluation

## Scripts Overview

### `process_scileo_csvs.py`

**Purpose**: Process all CSV files in `scileo_ablations/` and compute pass rates for plotting.

**Workflow**:
1. Load CSV file
2. Normalize column names (map to PRIMARY_METRICS: kp, novelty, toxicity, motifs, similarity)
3. Compute `aggregate_score` = geometric mean (product) of PRIMARY_METRICS
4. Compute held-out metrics (QED, SA, MW, PAINS, BRENK, DeepDL, ring_score, antibiotics_motifs_filter)
5. **Select top 100 diverse molecules** using `select_diverse_smiles()` (Tanimoto < 0.6)
6. **Compute pass rates ONLY on top 100 diverse molecules** (NOT all molecules!)
7. Save results to `results/held_out_pass_rates.csv`

**Key Functions**:
- `normalize_columns()`: Maps CSV columns to PRIMARY_METRICS (handles different naming conventions)
- `select_diverse_smiles()`: Greedy diversity selection (Tanimoto < 0.6)
- `summarize_all_metrics()`: Computes pass rates using PRIMARY_THRESHOLDS

**Output**: `results/held_out_pass_rates.csv` - Used by `plot_data.ipynb` for visualization

### `evaluation_type2.py`

**Purpose**: Comprehensive evaluation with filter cascades and passing molecule lists.

**Workflow**:
1. Load data (SciLeo or baseline)
2. Select top 100 diverse molecules (if not pre-computed)
3. Compute held-out metrics
4. Apply filter cascade at activity thresholds 0.05 and 0.1
5. Generate plots and save passing molecules

**Output**: `type2_results/` directory with plots and passing molecule CSVs

## Column Mapping

Different CSV files use different column names. `normalize_columns()` handles mapping:

- **KP**: `klebsiella_pneumoniae_minimol` or `klebsiella_pneumoniae` → `kp`
- **Novelty**: `antibiotics_novelty` → `novelty`
- **Toxicity**: `toxicity_safety_chemprop` → `toxicity`
- **Motifs**: `antibiotics_motifs_filter` → `motifs` (boolean → float: True=1.0, False=0.0)
- **Similarity**: `arthor_similarity` or `similarity` → `similarity`

## When to Recompute

**You MUST recompute `held_out_pass_rates.csv` when:**
1. Thresholds change in `evaluation_type2.py` → Update `PRIMARY_THRESHOLDS` in `process_scileo_csvs.py` and run `process_scileo_csvs.py --recompute-held-out`
2. New CSV files are added to `scileo_ablations/`
3. CSV files are updated with new data
4. Column mappings change

**Command to recompute:**
```bash
conda run -n genesis python process_scileo_csvs.py --recompute-held-out
```

## Notes

- **CRITICAL**: Pass rates are computed on top 100 diverse molecules, NOT all molecules
- Top 100 diverse molecules ranked by `aggregate_score` (geometric mean of PRIMARY_METRICS)
- Diversity threshold: Tanimoto similarity < 0.6
- REINVENT4 limited to first 10k molecules before top-100 diverse selection (in `evaluation_type2.py`)
- All thresholds MUST match between `process_scileo_csvs.py` and `evaluation_type2.py`
