# Small Molecule Drug Design - Material Collection

This document consolidates all code paths, experiment instructions, evaluation scripts, and documentation for the small molecule drug design project.

## Table of Contents

1. [Code Locations](#code-locations)
2. [Results Location](#results-location)
3. [How the Optimizer Works](#how-the-optimizer-works)
4. [Experiment Analysis](#experiment-analysis)
5. [Known Caveats and Issues](#known-caveats-and-issues)

---

## Code Locations

### Experiment Scripts (Levels 0-3)

**Main Experiment Script:**
- `exps/small_molecule_drug_design/exp_kp.py`

**Commands:**
```bash
# Level 0: 1 iteration, both feedbacks enabled
python exp_kp_manual_level_0.py --level 0 --run-name kp_manual_level_0 --seed 7

# Level 1: Multiple iterations, both feedbacks enabled
python exp_kp.py --level 1 --run-name kp_level_1 --seed 7

# Level 2: Multiple iterations, analyzer feedback only
python exp_kp.py --level 2 --run-name kp_level_2 --seed 7

# Level 3: Multiple iterations, fully automated
python exp_kp.py --level 3 --run-name kp_level_3 --seed 7
```

### Optimizer-Only Test (MolGA / No-LLM)

**Script:** `exps/small_molecule_drug_design/exp_kp_manual_level_0.py`

**Command:**
```bash
# MolGA baseline (no LLM, graph-based GA only)
python exp_kp_manual_level_0.py --level 0 --no-llm --run-name kp_molga_level_0 --seed 7

# Level 0 with LLM
python exp_kp_manual_level_0.py --level 0 --run-name kp_manual_level_0 --seed 7
```

The `--no-llm` flag disables LLM-guided crossover/mutation and uses graph-based genetic algorithm (GB-GA) operations only.

### Coding Agent Test (Scorer Reproduction)

**Location:** `exps/small_molecule_drug_design/scorer_reproduction_test/`

**Purpose:** Evaluates the ability of a coding agent to reproduce expert-implemented molecular property scorers using only natural language descriptions.

**Instructions:**
```bash
# 1. Extract scorer descriptions
python extract_scorer_descriptions.py

# 2. Create test dataset
python create_test_dataset.py

# 3. Run evaluation (all scorers)
python run_scorer_reproduction.py

# Evaluate specific scorers
python run_scorer_reproduction.py --scorers mpro_unidock qed sa_score

# Evaluate by category
python run_scorer_reproduction.py --category property
```

See `exps/small_molecule_drug_design/scorer_reproduction_test/README.md` for full documentation.

### Baselines (Domain-Specific Methods + TextGrad)

> **TODO (Tianyu):** Document baseline implementation locations and run instructions.

The following baselines are implemented in external repositories:
- REINVENT4
- NatureLM
- MolT5
- TextGrad

Baseline outputs for evaluation are stored in:
- `modules/small_molecule_drug_design/ablations/scileo_ablations/`

### Evaluation and Plotting Scripts

**Single Run Evaluation** (for evaluating a new experiment):
```bash
conda activate genesis
python modules/small_molecule_drug_design/ablations/evaluate_iteration.py runs/{run_id}/logs/iteration_X
```
- Aggregates CSVs from one iteration directory
- Selects top 100 diverse molecules (Tanimoto < 0.6)
- Computes held-out metrics (QED, SA, PAINS, BRENK, etc.)
- Outputs pass rates for that iteration

<!-- **Batch Comparison** (for generating paper figures across all methods):
```bash
conda activate genesis
# Step 1: Process all CSVs in scileo_ablations/ and compute pass rates
python modules/small_molecule_drug_design/ablations/process_scileo_csvs.py --recompute-held-out

# Step 2: Generate plots and filter cascade visualizations
python modules/small_molecule_drug_design/ablations/evaluation_type2.py
``` -->

**Plotting Notebook:**
- `modules/small_molecule_drug_design/ablations/plot_data.ipynb`
- Uses `results/held_out_pass_rates.csv` from Step 1 above

**Key Evaluation Scripts:**
| Script | Purpose | Input |
|--------|---------|-------|
| `evaluate_iteration.py` | Evaluate single run | `runs/{run_id}/logs/iteration_X` |
| `process_scileo_csvs.py` | Batch process all methods | `scileo_ablations/*.csv` |
| `evaluation_type2.py` | Generate plots & passing molecules | Pre-computed top 100 files |
| `plot_data.ipynb` | Comparison bar charts | `results/held_out_pass_rates.csv` |

See `modules/small_molecule_drug_design/ablations/README.md` for detailed evaluation documentation.

---

## Results Location

**Google Drive (Primary):**
https://drive.google.com/drive/folders/1xjeCyTcxOX7yH01zoVcHhTm789Wh7y-Q

**Local Runs:**
- Run outputs: `runs/{run_id}/`
- Logs: `runs/{run_id}/logs/iteration_X/`
- Per-run CSVs: `runs/{run_id}/logs/iteration_X/per_run/`

**Ablation Study Data:**
- `modules/small_molecule_drug_design/ablations/scileo_ablations/` - Scored molecule CSVs
- `modules/small_molecule_drug_design/ablations/results/` - Pass rate results
- `modules/small_molecule_drug_design/ablations/type2_results/` - Type 2 evaluation outputs

---

## How the Optimizer Works

The `LLMSBDDOptimizer` (`modules/small_molecule_drug_design/llm_sbdd_optimizer.py`) implements an LLM-enhanced genetic algorithm for multi-objective molecular optimization.

### Algorithm Overview

1. **Initialization:** Sample 120 molecules from Enamine screening collection
2. **Evaluation:** Score population using multi-objective scorers (product combination)
3. **Parent Selection:** Tournament selection (size=3) based on aggregated fitness
4. **Crossover:** Generate 70 offspring via LLM-guided crossover
5. **Mutation:** Apply LLM-guided mutation to top 7 candidates
6. **Survival Selection:** Select next generation using diverse_top (Tanimoto < 0.4)
7. **Repeat** until oracle budget (10,000) exhausted

### Key Components

| Component | Description |
|-----------|-------------|
| **LLM Crossover** | GPT-5-mini combines parent molecules via barebone prompts |
| **LLM Mutation** | GPT-5-mini suggests modifications to top candidates |
| **GB-GA Fallback** | Graph-based genetic operations when `--no-llm` flag used |
| **Tournament Selection** | Selects 2 parents via size-3 tournament |
| **Multi-Objective Combiner** | Product of all objective scores (simple_product) |
| **Elitism** | Preserves top 5% of candidates (by KP activity score) |

### Optimizer Parameters (as configured in exp_kp.py)

| Parameter | Value | Description |
|-----------|-------|-------------|
| `population_size` | 120 | Number of candidates in population |
| `offspring_size` | 70 | Number of offspring per generation |
| `mutation_size` | 7 | Number of mutations per generation |
| `oracle_budget` | 10000 | Maximum oracle evaluations |
| `tournament_size` | 3 | Tournament size for parent selection |
| `survival_selection_method` | "diverse_top" | Diversity-aware selection |
| `mutation_mode` | "llm" | Use LLM for mutations |
| `elitism_fraction` | 0.05 | Fraction of elites to preserve |
| `elitism_fields` | ["klebsiella_pneumoniae_minimol"] | Elite selection by KP activity |
| `use_barebone_prompts` | True | Short prompts for speed |
| `seed` | 42 | Random seed |

### Loop Configuration (exp_kp.py)

| Parameter | Level 0 | Levels 1-3 |
|-----------|---------|------------|
| `max_iterations` | 1 | 5 |
| `random_candidate_ratio` | 1.0 | 1.0 |

### LLM Configuration

| Role | Model |
|------|-------|
| Optimizer (crossover/mutation) | `openai/gpt-5-mini` |
| Planner | `anthropic/claude-sonnet-4-5-20250929` |
| Scorer Creator | `anthropic/claude-sonnet-4-5-20250929` |
| Analyzer | `anthropic/claude-sonnet-4-5-20250929` |

### Initial Population

The initial population is sampled from:
```
@modules/small_molecule_drug_design/data/molecules/Enamine_screening_collection_202510.smi
```
(120 molecules sampled)

### Objectives (K. pneumoniae Antibiotics)

| Objective | Direction | Type | Description |
|-----------|-----------|------|-------------|
| `klebsiella_pneumoniae_minimol` | maximize | candidate | MiniMol predicted activity vs K. pneumoniae |
| `antibiotics_novelty` | maximize | candidate | Tanimoto distance from known antibiotics |
| `toxicity_safety_chemprop` | maximize | candidate | ChemProp toxicity safety score |
| `antibiotics_motifs_filter` | - | filter | Avoid problematic antibiotic motifs |
| `local_similarity` | maximize | candidate | Similarity to Enamine purchasable molecules |

### Survival Selection Strategies

- **fitness:** Sort by aggregated score, keep top N (code default)
- **diverse_top:** Top-k with diversity preservation via Tanimoto < 0.4 filtering (**used in exp_kp.py**)
- **butina_cluster:** Butina clustering, pick top per cluster, round-robin selection

### Multi-Objective Combination

Configured via `objective_combiner` (default: `simple_product`):
- `simple_product`: Multiply all objective scores (**used**)
- `simple_sum`: Sum all objective scores
- `weighted_sum`: Weighted sum of scores
- `antibiotic_geomean`: Domain-specific geometric mean

---

## Experiment Analysis

> **TODO:** Add experiment analysis documentation here.

---

## Known Caveats and Issues

> **TODO:** Add known issues and caveats here.

---

## Quick Reference

### Environment Setup
```bash
conda activate genesis
```

### Run Level 1 Experiment
```bash
cd exps/small_molecule_drug_design
python exp_kp.py --level 1 --run-name my_experiment --seed 42
```

### Evaluate Results
```bash
python modules/small_molecule_drug_design/ablations/evaluate_iteration.py runs/{run_id}/logs/iteration_1
```

### Generate Plots
```bash
cd modules/small_molecule_drug_design/ablations
jupyter notebook plot_data.ipynb
```

