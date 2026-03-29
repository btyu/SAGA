# Small Molecule Drug Design Module

This module provides molecular optimization capabilities using evolutionary algorithms with LLM-enhanced optimization for drug design. It supports multi-objective optimization with docking scores, druglikeness, toxicity predictions, and antibiotics-specific properties.

## Installation

### Prerequisites

- **Conda/Miniconda**: Required for managing dependencies
- **CUDA 12.1**: Required for GPU acceleration (PyTorch with CUDA support)
- **Git**: For cloning dependency repositories

### Step 1: Create Conda Environment

```
conda create -n genesis python=3.11
conda activate genesis
```

### Step 2: Install Conda Packages

Install scientific computing and cheminformatics packages:

```bash
conda install --yes -c conda-forge \
  unidock ambertools parmed openbabel graphium \
  biopython rdkit numpy scipy pandas packaging openmm \
  lightning umap-learn plip contourpy

conda install faiss-gpu -c pytorch -y # faiss-gpu is required for faiss clustering
```

**What's being installed:**
- `unidock`: Molecular docking tool
- `rdkit`, `openbabel`: Cheminformatics libraries for molecular manipulation
- `ambertools`, `parmed`, `openmm`: Molecular dynamics and force field tools
- `biopython`: Biological computation tools
- Standard scientific libraries: `numpy`, `scipy`, `pandas`

### Step 3: Install PyTorch with CUDA Support

```bash
# Install PyTorch 2.4.0 with CUDA 12.1 support
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
  --index-url https://download.pytorch.org/whl/cu121

# Install PyTorch Geometric and dependencies
pip install --force-reinstall \
  -f https://data.pyg.org/whl/torch-2.4.0+cu121.html \
  pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
```

**Note:** If you don't have CUDA 12.1, modify the URL to match your CUDA version or use CPU-only PyTorch.

### Step 4: Install SciLeo Framework and Dependencies

Navigate to the SciLeoAgent repository root:

```bash
cd /path/to/SciLeoAgent
```

Install the main framework requirements:

```bash
pip install -r requirements.txt
```

**What's in requirements.txt:**
- Core SciLeo framework dependencies
- LLM client libraries (OpenAI, Anthropic, etc.)
- Logging and utilities

### Step 5: Install Domain-Specific Drug Design Packages

Install molecular optimization and scoring tools:

```bash
pip install minimol \
  "unidock_tools @ git+https://github.com/dptech-corp/Uni-Dock.git@1.1.2#subdirectory=unidock_tools" \
  "descriptastorus @ git+https://github.com/bp-kelley/descriptastorus" \
  "git+https://github.com/SeonghwanSeo/drug-likeness.git" \
  useful-rdkit-utils
```

**What's being installed:**
- `minimol`: Machine learning models for molecular property prediction
- `unidock_tools`: Python interface for UniDock molecular docking
- `descriptastorus`: Molecular descriptor calculation
- `drug-likeness`: Drug-likeness scoring models
- `useful-rdkit-utils`: RDKit utilities for ring system analysis (RingSystemLookup) - required for optimizer objectives

Fix compatibility issues:

```bash
pip install chemprop==1.6.1 scipy==1.12
```

### Step 6: Setup Antibiotics Oracles (Required for Antibiotics Optimization)

Download and setup the antibiotics prediction models and data:

```bash
# Install gdown if not already installed
pip install gdown

# Download the antibiotics oracles archive from Google Drive
gdown 1uLBn6_tSEw7IrkldOHmx2NKTfVVUfuDa -O antibiotics_oracles.zip
# Unzip the antibiotics oracles into two required locations:
# 1. Oracle models directory
unzip antibiotics_oracles.zip -d modules/small_molecule_drug_design/oracles/minimol_antibiotics/

# 2. Scorer MCP data directory
unzip antibiotics_oracles.zip -d modules/small_molecule_drug_design/scorer_mcp/minimol_scorer_mcp/scorer_data/minimol_antibiotics/

# Clean up the zip file (optional)
rm antibiotics_oracles.zip
```


### Step 7: Setup ChemProp Model Weights

Download and setup the ChemProp model weights for MCP scorer:

```bash
# Create the target directory if it doesn't exist
mkdir -p modules/small_molecule_drug_design/scorer_mcp/chemprop_scorers_mcp/scorer_data/antibiotics/models/

# Download ChemProp weights from Dropbox
# Note: For Dropbox shared folders, you may need to download manually or use the Dropbox CLI
# Option 1: Manual download (recommended)
# 1. Visit: https://www.dropbox.com/scl/fo/strzroxdkr2pertl2u444/h?rlkey=z5k2t4tt0v356vsy9p33i6pym&st=jod66ytv&dl=0
# 2. Download the folder as a ZIP file
# 3. Extract to: modules/small_molecule_drug_design/scorer_mcp/chemprop_scorers_mcp/scorer_data/antibiotics/models/

# Option 2: Using Dropbox CLI (if installed)
# Install: pip install dropbox
# Then use: dropbox download /path/to/folder --output modules/small_molecule_drug_design/scorer_mcp/chemprop_scorers_mcp/scorer_data/antibiotics/models/
```

**What's being installed:**
- ChemProp model weights for primary cell toxicity prediction (stored in `primary_cell_toxicity_model/train/checkpoints*/fold_0/model_0/model.pt`)
- ChemProp model weights for Staphylococcus aureus activity prediction (stored in `staph_aureus_model/checkpoints*/fold_0/model_0/model.pt`)
- These weights are required for the `toxicity_safety_chemprop` scorer

### Step 8: Setup Nearest Neighbor Files

```bash
gdown --folder 1Xe8vsrTBM0g7SGYGVCLJaCxmIvzzzvUw -O modules/small_molecule_drug_design/scorer_mcp/local_similarity_scorer_mcp/data

cd /home/tsa87/SciLeoAgent/exps/small_molecule_drug_design

# IMPORTANT: Make sure test_faiss_gpu.py and test_similarity.py works. You need to have a GPU.
conda install faiss-gpu -c pytorch -y
python test_faiss_gpu.py
python test_similarity.py 
```

<!-- **What's being installed:**
- MiniMol antibiotics prediction models for gram-negative bacteria (E. coli, K. pneumoniae, etc.)
- Supporting data files and cutoffs for scoring -->

<!-- ### Step 7: (Optional) Setup RA Score Oracle

RA Score predicts retrosynthetic accessibility:

```bash
mkdir -p modules/small_molecule_drug_design/oracles/ra_score
wget -O modules/small_molecule_drug_design/oracles/ra_score/models.zip \
  https://github.com/reymond-group/RAscore/raw/master/RAscore/models/models.zip
unzip -o modules/small_molecule_drug_design/oracles/ra_score/models.zip \
  -d modules/small_molecule_drug_design/oracles/ra_score
``` -->
<!-- 
### Step 8: Prepare Target Protein and Reference Data

Download and process PDB files for docking targets and reference molecule libraries:

```bash
cd modules/small_molecule_drug_design
python scripts/prepare_pdb_files.py      # Downloads and prepares protein structures
python scripts/prepare_zinc_250k_file.py # Prepares ZINC molecule database subset
```

**What this does:**
- `prepare_pdb_files.py`: Downloads PDB structures for target proteins (e.g., Mpro, bacterial targets)
- `prepare_zinc_250k_file.py`: Prepares a curated subset of ZINC database for initial populations
## Testing the Setup

### Test UniDock Docking Module

```bash
cd modules/small_molecule_drug_design
python docking/unidock.py
```

### Test UniDock Scorer Integration

```bash
cd modules/small_molecule_drug_design
python unidock_scorer.py
```

## Running Molecular Optimization

### Basic Usage

```bash
# Direct execution
python modules/small_molecule_drug_design/run_optimization.py \
  --targets <target_list> \
  --combiner <combiner_name> \
  --name <run_name> \
  --seed <seed>

# Using SLURM for HPC
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh <args>
``` -->
<!-- 
### Common Parameters

- `--targets`: Space-separated list of optimization targets (e.g., `escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw`)
- `--combiner`: Multi-objective combination strategy (`antibiotic_geomean`, `covid_simple`)
- `--survival-selection-method`: Selection strategy (`butina_cluster`, `diverse_top`)
- `--init_group`: Initial population from CSV file (prefix with `@`, e.g., `@filtered_E_coli_avg.csv`)
- `--elitism-fraction`: Fraction of top candidates to preserve (e.g., `0.05`)
- `--elitism-fields`: Fields to use for elitism selection
- `--prompt-style`: Prompt style for LLM (`short`, `long`, `barebone`)
  - `short`: Concise prompts with SMILES output only and basic requirements
  - `long`: Detailed prompts with full guardrails and reasoning
  - `barebone`: Minimal prompts with no requirements - just maximize reward
- `--enamine-percentage`: Percentage of molecules to sample from Enamine dataset (0.0-1.0). Remaining percentage comes from init_group source (default: 1.0)
- `--seed`: Random seed for reproducibility
- `--name`: Run name for output logging

### Example: Antibiotics Optimization

#### Level 0 (Basic Objectives)

Using short prompt with basic requirements:
```bash
# LEVEL 0: Basic Objectives
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity antibiotics_novelty brenk qed mw\
  --name ecoli_level0_final --prompt-style short  \
  --seed 1

sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets klebsiella_pneumoniae toxicity antibiotics_novelty \
  --name klebsiella_pneumoniae_level0_short_updated --prompt-style short  \
  --seed 1

# LEVEL 1: Antibiotics Specific Objectives
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
  --name escherichia_coli_level1_100_pct_enamine --prompt-style short  \
  --init_group @ecoli_level0_last_generation.csv \
  --enamine-percentage 1.0 --seed 777

sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
  --name escherichia_coli_level1_80_pct_enamine --prompt-style short  \
  --init_group @ecoli_level0_last_generation.csv \
  --enamine-percentage 0.8 --seed 777

sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
  --name escherichia_coli_level1_50_pct_enamine --prompt-style short  \
  --init_group @ecoli_level0_last_generation.csv \
  --enamine-percentage 0.5 --seed 777

sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
  --name escherichia_coli_level1_0_pct_enamine --prompt-style short  \
  --init_group @ecoli_level0_last_generation.csv \
  --enamine-percentage 0.0 --seed 777
```

Using barebone prompt (no requirements - just maximize reward):
```bash
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity antibiotics_novelty \
  --name ecoli_level0_barebone_updated --prompt-style barebone  \
  --seed 1

sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets klebsiella_pneumoniae toxicity antibiotics_novelty \
  --name klebsiella_pneumoniae_level0_barebone_updated --prompt-style barebone  \
  --seed 1
```

### Running the new Klebsiella pneumoniae experiment (`exp_kp_new.py`) via SLURM

Use the dedicated SLURM wrapper we added in `exps/small_molecule_drug_design/slurm/run_exp_kp_new.sh`. Example (Level 0, seed 42):

```bash
sbatch exps/small_molecule_drug_design/slurm/run_exp_kp_new.sh \
  --level 0 \
  --seed 42
```




#### With Random Initial Population

```bash
# E. coli optimization
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
  --survival-selection-method butina_cluster \
  --combiner antibiotic_geomean \
  --name escherichia_coli_v1 \
  --seed 1

# K. pneumoniae optimization
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets klebsiella_pneumoniae toxicity pains brenk antibiotics_novelty deepdl mw \
  --survival-selection-method butina_cluster \
  --combiner antibiotic_geomean \
  --name klebsiella_pneumoniae_v1 \
  --seed 1
```

#### With Screened Initial Population

```bash
# E. coli with diverse top selection
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets escherichia_coli toxicity pains brenk antibiotics_novelty deepdl mw \
  --survival-selection-method diverse_top \
  --combiner antibiotic_geomean \
  --name escherichia_coli_screened_v2 \
  --elitism-fraction 0.05 \
  --elitism-fields escherichia_coli \
  --init_group @filtered_E_coli_avg_filtered.csv \
  --seed 99

# K. pneumoniae with diverse top selection
sbatch modules/small_molecule_drug_design/slurm/run_scileo.sh \
  --targets klebsiella_pneumoniae toxicity pains brenk antibiotics_novelty deepdl mw \
  --survival-selection-method diverse_top \
  --combiner antibiotic_geomean \
  --name klebsiella_pneumoniae_screened_v2 \
  --elitism-fraction 0.05 \
  --elitism-fields klebsiella_pneumoniae \
  --init_group @filtered_K_pneumoniae_avg_filtered.csv \
  --seed 999
```

### Selection Strategies

- **`butina_cluster`**: Clusters non-elite candidates using Butina clustering, picks top per cluster, then round-robin up to 3 per cluster for diversity
- **`diverse_top`**: Top-k selection with diversity preservation through minimal elitism

### Example: COVID-19 Mpro Optimization

```bash
# Default configuration
python modules/small_molecule_drug_design/run_optimization.py \
  --targets mpro deepdl pains brenk mpro_his161_a mpro_glu164_a mpro_his39_a \
  --combiner covid_simple \
  --name mpro_default

# With known binders as initial population
python modules/small_molecule_drug_design/run_optimization.py \
  --targets mpro deepdl pains brenk mpro_his161_a mpro_glu164_a mpro_his39_a \
  --combiner covid_simple \
  --init_group covid \
  --name mpro_known_binders

# With known binders and diverse top selection
python modules/small_molecule_drug_design/run_optimization.py \
  --targets mpro deepdl pains brenk mpro_his161_a mpro_glu164_a mpro_his39_a \
  --combiner covid_simple \
  --init_group covid \
  --survival-selection-method diverse_top \
  --name mpro_known_binders_diverse
```

## Analyzing Results

### Evaluate Antibiotics Optimization Performance

```bash
python -m modules.small_molecule_drug_design.experimental.antibiotics_results_analyzer \
  --input-dir /path/to/logs/run_folder
```

## Baseline Comparisons

To compare against GraphGA or other baseline methods:

```bash
cd modules/small_molecule_drug_design
git clone https://github.com/wenhao-gao/mol-opt.git
``` -->