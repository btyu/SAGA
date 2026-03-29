#! /bin/bash
# The runtime system will be Ubuntu 22.04, with CUDA 12.4 and cuDNN installed. 
# It has Python 3.11 via Miniconda with the following packages pre-installed:

# Core Python & Scientific Computing
# - python==3.11.13
# - numpy==1.26.4
# - cupy==13.6.0
# - scipy==1.16.2
# - pandas==2.3.2
# - scikit-learn==1.7.2
# - matplotlib-base, plotly, seaborn
# - jupyter==1.1.1, notebook==7.4.5

# Machine Learning & Deep Learning
# - pytorch==2.4.1 (with CUDA 12.4 support)
# - pytorch-lightning==2.5.5
# - transformers==4.56.1
# - datasets==4.0.0
# - dgl==2.4.0 (Deep Graph Library)
# - optuna==3.6.1
# - wandb==0.21.1
# - tensorboard==2.20.0

# Chemistry & Materials Science
# - rdkit==2024.03.6
# - rdchiral==1.1.0
# - pymatgen==2025.6.14
# - ase==3.26.0 (Atomic Simulation Environment)
# - ambertools==24.8
# - openmm==8.1.2
# - openbabel==3.1.1
# - xtb-python==22.1
# - alignn==2024.12.12
# - unidock==1.1.2
# - unidock_tools @ git+https://github.com/dptech-corp/Uni-Dock.git@1.1.2#subdirectory=unidock_tools
# - descriptastorus @ git+https://github.com/bp-kelley/descriptastorus.git
# - drug-likeness @ git+https://github.com/SeonghwanSeo/drug-likeness.git
# - uxtbpy @ git+https://github.com/hkneiding/uxtbpy@089b930c71ef5fea0600c6c8099dfc7ac3d54d71

# Bioinformatics & Genomics
# - biopython==1.85
# - grelu==1.0.4.post1.dev0

# Additional Tools
# - litellm==1.77.0
# - mcp==1.14.0
# - pydantic==2.11.7
# - pydantic-settings==2.10.1
# - python-dotenv==1.1.1
# - typing-extensions==4.15.0
# - pyyaml==6.0.2
# - loguru==0.7.3
# - rich==14.1.0
# - pytest==8.4.2
# - ray==2.49.1
# - xgboost==3.0.5
# - umap-learn==0.5.9

# If the current environment cannot meet the requirements of running your scorer, you can write the installation commands here using conda, mamba, or pip. This script will be executed in the runtime environment before running your scorer.

# Install PLIP for protein-ligand interaction profiling (needed for MPRO scorers)
# Use --no-deps to avoid reinstalling openbabel which is already installed via conda
pip install plip --no-deps
