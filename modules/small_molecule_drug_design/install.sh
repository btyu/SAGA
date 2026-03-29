#!/usr/bin/env bash
set -euo pipefail

# Simple installer for SciLeoAgent dependencies
# Usage:
#   bash install.sh           # CUDA 12.1 build (default)
#   bash install.sh --cpu     # CPU-only build

USE_CPU=0
if [[ "${1:-}" == "--cpu" ]]; then
  USE_CPU=1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH. Activate your conda and re-run." 1>&2
  exit 1
fi

# Move to repo root (directory containing this script)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "[1/5] Installing conda packages..."
conda install -y -c conda-forge \
  unidock ambertools parmed openbabel graphium \
  biopython rdkit numpy scipy pandas packaging openmm \
  lightning umap-learn plip contourpy

echo "[2/5] Installing PyTorch..."
if [[ "$USE_CPU" -eq 1 ]]; then
  python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cpu
else
  python -m pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 \
    --index-url https://download.pytorch.org/whl/cu121
fi

echo "[3/5] Installing PyG (torch-geometric) stack..."
if [[ "$USE_CPU" -eq 1 ]]; then
  python -m pip install --force-reinstall \
    -f https://data.pyg.org/whl/torch-2.4.0+cpu.html \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
else
  python -m pip install --force-reinstall \
    -f https://data.pyg.org/whl/torch-2.4.0+cu121.html \
    pyg-lib torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric
fi

echo "[4/5] Installing project requirements..."
python -m pip install -r requirements.txt

echo "[5/5] Installing extra packages..."
python -m pip install minimol \
  "unidock_tools @ git+https://github.com/dptech-corp/Uni-Dock.git@1.1.2#subdirectory=unidock_tools" \
  "descriptastorus @ git+https://github.com/bp-kelley/descriptastorus" \
  "git+https://github.com/SeonghwanSeo/drug-likeness.git" \
  useful-rdkit-utils

echo "Done."

conda activate scileo

# Unidock and AmberTools are required for docking and scoring
conda install unidock -c conda-forge -y
conda install ambertools parmed -c conda-forge -y
conda install -c conda-forge openbabel -y
# faiss-gpu is required for faiss clustering
conda install faiss-gpu -c pytorch

# RDKit is required for scoring (conda version includes SA_Score)
conda install -c conda-forge rdkit -y


pip install -r requirements.txt -y
pip install -r modules/small_molecule_drug_design/requirements.txt -y
pip install git+https://github.com/bp-kelley/descriptastorus -y