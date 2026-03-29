#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1 # This needs to be equal to the number of GPUs
#SBATCH --time=2-00:00:00
#SBATCH --job-name=scileo
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=8
#SBATCH --output=scileo.%j.out
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL


python modules/small_molecule_drug_design/run_optimization.py "$@"