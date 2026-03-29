#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=3-00:00:00
#SBATCH --job-name=minimol_array
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=16
#SBATCH --mail-type=ALL
#SBATCH --exclude=d404
#SBATCH --array=0-9%5  # Run 10 jobs, max 5 concurrent

# Input and output files
INPUT_FILE="enamine.csv"
OUTPUT_PREFIX="antibiotics_enamine"

# Create chunk files if they don't exist
if [ ! -f "input_chunk_000.csv" ]; then
    echo "Creating input chunks..."
    python split_input.py $INPUT_FILE --chunks=10 --prefix=input
fi

# Get the chunk file for this array task
CHUNK_FILE="input_chunk_$(printf "%03d" $SLURM_ARRAY_TASK_ID).csv"
OUTPUT_FILE="${OUTPUT_PREFIX}_chunk_$(printf "%03d" $SLURM_ARRAY_TASK_ID).csv"

echo "Processing chunk $SLURM_ARRAY_TASK_ID: $CHUNK_FILE -> $OUTPUT_FILE"

# Run the processing
python smiles_minimol_scoring.py $CHUNK_FILE $OUTPUT_FILE

