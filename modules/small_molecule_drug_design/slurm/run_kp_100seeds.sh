#!/bin/bash
#SBATCH --partition=jlab
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks-per-node=1
#SBATCH --time=1-00:00:00
#SBATCH --job-name=kp_100seeds
#SBATCH --mem=32GB
#SBATCH --cpus-per-task=32
#SBATCH --output=logs/kp_100seeds_%A_%a.out
#SBATCH --mail-user=tonyzshen@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=0-24%8

# Calculate the 4 seeds for this array job
# Each array task (0-24) will run 4 seeds in parallel
BASE_SEED=$((SLURM_ARRAY_TASK_ID * 4))
SEED1=$BASE_SEED
SEED2=$((BASE_SEED + 1))
SEED3=$((BASE_SEED + 2))
SEED4=$((BASE_SEED + 3))

# Create logs directory if it doesn't exist
mkdir -p logs

# Enable LiteLLM debug logging for downstream processes
export SCILEO_LITELLM_DEBUG=debug

# Base command
CMD="python modules/small_molecule_drug_design/run_optimization.py \
  --targets klebsiella_pneumoniae antibiotics_novelty deepdl toxicity qed antibiotics_motifs arthor_similarity \
  --budget 2000 \
  --population-size 40 \
  --offspring-size 35 \
  --mutation-size 5 \
  --survival-selection-method top_diverse \
  --combiner simple_product \
  --prompt-style short \
  --init_group @modules/small_molecule_drug_design/data/molecules/Enamine_screening_collection_202510.smi \
  --file-sample-size 30 \
  --no-map-to-synthesizable-analogs"

# Run 4 seeds in parallel using background processes
echo "Starting array job ${SLURM_ARRAY_TASK_ID}: Running seeds ${SEED1}, ${SEED2}, ${SEED3}, ${SEED4}"

$CMD --seed $SEED1 > logs/seed_${SEED1}.log 2>&1 &
$CMD --seed $SEED2 > logs/seed_${SEED2}.log 2>&1 &
$CMD --seed $SEED3 > logs/seed_${SEED3}.log 2>&1 &
$CMD --seed $SEED4 > logs/seed_${SEED4}.log 2>&1 &

# Wait for all background jobs to complete
wait

echo "Completed array job ${SLURM_ARRAY_TASK_ID}: seeds ${SEED1}, ${SEED2}, ${SEED3}, ${SEED4}"
