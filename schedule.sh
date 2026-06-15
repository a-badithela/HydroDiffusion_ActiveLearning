#!/bin/bash -l

# Usage: bash schedule.sh [model] [n_seeds] [stride]
#   model   : encdec_lstm | seq2seq_lstm | diffusion_lstm | decoder_only_ssm  (default: encdec_lstm)
#   n_seeds : number of ensemble seeds to run                                 (default: 5)
#   stride  : dataset window stride (default: 1; use 90 for fast testing)
#
# Examples:
#   bash schedule.sh                          # encdec_lstm, 5 seeds, stride 1
#   bash schedule.sh encdec_lstm 3            # encdec_lstm, 3 seeds, stride 1
#   bash schedule.sh encdec_lstm 5 90         # encdec_lstm, 5 seeds, stride 90

MODEL=${1:-encdec_lstm}
N_SEEDS=${2:-5}
STRIDE=${3:-1}
FIRST_SEED=3407

EMAIL="badit004@umn.edu"
LOG_DIR=/projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning/logs/seed_split_geo/$MODEL/$N_SEEDS
mkdir -p "$LOG_DIR"

echo "Model:   $MODEL"
echo "Seeds:   $FIRST_SEED to $((FIRST_SEED + N_SEEDS - 1))"
echo "Stride:  $STRIDE"
echo "Logs:    $LOG_DIR"
echo "================================"

for (( i=0; i<N_SEEDS; i++ )); do
    SEED=$((FIRST_SEED + i))
    echo "Submitting seed=$SEED ..."
    sbatch \
        --export=ALL,SEED=$SEED,MODEL=$MODEL,STRIDE=$STRIDE \
        --output="$LOG_DIR/${MODEL}_geo-split-100_seed${SEED}_%j.txt" \
        slurm_encdec_lstm.sh
done
