#!/bin/bash -l

# Usage: bash schedule.sh [model] [n_seeds]
#   model   : encdec_lstm | seq2seq_lstm | diffusion_lstm | decoder_only_ssm  (default: encdec_lstm)
#   n_seeds : number of ensemble seeds to run                                 (default: 5)
#
# Examples:
#   bash schedule.sh                          # encdec_lstm, 5 seeds
#   bash schedule.sh encdec_lstm 3            # encdec_lstm, 3 seeds
#   bash schedule.sh diffusion_lstm 5         # diffusion model, 5 seeds

MODEL=${1:-encdec_lstm}
N_SEEDS=${2:-5}
FIRST_SEED=3407

EMAIL="badit004@umn.edu"
LOG_DIR=/projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning/logs/seed_split_geo/$MODEL/$N_SEEDS
mkdir -p "$LOG_DIR"

echo "Model:   $MODEL"
echo "Seeds:   $FIRST_SEED to $((FIRST_SEED + N_SEEDS - 1))"
echo "Logs:    $LOG_DIR"
echo "================================"

for (( i=0; i<N_SEEDS; i++ )); do
    SEED=$((FIRST_SEED + i))
    echo "Submitting seed=$SEED ..."
    sbatch \
        --export=ALL,SEED=$SEED,MODEL=$MODEL \
        --output="$LOG_DIR/${MODEL}_geo-split-100_seed${SEED}_%j.txt" \
        slurm_encdec_lstm.sh
done
