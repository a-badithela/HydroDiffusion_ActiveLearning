#!/bin/bash -l
#
# Active-learning orchestration script.
#
# Submits ALL rounds upfront using SLURM dependencies so the cluster handles
# the sequencing automatically:
#
#   round 0: train seed1 ─┐
#            train seed2 ─┼─→ score_r0 ─→ round 1: train seed1 ─┐
#            ...          ─┘                       train seed2 ─┼─→ score_r1 ─→ ...
#            train seedN                           ...          ─┘
#
# When to stop: if the sample pool empties, run_acquisition exits with code 2
# and the scoring job fails cleanly, cancelling any downstream jobs.
#
# Usage
# -----
#   bash schedule_al.sh [options]
#
# Options (all have defaults):
#   --split_csv    PATH   Initial AL split CSV
#                         (default: /projects/.../camels_split_geo_100_100.csv)
#   --acq_fn       NAME   random | ensemble | representativeness  (default: random)
#   --n_rounds     INT    Number of AL rounds to run             (default: 5)
#   --k            INT    Basins to promote per round            (default: 10)
#   --n_seeds      INT    Ensemble seeds per round               (default: 5)
#   --epochs       INT    Training epochs per seed               (default: 30)
#   --run_root     PATH   Root dir for logs and split CSVs       (default: runs/al_orchestration)
#   --runs_dir     PATH   Root dir for checkpoint runs           (default: <project>/runs)
#   --device       STR    Device for scoring (default: cuda:0)
#   --stride       INT    Inference stride in days (default: 90)
#   --alpha        FLOAT  Repr. alpha parameter (default: 1.0)
#   --model        NAME   Model architecture (default: encdec_lstm)
#   --train_stride INT    Training dataset stride (default: 1; use 90 for fast testing)
#
# Example
# -------
#   bash schedule_al.sh \
#       --split_csv /projects/.../camels_split_geo_100_100.csv \
#       --acq_fn ensemble \
#       --n_rounds 5 \
#       --k 10 \
#       --n_seeds 5

# ---------- defaults ----------
PROJ_DIR=/projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning
SPLIT_CSV_INIT=/projects/standard/kumarv/public/eacvi-fhnn-data/camels_active_learning_splits/camels_split_geo.csv
ACQ_FN=random
N_ROUNDS=5
K=10
N_SEEDS=5
EPOCHS=30
RUN_ROOT="$PROJ_DIR/runs/al_orchestration"
RUNS_DIR="$PROJ_DIR/runs"
DEVICE=cuda:0
STRIDE=1
ALPHA=1.0
MODEL=decoder_only_ssm
TRAIN_STRIDE=1
FIRST_SEED=3407

# Capture date ONCE so all seeds in this experiment share the same date tag.
RUN_DATE=$(date +%Y%m%d)

# ---------- parse args ----------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --split_csv)    SPLIT_CSV_INIT="$2"; shift 2 ;;
        --acq_fn)       ACQ_FN="$2";         shift 2 ;;
        --n_rounds)     N_ROUNDS="$2";       shift 2 ;;
        --k)            K="$2";              shift 2 ;;
        --n_seeds)      N_SEEDS="$2";        shift 2 ;;
        --epochs)       EPOCHS="$2";         shift 2 ;;
        --run_root)     RUN_ROOT="$2";       shift 2 ;;
        --runs_dir)     RUNS_DIR="$2";       shift 2 ;;
        --device)       DEVICE="$2";         shift 2 ;;
        --stride)       STRIDE="$2";         shift 2 ;;
        --alpha)        ALPHA="$2";          shift 2 ;;
        --model)        MODEL="$2";          shift 2 ;;
        --train_stride) TRAIN_STRIDE="$2";   shift 2 ;;
        --train_time)   TRAIN_TIME="$2";     shift 2 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

# decoder_only_ssm paper default is 60 epochs; auto-upgrade only when the user
# left EPOCHS at the generic default of 30 (i.e. didn't pass --epochs explicitly).
if [ "$MODEL" = "decoder_only_ssm" ] && [ "$EPOCHS" = "30" ]; then
    EPOCHS=60
fi

# SLURM wall-time for training jobs; SSM with 60 epochs needs ~2-3× more time.
if [ "$MODEL" = "decoder_only_ssm" ]; then
    TRAIN_TIME=${TRAIN_TIME:-03:00:00}
else
    TRAIN_TIME=${TRAIN_TIME:-01:30:00}
fi

# ---------- derived tags ----------
# model tag: underscores → dashes  (e.g. encdec_lstm → encdec-lstm)
MODEL_TAG="${MODEL//_/-}"

# split tag from CSV stem: camels_split_geo_100_100* → geo-100; camels_split_geo* → geo; else stem
_stem=$(basename "$SPLIT_CSV_INIT" .csv)
if [[ "$_stem" =~ camels_split_(geo|hyd)_([0-9]+) ]]; then
    SPLIT_TAG="${BASH_REMATCH[1]}-${BASH_REMATCH[2]}"
elif [[ "$_stem" =~ camels_split_(geo|hyd) ]]; then
    SPLIT_TAG="${BASH_REMATCH[1]}"
else
    SPLIT_TAG="$_stem"
fi

mkdir -p "$RUN_ROOT" "$RUNS_DIR"

echo "==========================================="
echo "  Active Learning Schedule"
echo "  RUN_DATE    : $RUN_DATE"
echo "  split_csv   : $SPLIT_CSV_INIT"
echo "  split_tag   : $SPLIT_TAG"
echo "  acq_fn      : $ACQ_FN"
echo "  n_rounds    : $N_ROUNDS"
echo "  k/round     : $K"
echo "  n_seeds     : $N_SEEDS"
echo "  epochs      : $EPOCHS"
echo "  runs_dir    : $RUNS_DIR"
echo "  model       : $MODEL  ($MODEL_TAG)"
echo "  train_stride: $TRAIN_STRIDE"
echo "  train_time  : $TRAIN_TIME"
echo "==========================================="

# ---------------------------------------------------------------------------
# Helper: submit one training job, print the job ID.
# ---------------------------------------------------------------------------
submit_train() {
    local round=$1 seed=$2 split_csv=$3 dep_arg=$4
    local run_dir="${RUNS_DIR}/${RUN_DATE}/al-r${round}-${ACQ_FN}_${MODEL_TAG}_${SPLIT_TAG}_s${TRAIN_STRIDE}_seed${seed}"
    local log_file="$RUN_ROOT/round${round}/logs/train_seed${seed}_%j.txt"
    mkdir -p "$RUN_ROOT/round${round}/logs"

    local dep_flag=""
    if [ -n "$dep_arg" ]; then
        dep_flag="--dependency=afterok:${dep_arg}"
    fi

    sbatch --parsable \
        --time=$TRAIN_TIME \
        $dep_flag \
        --export=ALL,SEED=$seed,MODEL=$MODEL,SPLIT_CSV=$split_csv,RUN_DIR=$run_dir,EPOCHS=$EPOCHS,TRAIN_STRIDE=$TRAIN_STRIDE,AL_ROUND=$round,ACQ_FN=$ACQ_FN \
        --output="$log_file" \
        slurm_al_train.sh
}

# ---------------------------------------------------------------------------
# Helper: submit the scoring job for a round, depending on all training jobs.
# Returns the scoring job ID.
# ---------------------------------------------------------------------------
submit_score() {
    local round=$1 dep_ids=$2 split_csv=$3 out_csv=$4
    local ckpt_pattern="${RUN_DATE}/al-r${round}-${ACQ_FN}_${MODEL_TAG}_${SPLIT_TAG}_s${TRAIN_STRIDE}_seed*"
    local log_file="$RUN_ROOT/round${round}/logs/score_%j.txt"

    sbatch --parsable \
        --dependency=afterok:$dep_ids \
        --export=ALL,AL_ROUND=$round,CKPT_PATTERN=$ckpt_pattern,RUNS_DIR=$RUNS_DIR,SPLIT_CSV=$split_csv,OUT_CSV=$out_csv,ACQ_FN=$ACQ_FN,K=$K,DEVICE=$DEVICE,STRIDE=$STRIDE,ALPHA=$ALPHA \
        --output="$log_file" \
        slurm_al_score.sh
}

# ---------------------------------------------------------------------------
# Main submission loop
# ---------------------------------------------------------------------------
prev_score_jid=""
current_split="$SPLIT_CSV_INIT"

for (( r=0; r<N_ROUNDS; r++ )); do
    echo ""
    echo "--- Submitting round $r ---"

    # Auditable snapshot of the split for this round.
    round_split="$RUN_ROOT/round${r}/split.csv"
    mkdir -p "$RUN_ROOT/round${r}/logs"

    if [ $r -eq 0 ]; then
        cp "$SPLIT_CSV_INIT" "$round_split"
    fi
    # For r>0, the scoring job of round r-1 writes round_split directly.

    # Submit N training seeds for this round.
    train_jids=""
    for (( i=0; i<N_SEEDS; i++ )); do
        seed=$(( FIRST_SEED + i ))
        jid=$(submit_train $r $seed "$round_split" "$prev_score_jid")
        echo "  train seed=$seed → job $jid"
        train_jids="${train_jids}${jid}:"
    done

    # Remove trailing colon for --dependency format.
    train_dep="${train_jids%:}"

    # Submit scoring job, depending on ALL training seeds finishing.
    next_split="$RUN_ROOT/round$(( r+1 ))/split.csv"
    mkdir -p "$RUN_ROOT/round$(( r+1 ))/logs"

    score_jid=$(submit_score $r "$train_dep" "$round_split" "$next_split")
    echo "  score round=$r  → job $score_jid  (depends on: $train_dep)"

    prev_score_jid=$score_jid
done

echo ""
echo "==========================================="
echo "  All jobs submitted."
echo "  Logs/splits : $RUN_ROOT/round{0..$(( N_ROUNDS-1 ))}/"
echo "  Run dirs    : $RUNS_DIR/${RUN_DATE}_al-r*"
echo "  Use 'squeue -u \$USER' to monitor progress."
echo "==========================================="
