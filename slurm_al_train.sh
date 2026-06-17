#!/bin/bash -l
#
# Active-learning training job.
# Trains ONE model seed on the current seed basins and saves to a predictable path.
#
# Environment variables (set via sbatch --export=ALL,...):
#   SEED         — random seed              (required)
#   MODEL        — model name               (default: encdec_lstm)
#   SPLIT_CSV    — path to AL split CSV     (required)
#   RUN_DIR      — where to save checkpoint (required)
#   EPOCHS       — training epochs          (default: 30)
#   TRAIN_STRIDE — dataset window stride    (default: 1; use e.g. 90 for fast testing)
#   AL_ROUND     — round index (metadata)   (default: 0)
#   ACQ_FN       — acquisition fn (metadata)(default: random)

#SBATCH --time=01:30:00
#SBATCH --ntasks=1
#SBATCH --mem=5gb
#SBATCH --mail-type=FAIL
#SBATCH --mail-user='badit004@umn.edu'
#SBATCH -p a100-4
#SBATCH --gres=gpu:1

source /users/2/badit004/anaconda3/etc/profile.d/conda.sh
conda activate hydrodiff

cd /projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning

SEED=${SEED:-3407}
MODEL=${MODEL:-encdec_lstm}
EPOCHS=${EPOCHS:-30}
TRAIN_STRIDE=${TRAIN_STRIDE:-1}
AL_ROUND=${AL_ROUND:-0}
ACQ_FN=${ACQ_FN:-random}

# Both SPLIT_CSV and RUN_DIR must be provided for AL runs.
if [ -z "$SPLIT_CSV" ]; then
    echo "ERROR: SPLIT_CSV is not set." >&2; exit 1
fi
if [ -z "$RUN_DIR" ]; then
    echo "ERROR: RUN_DIR is not set." >&2; exit 1
fi

mkdir -p "$RUN_DIR"

echo "=== AL Training Job ==="
echo "  SEED:         $SEED"
echo "  MODEL:        $MODEL"
echo "  SPLIT_CSV:    $SPLIT_CSV"
echo "  RUN_DIR:      $RUN_DIR"
echo "  EPOCHS:       $EPOCHS"
echo "  TRAIN_STRIDE: $TRAIN_STRIDE"
echo "  AL_ROUND:     $AL_ROUND"
echo "  ACQ_FN:       $ACQ_FN"
echo "========================"

if [ "$MODEL" = "decoder_only_ssm" ]; then
    # Paper hyperparameters from train.sh — must be passed explicitly because
    # main.py defaults for lr_min (0.001), ssm_dropout (0.3), and wd (0.02)
    # all diverge from the published values.
    python main.py train_npy \
        --model_name="$MODEL" \
        --seed="$SEED" \
        --gpu=0 \
        --no_static=false \
        --concat_static=true \
        --epochs="$EPOCHS" \
        --stride="$TRAIN_STRIDE" \
        --run_dir="$RUN_DIR" \
        --basin_split_csv="$SPLIT_CSV" \
        --al_round="$AL_ROUND" \
        --al_acq_fn="$ACQ_FN" \
        --d_model=256 \
        --d_state=256 \
        --n_layers=6 \
        --lr=3e-5 \
        --lr_min=3e-6 \
        --weight_decay=0.00 \
        --wd=4e-5 \
        --lr_dt=0.001 \
        --min_dt=0.01 \
        --max_dt=0.1 \
        --warmup=1 \
        --ssm_dropout=0.2 \
        --cfi=10 \
        --cfr=10 \
        --forcing_source=daymet
else
    python main.py train_npy \
        --model_name="$MODEL" \
        --seed="$SEED" \
        --gpu=0 \
        --no_static=false \
        --concat_static=true \
        --epochs="$EPOCHS" \
        --stride="$TRAIN_STRIDE" \
        --run_dir="$RUN_DIR" \
        --basin_split_csv="$SPLIT_CSV" \
        --al_round="$AL_ROUND" \
        --al_acq_fn="$ACQ_FN"
fi
