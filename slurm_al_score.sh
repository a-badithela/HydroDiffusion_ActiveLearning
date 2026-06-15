#!/bin/bash -l
#
# Active-learning scoring / acquisition job.
# Runs inference on sample+eval basins, scores them, promotes top-k to seed,
# and writes the updated split CSV for the next round.
#
# Environment variables (set via sbatch --export=ALL,...):
#   AL_ROUND     — current round index (0-based)                  (required)
#   CKPT_PATTERN — glob pattern for seed run dirs (basename only) (required for ensemble/repr)
#   RUNS_DIR     — directory containing all runs                  (required for ensemble/repr)
#   SPLIT_CSV    — current round's split CSV                      (required)
#   OUT_CSV      — next round's split CSV output path             (required)
#   ACQ_FN       — random | ensemble | representativeness         (default: random)
#   K            — basins to promote per round                    (default: 10)
#   DEVICE       — pytorch device                                 (default: cpu)
#   STRIDE       — inference stride in days                       (default: 90)
#   ALPHA        — seed-sim penalty (representativeness)          (default: 1.0)

#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --mem=10gb
#SBATCH --mail-type=FAIL
#SBATCH --mail-user='badit004@umn.edu'
#SBATCH -p a100-4
#SBATCH --gres=gpu:1

source /users/2/badit004/anaconda3/etc/profile.d/conda.sh
conda activate hydrodiff

cd /projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning

AL_ROUND=${AL_ROUND:-0}
ACQ_FN=${ACQ_FN:-random}
K=${K:-10}
DEVICE=${DEVICE:-cuda:0}
STRIDE=${STRIDE:-180}
ALPHA=${ALPHA:-1.0}

if [ -z "$SPLIT_CSV" ]; then
    echo "ERROR: SPLIT_CSV is not set." >&2; exit 1
fi
if [ -z "$OUT_CSV" ]; then
    echo "ERROR: OUT_CSV is not set." >&2; exit 1
fi

echo "=== AL Scoring Job ==="
echo "  AL_ROUND:     $AL_ROUND"
echo "  ACQ_FN:       $ACQ_FN"
echo "  K:            $K"
echo "  CKPT_PATTERN: ${CKPT_PATTERN:-'(not set — random acquisition)'}"
echo "  RUNS_DIR:     ${RUNS_DIR:-'(not set)'}"
echo "  SPLIT_CSV:    $SPLIT_CSV"
echo "  OUT_CSV:      $OUT_CSV"
echo "  DEVICE:       $DEVICE"
echo "======================="

# Build optional args
CKPT_ARG=""
if [ -n "$CKPT_PATTERN" ] && [ -n "$RUNS_DIR" ]; then
    CKPT_ARG="--ckpt_pattern=${RUNS_DIR}/${CKPT_PATTERN}"
fi

python -m active_learning.run_acquisition \
    --split_csv="$SPLIT_CSV" \
    --out_csv="$OUT_CSV" \
    --acquisition="$ACQ_FN" \
    --k="$K" \
    --device="$DEVICE" \
    --stride="$STRIDE" \
    --alpha="$ALPHA" \
    $CKPT_ARG

EXIT_CODE=$?
echo "[slurm_al_score] run_acquisition exited with code $EXIT_CODE"
exit $EXIT_CODE
