#!/bin/bash -l

#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10gb
#SBATCH --mail-type=FAIL
#SBATCH --mail-user='badit004@umn.edu'
#SBATCH -p a100-4,kgml03,kgml01
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/encdec_lstm_seed_%j.txt

source /users/2/badit004/anaconda3/etc/profile.d/conda.sh
conda activate hydrodiff

cd /projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning

SEED=${SEED:-3407}
MODEL=${MODEL:-encdec_lstm}
STRIDE=${STRIDE:-1}
# Leave SPLIT_CSV unset for global (all-basin) runs.
# Pass SPLIT_CSV=path/to/csv via sbatch --export to train on a basin subset.
SPLIT_CSV=${SPLIT_CSV:-/projects/standard/kumarv/public/eacvi-fhnn-data/camels_active_learning_splits/camels_split_geo.csv}

SPLIT_ARG=""
if [ -n "$SPLIT_CSV" ]; then
    SPLIT_ARG="--basin_split_csv=$SPLIT_CSV"
fi

python main.py train_npy \
    --model_name=$MODEL \
    --seed=$SEED \
    --gpu=0 \
    --no_static=false \
    --concat_static=true \
    --epochs=30 \
    --stride=$STRIDE \
    $SPLIT_ARG
