#!/bin/bash -l

#SBATCH --time=05:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20gb
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user='badit004@umn.edu'
#SBATCH -p a100-4,kgml03,kgml01
#SBATCH --gres=gpu:1
#SBATCH --output=../logs/encdec_lstm_seed_%j.txt

source /users/2/badit004/anaconda3/etc/profile.d/conda.sh
conda activate hydrodiff

cd /projects/standard/kumarv/badit004/HydroDiffusion_ActiveLearning

SPLIT_CSV=/projects/standard/kumarv/public/eacvi-fhnn-data/camels_split_geo.csv
SEED=${SEED:-3407}
MODEL=${MODEL:-encdec_lstm}

python main.py train_npy \
    --model_name=$MODEL \
    --seed=$SEED \
    --gpu=0 \
    --no_static=false \
    --concat_static=true \
    --epochs=30 \
    --forcing_source=daymet \
    --basin_split_csv=$SPLIT_CSV
