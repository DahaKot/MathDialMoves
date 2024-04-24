#!/bin/bash

#SBATCH --job-name=prepare_mathdialmoves_data # Job name
#SBATCH --error=/home/daria.kotova/nlp804/MathDialMoves/logs/%j%x.err # error file
#SBATCH --output=/home/daria.kotova/nlp804/MathDialMoves/logs/%j%x.out # output log file
#SBATCH --time=24:00:00 # 10 hours of wall time
#SBATCH --nodes=1  # 1 GPU node
#SBATCH --mem=45000 # 32 GB of RAM
#SBATCH --nodelist=ws-l6-013


echo "Start."

WINDOW_SIZE=3

python data_preparation.py \
--data_path="./mathdial/data/train.csv" \
--save_file="./data/train_replics.csv" \
--window_size=$WINDOW_SIZE

python data_preparation.py \
--data_path="./mathdial/data/test.csv" \
--save_file="./data/test_replics.csv" \
--window_size=$WINDOW_SIZE


echo "End."