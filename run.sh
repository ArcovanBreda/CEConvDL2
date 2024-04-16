#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Exp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=run_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source ~/.bashrc
conda activate CEConv

# change working dir
cd $HOME/CEConv

# set env vars and print them
export DATA_DIR=./
export WANDB_DIR=$HOME/.conda/envs/CEConv/lib/python3.12/site-packages
export OUT_DIR=./output
env | grep DATA_DIR
env | grep WANDB_DIR
env | grep OUT_DIR

python -m experiments.classification.train --rotations 1