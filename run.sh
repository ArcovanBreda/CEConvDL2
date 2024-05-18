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
cd $HOME/CEConvDL2/CEConv

# set env vars
export DATA_DIR=./DATA
export WANDB_DIR=$HOME/CEConvDL2/CEConv/WANDB
export OUT_DIR=./output
export WANDB_API_KEY=$YOUR_API_KEY
export WANDB_NAME=$RUN_NAME_ON_WANDB

# ########  Extension  ######## 
# ##### HSV
# #### Hue
# # Train + evaluation of a hue shifted image - 3 rotations
# # Baseline
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --nonorm
# # Baseline + jitter
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --jitter 0.5 --nonorm
# # Hue equivariant network
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --nonorm
# # Hue equivariant network + jitter
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --jitter 0.5 --nonorm

# # Train + evaluation of a hue shifted kernel - 3 rotations
# # Baseline
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --nonorm
# # Baseline + jitter
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --jitter 0.5 --nonorm
# # Hue equivariant network
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --nonorm
# # Hue equivariant network + jitter
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --jitter 0.5 --nonorm


# #### Saturation
# # Train + evaluation of a saturation shifted image - 5 shifts
# # Baseline
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm --img_shift
# # Baseline + jitter
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm --img_shift
# # Saturation equivariant network
# python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm --img_shift
# # Saturation equivariant network + jitter
# python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm --img_shift

# # Train + evaluation of a saturation shifted kernel - 5 shifts
# # Baseline
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm
# # Baseline + jitter
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm
# # Saturation equivariant network
# python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm
# # Saturation equivariant network + jitter
# python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm


# #### Value
# # Baseline
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test
# # Baseline + jitter
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test --value_jitter 0 100
# # Value equivariance
# python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test
# # Value equivariance + jitter
# python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test --value_jitter 0 100


# ##### LAB 
# #### Hue
# # Baseline
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm
# # Baseline + jitter
# python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm --jitter 0.5
# # Hue lab space equivariance
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm
# # Hue lab space equivariance + jitter
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm --jitter 0.5
# # Hue lab space equivariance + test images hue shifted in lab space
# python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm --lab_test
