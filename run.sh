#!/bin/bash

#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --job-name=Exp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=18
#SBATCH --time=00:15:00
#SBATCH --output=run_model_data_%A.out

module purge
module load 2022
module load Anaconda3/2022.05

source ~/.bashrc
conda activate CEConv

# change working dir
cd $HOME/CEConvDL2/CEConv

# set env vars and print them
export DATA_DIR=./data
export WANDB_DIR=$HOME/.conda/envs/CEConv/lib/python3.12/site-packages
export OUT_DIR=./output

# ResNet44 (or optionally 18)
# Baseline, grayscale and color jitter.
python -m experiments.classification.train --rotations 1
# python -m experiments.classification.train --rotations 1 --grayscale
# python -m experiments.classification.train --rotations 1 --jitter 0.5

# # Color Equivariant with and without group coset pooling and color jitter.
# python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable
# python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --jitter 0.5

# # Hybrid Color Equivariant architectures.
# python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 1 --width 31
# python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 2 --width 30

# ImageNet
# Baseline.
# python -m experiments.imagenet.main --rotations 1 --jitter 0.0 --arch 'resnet18'

# # Color Equivariant.
# python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --groupcosetmaxpool --separable

# # Hybrid Color Equivariant architectures.
# python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --network_width 63 --run_name 'hybrid_1' --groupcosetmaxpool --separable --ce_stages 1
# python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --network_width 63 --run_name 'hybrid_2' --groupcosetmaxpool --separable --ce_stages 2
# python -m experiments.imagenet.main --rotations 3 --batch-size 256 --jitter 0.0 --workers 4 --arch 'resnet18' --network_width 61 --run_name 'hybrid_3' --groupcosetmaxpool --separable --ce_stages 3
