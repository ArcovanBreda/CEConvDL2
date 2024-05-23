# A deep dive into colorspace equivariant networks
This repository contains a reproduction and extension on CEConv - Color Equivariant Convolutional Networks [[ArXiv](https://arxiv.org/abs/2310.19368)] - NeurIPS 2023, by Attila Lengyel, Ombretta Strafforello, Robert-Jan Bruintjes, Alexander Gielisse, and Jan van Gemert. This readme contains all commands in order to run the performed experiments, for an explanation about the project we refer you to our extensive [blogpost](./blogpost.md). We sincerely thank the original authors for providing the original [code](https://github.com/Attila94/CEConv).

## Prerequisites
* A machine running Linux / WSL on windows also works
* [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
* A CUDA enabled GPU with at least 10GB of VRAM, classification experiments may require 12GB or more (decreasing the batch size may help at the cost of training time).

## Setup

Create a local clone of this repository:
```bash
git clone https://github.com/ArcovanBreda/CEConvDL2.git
```
Create and activate a conda environment:
```bash
conda create -n CEConv python=3.10
conda activate CEConv 
```
See `requirements.txt` for the required Python packages. You can install them using:
```bash
pip install -r requirements.txt
```

Install CEConv:
```bash
python setup.py install
```

Set the required environment variables:
```bash
export WANDB_DIR=path_to_wandb_logs  # Store wandb logs here.
export DATA_DIR=path_to_datasets  # Store datasets here.
export OUT_DIR=path_to_model_checkpoints  # Store models here.
```

## How to use

CEConv can be used in the same way as a regular Conv2d layer. The following code snippet shows how to use CEConv in a CNN architecture:

```python
import torch
import torch.nn as nn
from ceconv import CEConv2d

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()

        # Args: input rotations, output rotations, input channels, output channels, kernel size, padding.
        self.conv1 = CEConv2d(1, 3, 3, 32, 3,
        lab_space = False, # indicates input image is in LAB color space
        hsv_space = False, # indicates input image is in HSV color space
        img_shift = False, # indicates group action is applied on image instead of kernel
        sat_shift = False, # indicates saturation equivariance in HSV space
        hue_shift = False, # indicates hue equivariance in HSV space
        val_shift = False, # indicates value equivariance in HSV space
        padding=1)
        self.conv2 = CEConv2d(3, 3, 32, 64, 3,
        lab_space = False,
        hsv_space = False,
        img_shift = False,
        sat_shift = False,
        hue_shift = False,
        val_shift = False,
        padding=1)
        
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # Average pooling over spatial and color dimensions.
        x = torch.mean(x, dim=(2, 3, 4))
        
        x = self.fc(x)
        return x
```

## Experiments
The experiments from our blog post can be reproduced by running the following commands from the CEConv folder:
```
bash
cd CEConv
```
### Reproduction
#### ColorMNIST
**Generate ColorMNIST datasets**
```bash
python -m experiments.color_mnist.colormnist_longtailed
```

**Longtailed ColorMNIST**
```bash
# Baseline:
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 1
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 2
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 3
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 4
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 5
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 6
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 7
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20 --seed 8
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 9
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 20  --seed 10
# Color equivariant CNN:
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 1
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 2
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 3
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 4
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 5
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 6
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 7
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 8
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 9
python -m experiments.color_mnist.train_longtailed --rotations 3 --planes 17 --separable --seed 10
```

<!-- **Biased ColorMNIST**
```bash
# Baseline and grayscale.
python -m experiments.color_mnist.train_biased --std $2 --rotations 1 --planes 20 
python -m experiments.color_mnist.train_biased --std $2 --rotations 1 --planes 20 --grayscale

# Color equivariant with and without group coset pooling.
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 17 --separable
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 17 --separable --groupcosetpool

# Hybrid Color Equivariant architectures.
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 19 --ce_layers 2 --separable --groupcosetpool
python -m experiments.color_mnist.train_biased --std $2 --rotations 3 --planes 18 --ce_layers 4 --separable --groupcosetpool
``` -->

#### Classification performance with test-time hue shifts on Flowers dataset:
```bash
# Baseline:
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18
# Baseline + jitter:
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --jitter 0.5
# CEConv:
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable
# CEConv + jitter:
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --jitter 0.5
```

#### Color Selectivity
```bash
# flowers102 dataset without jitter
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset flowers102 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 1 --architecture resnet18 --dataset flowers102 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 2 --architecture resnet18 --dataset flowers102 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 3 --architecture resnet18 --dataset flowers102 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 4 --architecture resnet18 --dataset flowers102 --bs 64

# flowers102 dataset with jitter
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset flowers102 --bs 64 --jitter 0.5
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 1 --architecture resnet18 --dataset flowers102 --jitter 0.5 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 2 --architecture resnet18 --dataset flowers102 --jitter 0.5 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 3 --architecture resnet18 --dataset flowers102 --jitter 0.5 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 4 --architecture resnet18 --dataset flowers102 --jitter 0.5 --bs 64

# stl10 dataset without jitter
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset stl10 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 1 --architecture resnet18 --dataset stl10 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 2 --architecture resnet18 --dataset stl10 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 3 --architecture resnet18 --dataset stl10 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 4 --architecture resnet18 --dataset stl10 --bs 64

# stl10 with jitter
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset stl10 --bs 64 --jitter 0.5
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 1 --architecture resnet18 --dataset stl10 --jitter 0.5 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 2 --architecture resnet18 --dataset stl10 --jitter 0.5 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 3 --architecture resnet18 --dataset stl10 --jitter 0.5 --bs 64
python -m experiments.classification.train --rotations 3 --groupcosetmaxpool --separable --ce_stages 4 --architecture resnet18 --dataset stl10 --jitter 0.5 --bs 64
```

#### Rotations
```bash
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset flowers102 --bs 64 --separable --groupcosetmaxpool
python -m experiments.classification.train --rotations 5 --architecture resnet18 --dataset flowers102 --bs 64 --separable --groupcosetmaxpool
python -m experiments.classification.train --rotations 10 --architecture resnet18 --dataset flowers102 --bs 64 --separable --groupcosetmaxpool
```

#### Jitter
```bash
# without equivariance
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset flowers102 --bs 64 --jitter 0.2
python -m experiments.classification.train --rotations 1 --architecture resnet18 --dataset flowers102 --bs 64 --jitter 0.4

# with equivariance
python -m experiments.classification.train --rotations 3 --architecture resnet18 --dataset flowers102 --bs 64 --jitter 0.1 --groupcosetmaxpool --separable
python -m experiments.classification.train --rotations 3 --architecture resnet18 --dataset flowers102 --bs 64 --jitter 0.2 --groupcosetmaxpool --separable
```

### Extension
#### HSV
**Hue**

```bash
# Train + evaluation of a hue shifted image - 3 rotations
# Baseline
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --nonorm
# Baseline + jitter
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --jitter 0.5 --nonorm
# Hue equivariant network
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --nonorm
# Hue equivariant network + jitter
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --img_shift --jitter 0.5 --nonorm

# Train + evaluation of a hue shifted kernel - 3 rotations
# Baseline
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --nonorm
# Baseline + jitter
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --jitter 0.5 --nonorm
# Hue equivariant network
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --nonorm
# Hue equivariant network + jitter
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --hue_shift --jitter 0.5 --nonorm
```

**Saturation**
```bash
# Train + evaluation of a saturation shifted image - 5 shifts
# Baseline
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm --img_shift
# Baseline + jitter
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm --img_shift
# Saturation equivariant network
python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm --img_shift
# Saturation equivariant network + jitter
python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm --img_shift

# Train + evaluation of a saturation shifted kernel - 5 shifts
# Baseline
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm
# Baseline + jitter
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm
# Saturation equivariant network
python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hsv_test --nonorm
# Saturation equivariant network + jitter
python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --sat_jitter 0 100 --separable --hsv --sat_shift --hsv_test --nonorm
```
**Value**
```bash
# Baseline
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test
# Baseline + jitter
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test --value_jitter 0 100

# Value equivariance
python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test
# Value equivariance + jitter
python -m experiments.classification.train --rotations 5 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --img_shift --value_shift --epochs 200 --nonorm --hsv_test --value_jitter 0 100
```
<!-- **Hue and Saturation**
```bash
# Image shift, 3 rotations for hue and 3 shifts for saturation
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hue_shift --hsv_test --nonorm
# Kernel shift, 3 rotations for hue and 3 shifts for saturation
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --hsv --sat_shift --hue_shift --hsv_test --img_shift --nonorm
``` -->
#### LAB 
**Hue**
```bash
# Baseline
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm
# Baseline + jitter
python -m experiments.classification.train --rotations 1 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm --jitter 0.5

# Hue lab space equivariance
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm
# Hue lab space equivariance + jitter
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm --jitter 0.5

# Hue lab space equivariance + test images hue shifted in lab space
python -m experiments.classification.train --rotations 3 --dataset flowers102 --bs 64 --epoch 200 --architecture resnet18 --groupcosetmaxpool --separable --lab --epochs 200 --nonorm --lab_test
```

#### Longtailed ColorMNIST - Time
```bash
# Baseline:
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 1
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 2
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 3
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 4
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 5
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70 --seed 6
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 7
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 8
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 9
python -m experiments.color_mnist.train_longtailed --rotations 1 --planes 70  --seed 10
```
