# MoCoViT: Mobile Convolutional Vision Transformer

<table>
<tr><td colspan="2"><img src="./figures/figure2.png" width="750"></td></tr>
<tr><td><b>ViT</b></td><td><b>MoCoViT</b></td></tr>
</table>

## Introduction
This repository is a PyTorch implementation of "MoCoVit: Mobile Convolutional Vision Transformer" by [Ma et al](https://arxiv.org/abs/2205.12635v1), a convolutional/transformer hybrid model designed for mobile applications.

MoCoViT is heavily based off "GhostNet: More Features from Cheap Operations" by [Han et al.](https://arxiv.org/abs/1911.11907) and this implementation utilizes code from the [ghostnet.pytorch](https://github.com/iamhankai/ghostnet.pytorch) repository. Unlike the original [Vision Transformer](https://arxiv.org/abs/2010.11929) (on the left in the figure above), MoCoViT (right) utilizes convolutional layers (using the feature extractor from GhostNet) to generate the patches passed to the transformer blocks. From there, the Mobile Transformer Block utlizes simplifed versions of self-attention and feed forward networks described in the paper and illustrated above. 

In its current state, the output of MoCoViT is predictions on 1000 classes for use with ImageNet.

## Installation
This repo uses [ghostnet.pytorch](https://github.com/iamhankai/ghostnet.pytorch) as a git submodule, so cloning must be done recursively:
```
git clone --recursive https://github.com/smitheric95/MoCoViT-PyTorch.git
```

### Requirements
```
PyTorch 1.0+
Torchvision
Argparse
Pillow
```

## Usage
This repo comes with training and testing scripts for use witht he ImageNet-1k dataset. Running these scripts requires dowloading the [ILSVRC2012](https://image-net.org/challenges/LSVRC/2012/2012-downloads.php) image sets. Instructions for running both scripts are below.

### Training
```
usage: train.py [-h] [--imagenet_path IMAGENET_PATH] [--gpu GPU] [--epochs EPOCHS] [--validate {True,False}]
                [--train_batch TRAIN_BATCH] [--val_batch VAL_BATCH] [--num_workers NUM_WORKERS]

Train a MoCoViT model on an ImageNet-1k dataset.

optional arguments:
  -h, --help            show this help message and exit
  --imagenet_path IMAGENET_PATH
                        Path to ImageNet-1k directory containing 'train' and 'val' folders. Default './imagenet'.
  --gpu GPU             GPU to use for training. Default 0.
  --epochs EPOCHS       Number of epochs for which to train. Default 20.
  --validate {True,False}
                        If True, run validation after each epoch. Default True.
  --train_batch TRAIN_BATCH
                        Batch size to use for training. Default 128.
  --val_batch VAL_BATCH
                        Batch size to use for validation. Default 1024.
  --num_workers NUM_WORKERS
                        Number of workers to use while loading dataset splits. Default 4.
```

### Testing
```
usage: test.py [-h] [--imagenet_path IMAGENET_PATH] [--gpu GPU] [--epoch EPOCH] [--num_workers NUM_WORKERS]
               [--verbose {True,False}]

Test a MoCoViT model against an ImageNet-1k dataset.

optional arguments:
  -h, --help            show this help message and exit
  --imagenet_path IMAGENET_PATH
                        Path to ImageNet-1k directory containing 'test' folder. Default './imagenet'.
  --gpu GPU             GPU to use for testing. Default 0.
  --epoch EPOCH         Epoch of model to use for testing. Default './checkpoints/default.pt'
  --num_workers NUM_WORKERS
                        Number of workers to use while loading dataset splits. Default 4.
  --verbose {True,False}
                        If True, print prediction information. Default False.
```

## Citation
```
@misc{
  url = {https://arxiv.org/abs/2205.12635v1},
  author = {Ma, Hailong and Xia, Xin and Wang, Xing and Xiao, Xuefeng and Li, Jiashi and Zheng, Min},
  title = {MoCoViT: Mobile Convolutional Vision Transformer},
  year = {2022}
}
```