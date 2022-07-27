# MoCoViT: Mobile Convolutional Vision Transformer
### Work in Progress!


<table>
<tr><td colspan="2"><img src="https://d3i71xaburhd42.cloudfront.net/59e648aec20bd6c9db5f2214e9a3db20d2a1ae0f/5-Figure2-1.png" width="600"></td></tr>
<tr><td style="text-align:center">ViT</td><td style="text-align:center">MoCoViT</td></tr>
</table>

## Introduction
This repository is a PyTorch implementation of "MoCoVit: Mobile Convolutional Vision Transformer" by [Ma et al](https://arxiv.org/abs/2205.12635v1), a convolutional/transformer hybrid model designed for mobile applications.

MoCoViT is heavily based off "GhostNet: More Features from Cheap Operations" by [Han et al.](https://arxiv.org/abs/1911.11907) and this implementation utilizes code from the [ghostnet.pytorch](https://github.com/iamhankai/ghostnet.pytorch) repository. Unlike the original [Vision Transformer](https://arxiv.org/abs/2010.11929) (on the left in the figure above), MoCoViT (right) utilizes convolutional layers (using the feature extractor from GhostNet) to generate the patches passed to the transformer blocks. From there, the Mobile Transformer Block utlizes simplifed versions of self-attention and feed forward networks described in the paper and illustrated above. 

In its current state, the output of MoCoViT is predictions on 1000 classes for use with ImageNet.

## Installation
Lorem ipsum

## Training
Lorem ipsum

## Coming Soon
- Better training scripts.
- Model checkpoints.


## Citation
```
@misc{
  url = {https://arxiv.org/abs/2205.12635v1},
  author = {Ma, Hailong and Xia, Xin and Wang, Xing and Xiao, Xuefeng and Li, Jiashi and Zheng, Min},
  title = {MoCoViT: Mobile Convolutional Vision Transformer},
  year = {2022}
}
```