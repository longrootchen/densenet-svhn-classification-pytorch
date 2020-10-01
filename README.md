# Re-implementation of ConvNets on SVHN classification task with PyTorch

Contact email: imdchan@yahoo.com

## Introduction

Here are some re-implementations of Convolutional Networks on SVHN dataset (cropped digits) classification.

SVHN dataset consists of 73,257 images of cropped digits (32 x 32) for train, 26,032 images for test and 531,131 extra samples.
The common use is that selecting 400 images per class from the original 'train' set and 200 images per class from the 'extra' set for validation, resulting 6,000 images for validation, with the rest of 509,604 images for train.

## Requirements

- A single TITAN RTX (24G momery) GPU was used.

- Python 3.7+

- PyTorch 1.0+

## Usage

1. Clone this repository

        git clone https://github.com/longrootchen/densenet-svhn-classification-pytorch
    
2. Train a model, taking densenet100_bc_k12 as an example

        python -u train.py --work-dir ./experiments/densenet100_bc_k12 --resume ./experiments/densenet100_bc_k12/checkpoints/last_checkpoint.pth
    
3. Evaluate a model, taking densenet100_bc_k12 as an example

        python -u eval.py --work-dir ./experiments/densenet100_bc_k12 --ckpt-name last_checkpoint.pth
    
## Results

The DenseNet-250-BC model was trained for 40 epochs, achieving 1.82% error rate in the testing set.

| Error Rate (%) | original paper | re-implementation |
| ----- | ----- | ----- |
| DenseNet-100-BC, k=12 | 1.76 [1] |  |
| DenseNet-250-BC, k=24 | 1.74 [1] | 1.82 |

## References

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. Densely Connected Convolutional Networks. In CVPR, 2017.
