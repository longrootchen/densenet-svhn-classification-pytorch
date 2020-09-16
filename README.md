# Introduction

Here is a re-implementation of DenseNet-250-BC (k=24) on classification task for SVHN dataset (cropped digits) with PyTorch.

SVHN dataset consists of 73,257 images of cropped digits (32 x 32) for train, 26,032 images for test and 531,131 extra samples.
The common use is that selecting 400 images per class from the original 'train' set and 200 images per class from the 'extra' set for validation, resulting 6,000 images for validation, with the rest of 509,604 images for train.

Contact email: imdchan@yahoo.com

# Requirements

A single TITAN RTX (24G momery) GPU was used.

# Usage

    git clone https://github.com/longrootchen/densenet-svhn-classification-pytorch
    
# Train

    python train.py
    
# Evaluation

    python eval.py
    
# Results

The DenseNet-250-BC model was trained for 40 epochs, achieving 1.82% error rate in the testing set.

|  | error (%) |
| ----- | ----- |
| original paper | 1.74 |
| re-implementation | 1.82 |

Here is the confusion matrix for the test set.

![Confusion matrix](https://github.com/longrootchen/densenet-svhn-classification-pytorch/blob/master/images/confusion_matrix.jpg)

# References

[1] Gao Huang, Zhuang Liu, Laurens van der Maaten, Kilian Q. Weinberger. Densely Connected Convolutional Networks. In CVPR, 2017.
