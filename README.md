# Path-InfoMax
This repository holds the code used in our **IJCAI21** paper [Unsupervised Path Representation Learning with Curriculum Negative Sampling] (S. Yang, J. Hu, C. Guo, J. Tang and B. Yang, IJCAI2021):[]()

![image](https://github.com/Sean-Bin-Yang/Path-InfoMax/blob/502ea3a57a578b325b704f50eb9808ee5968b744/Fig1.png)

## Overview
Note that we propose two variants of GMI in the paper, the one is GMI-mean, and the other is GMI-adaptive. Since GMI-mean often outperforms GMI-adaptive (see the experiments in the paper), here we give a PyTorch implementation of GMI-mean. To make GMI more practical, we provide an alternative solution to compute FMI. Such a solution still ensures the effectiveness of GMI and improves the efficiency greatly. The repository is organized as follows:

- `data/` includes three benchmark datasets;
- `models/` contains the implementation of the GMI pipeline (`gmi.py`) and the logistic regression classifier (`logreg.py`);
- `layers/` contains the implementation of a standard GCN layer (`gcn.py`), the bilinear discriminator (`discriminator.py`), and the mean-pooling operator (`avgneighbor.py`);
- `utils/` contains the necessary processing tool (`process.py`).

To better understand the code, we recommend that you could read the code of DGI/Petar (https://arxiv.org/abs/1809.10341) in advance. Besides, you could further optimize the code based on your own needs. We display it in an easy-to-read form.

## Requirements

  * Ubuntu OS (16.04)
  * PyTorch >=1.2.0
  * Numpy >= 1.16.2
  * Pickle

## Usage

```python train.py```

## Cite
Please cite our paper if you make advantage of PIM in your research:

```
@inproceedings{
IJCAI21,
title="{Unsupervised Path Representation Learning with Curriculum Negative Sampling}",
author={Sean Bin Yang, Jilin Hu, Chenjuan Guo, Jian Tang and Bin Yang},
booktitle={Proceedings of The  30th International Joint Conference on Artificial Intelligence},
year={2021},
}
