# Path-InfoMax
This repository holds the code used in our **IJCAI-21** paper [Unsupervised Path Representation Learning with Curriculum Negative Sampling]().

![image](https://github.com/Sean-Bin-Yang/Path-InfoMax/blob/502ea3a57a578b325b704f50eb9808ee5968b744/Fig1.png)

## Overview
Here we give a PyTorch implementation of PIM. The repository is organized as follows:

- `data/` includes training data sample, when use your own data you can follow this format;
- `models/` contains the implementation of the PIM pipeline (`pim.py`);
- `layers/` contains the implementation of the bilinear discriminator (`discriminator.py`);
- `utils/` contains the necessary processing tool (`process.py`).

To better understand the code, we recommend that you could read the code of DGI/Petar (https://arxiv.org/abs/1809.10341) in advance. Besides, you could further optimize the code based on your own needs.

## Dataset
We use two dataset in our paper Aalborg and Harbin and download the [Harbin](https://drive.google.com/file/d/1tdgarnn28CM01o9hbeKLUiJ1o1lskrqA/view) dataset to train the PIM.

## Requirements

  * Ubuntu OS (16.04)
  * PyTorch >=1.2.0
  * Numpy >= 1.16.2
  * Pickle

Please refer to the source code to install the required packages in Python.

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
