# GotFlow3D

This repository contains the source code for our paper: "GotFlow3D: Recurrent Graph Optimal Transport for Learning 3D Flow Motion in Particle Tracking".

## Requirements

- The code is tested on Ubuntu 20.04.
- CUDA 11.3
- Python 3.8
- pytorch 1.10
- numpy
- scipy
- tqdm
- torch-scatter
- tensorboard
- imageio
- setuptools 59.5.0

## Installation

```shell
conda create -n gotflow3d python=3.8
conda activate gotflow3d
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda install tqdm tensorboard scipy imageio
pip install torch-scatter==2.0.9 -f https://pytorch-geometric.com/whl/torch-1.10.1+cu113.html
```

It can be installed within half an hour, which mainly depends on the internet speed.

## Required Data

To evaluate/train GotFlow3D, you will need to download the required datasets [FluidFlow3D-family](https://github.com/JiamingSkGrey/FluidFlow3D-family).  We also provide a small dataset in the data folder to demo the software/code.

## Usage

##### Train

```shell
python train.py --exp_path=test --num_epochs=30 --iters=8 --root=./
```

where `exp_path` is the experiment folder name, `num_epochs` is the number of epochs in training and `iters` denotes the    number  of iterations.

##### Test

```shell
python test.py --exp_path=test --iters=8 --root=./ --weights=./experiments/weights_GotFlow3D/checkpoints/best_checkpoint.params
```

It takes about 0.1s to estimate the flow of one sample containing about 2000 particles on the NVIDIA RTX 3090 GPU. The expected output of the network is the the dense flow field.

## Citation

If you find our work useful in your research, please consider citing:

```
@article{
  title={{GotFlow3D: Recurrent Graph Optimal Transport for Learning 3D Flow Motion in Particle Tracking}},
  author={Liang, Jiaming and Cai, Shengze and Xu, Chao},
}
```

