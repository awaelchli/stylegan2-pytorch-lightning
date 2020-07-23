# StyleGAN 2 in PyTorchLightning

This fork of the repository by [Kim Seonghyeon (rosinality)](https://github.com/rosinality/stylegan2-pytorch) ports the StyleGAN 2 implementation to the [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning) format. The code is not 100% equivalent but efforts will continue to achieve that.


## Requirements

Install the requirements in a new environment with

```bash
pip install -r requirements.txt
```

It installs PyTorch 1.5.1 and you will need CUDA 10.1/10.2 and a GCC compiler version >= 5.5.

## Data Preparation

Follow the steps in the [original repository](https://github.com/rosinality/stylegan2-pytorch) to prepare the LMDB dataset format. 

## Example Usage

Train on 4 gpus on images of size 128x128 and an effective batch size of 8 (2 batches per GPU)
```bash
python train.py --gpus 4 --batch_size 2 /path/to/lmdb_data --size 128
```
