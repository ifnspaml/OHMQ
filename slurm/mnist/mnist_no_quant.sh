#!/bin/bash

#SBATCH --time=200:00:0
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10G
#SBATCH --partition=gpub
#SBATCH --exclude=gpu05
#SBATCH --out=log/%j.out

source activate ohm

srun --unbuffered python train.py \
  -batch_size 512 \
  -log_interval 1000 \
  -n_epochs 1000 \
  -learning_rate 1e-4 \
  -encoder_channels 32 \
  -no_quant \
  --test_sets /beegfs/work/shared/mnist_png/validation /beegfs/work/shared/mnist_png/testing \
  --description 'mnist_no_quant'
