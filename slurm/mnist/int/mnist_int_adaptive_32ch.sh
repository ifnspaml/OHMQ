#!/bin/bash

#SBATCH --time=200:00:0
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --partition=gpub
#SBATCH --exclude=gpu05
#SBATCH --out=log/%j.out

source activate ohm
module load comp/gcc/8.3.0

srun --unbuffered python train.py \
  -batch_size 512 \
  -log_interval 1000 \
  -n_epochs 1000 \
  -learning_rate 1e-3 \
  -lambda_rd 0.95 0.999 \
  -encoder_channels 32 \
  -bottleneck_channels 0 32 \
  -no_quant_symbols 256 \
  -integer_quant \
  --mnist \
  --test_sets data/mnist_png/validation data/mnist_png/testing \
  --description 'mnist_int_adaptive_32ch'
