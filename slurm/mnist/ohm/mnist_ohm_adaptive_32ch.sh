#!/bin/bash

#SBATCH --time=200:00:0
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --out=log/%j.out

module load comp/gcc/8.3.0
source activate one-hot-max-quant

srun --unbuffered python train.py \
  -batch_size 512 \
  -log_interval 1000 \
  -n_epochs 1000 \
  -learning_rate 1e-3 \
  -lambda_rd 0.95 0.999 \
  -encoder_channels 8192 \
  -bottleneck_channels 0 32 \
  -no_quant_symbols 256 \
  --mnist \
  --test_sets /beegfs/work/shared/mnist_png/validation /beegfs/work/shared/mnist_png/testing \
  --description 'mnist_ohm_adaptive_32ch'
