#!/bin/bash

#SBATCH --time=200:00:0
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=10G
#SBATCH --partition=kidl
#SBATCH --out=log/%j.out

module load comp/gcc/8.3.0
source activate one-hot-max-quant

srun --unbuffered python train.py \
  -batch_size 32 \
  -n_epochs 10 \
  -learning_rate 1e-4 \
  -lambda_rd 0.95 0.999 \
  -encoder_channels 480 \
  -bottleneck_channels 0 480 \
  -no_quant_symbols 32 \
  -integer_quant \
  --test_sets /beegfs/work/shared/kodak/test/ \
  --description "int_adaptive_480ch"