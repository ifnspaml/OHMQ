#!/bin/bash

#SBATCH --time=200:00:0
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --partition=gpu
#SBATCH --out=log/%j.out
#SBATCH --array=0-5
#SBATCH --out=log/%A_%a.out

module load comp/gcc/8.3.0
source activate one-hot-max-quant

lambda_rd_min=(0.95 0.95 0.9 0.9 0.85 0.85)
lambda_rd_max=(0.999 0.999 0.999 0.999 0.9 0.9)
bottleneck_channels_min=(0 8 0 8 0 8)

echo "Array Job, lambda_rd_min: ${lambda_rd_min[${SLURM_ARRAY_TASK_ID:--1}]}, bottleneck_channels_min: ${bottleneck_channels_min[${SLURM_ARRAY_TASK_ID:--1}]}"

srun --unbuffered python train.py \
  -batch_size 512 \
  -log_interval 1000 \
  -n_epochs 1000 \
  -learning_rate 1e-3 \
  -lambda_rd ${lambda_rd_min[${SLURM_ARRAY_TASK_ID:--1}]} ${lambda_rd_max[${SLURM_ARRAY_TASK_ID:--1}]}  \
  -encoder_channels 4096 \
  -bottleneck_channels ${bottleneck_channels_min[${SLURM_ARRAY_TASK_ID:--1}]} 16 \
  -no_quant_symbols 256 \
  --mnist \
  --test_sets /beegfs/work/shared/mnist_png/validation /beegfs/work/shared/mnist_png/testing \
  --description 'mnist_ohm_adaptive_16ch'
