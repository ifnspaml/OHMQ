#!/bin/bash

#SBATCH --time=20:0
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --partition=debug
#SBATCH --out=log/%j.out

source activate ohm

srun --unbuffered python compress.py \
  --image_path $2 \
  --output_dir $3 \
  --checkpoint $1