#!/bin/bash

#SBATCH --job-name=wm-policy-eval
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=13:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --partition=all
#SBATCH --signal=USR1@60
#SBATCH --exclude=neu306

source ~/.bashrc

conda activate ctrl-world-v2

dataset_subdir="env_imgs2"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python scripts/interact_pi.py --dataset_subdir $dataset_subdir 

