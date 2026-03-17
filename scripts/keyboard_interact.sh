#!/bin/bash

#SBATCH --job-name=aigen
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=1:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --partition=all
#SBATCH --signal=USR1@60
#SBATCH --exclude=neu306

source ~/.bashrc

conda activate ctrl-world

svd_folder="/n/fs/irom-testing/world_models/Ctrl-World/stable-video-diffusion-img2vid"
clip_folder="/n/fs/irom-testing/world_models/Ctrl-World/clip-vit-base-patch32"
ckpt_path="/n/fs/irom-testing/world_models/Ctrl-World/checkpoints/checkpoint-10000.pt"

CUDA_VISIBLE_DEVICES=0 python scripts/rollout_key_board.py \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names droid_subset \
    --svd_model_path ${svd_folder} \
    --clip_model_path ${clip_folder} \
    --ckpt_path ${ckpt_path} \
    --task_type keyboard \
    --keyboard lllrrr