#!/bin/bash

#SBATCH --job-name=wm-policy-eval
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
policy_path="/n/fs/irom-testing/world_models/Ctrl-World/openpi/checkpoints/pi05_droid"

dataset_names="pick_marker"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python scripts/interact_pi_irom.py \
    --task_type ${dataset_names} \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names ${dataset_names} \
    --clip_model_path ${clip_folder} \
    --ckpt_path ${ckpt_path} \
    --pi_ckpt ${policy_path} 


# CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python scripts/rollout_interact_pi.py \
#     --task_type pickplace \
#     --dataset_root_path dataset_example \
#     --dataset_meta_info_path dataset_meta_info \
#     --dataset_names droid_subset \
#     --clip_model_path "/n/fs/irom-testing/world_models/Ctrl-World/clip-vit-base-patch32" \
#     --ckpt_path "/n/fs/irom-testing/world_models/Ctrl-World/checkpoints/checkpoint-10000.pt" \
#     --pi_ckpt "/n/fs/irom-testing/world_models/Ctrl-World/openpi/checkpoints/pi05_droid" \
#     --task_type "pickplace"