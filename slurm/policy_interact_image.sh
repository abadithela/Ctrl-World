#!/bin/bash

#SBATCH --job-name=wm-policy-eval
#SBATCH --output=logs/%A_evalpi0.out
#SBATCH --error=logs/%A_evalpi0.err
#SBATCH --time=1:00:00
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --partition=all
#SBATCH --signal=USR1@60
#SBATCH --exclude=neu306
#SBATCH --array=1-20  # one job per chunk; matches NUM_CHUNKS below

source ~/.bashrc

conda activate ctrl-world-v2

NUM_CHUNKS=20
CHUNK_IDX=$(expr $SLURM_ARRAY_TASK_ID - 1)

clip_folder="/n/fs/irom-testing/world_models/Ctrl-World/clip-vit-base-patch32"
ckpt_path="/n/fs/irom-testing/world_models/Ctrl-World/checkpoints/checkpoint-10000.pt"
policy_path="/n/fs/irom-testing/world_models/Ctrl-World/openpi/checkpoints/pi05"
save_root="/n/fs/irom-testing/world_models/Ctrl-World/wm_outputs"

dataset_names="pi05_test"

CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 uv run python scripts/policy_interact_image.py \
    --task_type ${dataset_names} \
    --clip_model_path ${clip_folder} \
    --ckpt_path ${ckpt_path} \
    --pi_ckpt ${policy_path} \
    --save_root_path ${save_root} \
    --num_chunks ${NUM_CHUNKS} \
    --chunk_idx ${CHUNK_IDX} \
    --num_trials 5 \