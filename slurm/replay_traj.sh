svd_folder="/n/fs/irom-testing/world_models/Ctrl-World/stable-video-diffusion-img2vid"
clip_folder="/n/fs/irom-testing/world_models/Ctrl-World/clip-vit-base-patch32"
ckpt_path="/n/fs/irom-testing/world_models/Ctrl-World/checkpoints/checkpoint-10000.pt"

CUDA_VISIBLE_DEVICES=0 python scripts/rollout_replay_traj.py \
    --dataset_root_path dataset_example \
    --dataset_meta_info_path dataset_meta_info \
    --dataset_names droid_subset \
    --svd_model_path ${svd_folder} \
    --clip_model_path ${clip_folder} \
    --ckpt_path ${ckpt_path}