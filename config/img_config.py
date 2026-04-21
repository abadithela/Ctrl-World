import torch
import os
import json
from dataclasses import dataclass
from pathlib import Path
import re

@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    PROJECT_ROOT = Path(__file__).resolve().parent
    pretrained_model_path = f"{PROJECT_ROOT}/svd"
    clip_model_path = f"{PROJECT_ROOT}/clip"
    ckpt_path = f"{PROJECT_ROOT}/checkpoints/checkpoint-10000.pt"
    pi_ckpt = f"{PROJECT_ROOT}/openpi/checkpoints/openpi-assets/checkpoints/pi05_droid"

    # dataset parameters
    dataset_root_path = f"{PROJECT_ROOT}/init_conditions"
    dataset_subdir = "env_imgs" # Figure out why these arguments are not clear!

    # meta info
    dataset_meta_info_path = 'dataset_meta_info' #'/cephfs/cjyyj/code/video_evaluation/exp_cfg'#'dataset_meta_info'
    dataset_cfgs = dataset_subdir
    prob=[1.0]
    annotation_name='annotation' #'annotation_all_skip1'
    num_workers=4
    down_sample=3 # downsample 15hz to 5hz
    skip_step = 1

    save_root_path = f"{PROJECT_ROOT}/wm_outputs"

    # logs parameters
    debug = False
    tag = 'droid_subset'
    output_dir = f"model_ckpt/{tag}"
    wandb_run_name = tag
    wandb_project_name = "droid_example"


    # training parameters
    learning_rate= 1e-5 # 5e-6
    gradient_accumulation_steps = 1
    mixed_precision = 'fp16'
    train_batch_size = 4
    shuffle = True
    num_train_epochs = 100
    max_train_steps = 500000
    checkpointing_steps = 20000
    validation_steps = 2500
    max_grad_norm = 1.0
    # for val
    video_num= 10

    ############################ model args ##############################
    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2 #7.5 #7.5 #7.5 #3.0
    num_inference_steps = 50
    decode_chunk_size = 7
    width = 320
    height = 192
    # num history and num future predictions
    num_frames= 5
    num_history = 6
    action_dim = 7
    text_cond = True
    frame_level_cond = True
    his_cond_zero = False
    dtype = torch.bfloat16 # [torch.float32, torch.bfloat16] # during inference, we can use bfloat16 to accelerate the inference speed and save memory

    ########################### rollout args ############################
    # policy
    task_type = "irom" # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict = {"irom": 0.75}
    policy_type = 'pi05' # choose from ['pi05', 'pi0', 'pi0fast']
    
    action_adapter = 'models/action_adapter/model2_15_9.pth' # adapat action from joint vel to cartesian pose
    pred_step = 5 # predict 5 steps (1s) action each time
    policy_skip_step = 2 # horizon = (pred_step-1) * policy_skip_step
    interact_num = 12 # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path = 'dataset_meta_info/droid/stat.json'
    val_model_path = ckpt_path
    history_idx = [0,0,-12,-9,-6,-3]

    # select different traj for different tasks
    def setup_paths(self):
        self.save_dir = f"{self.save_root_path}/{self.policy_type}"
        print("Saving to: ", self.save_dir)

        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)

        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        self.interact_num = 25
        self.val_dataset_dir = f"{self.dataset_root_path}/{self.dataset_subdir}"

        subfolders = sorted([
            name for name in os.listdir(f"{self.val_dataset_dir}")
            if os.path.isdir(os.path.join(self.val_dataset_dir, name))
        ])

        # 2. Extract digits from each folder name
        folder_ids = []
        for name in subfolders:
            # re.findall returns a list of all numeric sequences found in the string
            match = re.findall(r'\d+', name) 
            if match:
                folder_ids.append(int(match[0])) # Convert '100' to 100

        # 3. Determine your start point
        start_val = min(folder_ids) if folder_ids else 0
        self.val_id = subfolders
        self.start_idx = [start_val] * len(self.val_id)
        self.instruction = [
            "pick and place",
            ]
