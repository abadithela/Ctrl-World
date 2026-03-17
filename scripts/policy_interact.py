### Interact with policy using initial images from real_envs/envN directories
from dataclasses import dataclass
from openpi.training import config as config_pi
from openpi.policies import policy_config
from openpi_client import image_tools
from pathlib import Path
from PIL import Image
import numpy as np
import time
import h5py

from accelerate import Accelerator
import torch
from diffusers import StableVideoDiffusionPipeline
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import mediapy
import sys
from scipy.spatial.transform import Rotation as R

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
from models.ctrl_world import CtrlWorld
from models.utils import key_board_control, get_fk_solution


@dataclass
class wm_args:
    ########################### training args ##############################
    # model paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    pretrained_model_path = f"{PROJECT_ROOT}/svd"
    clip_model_path = f"{PROJECT_ROOT}/clip"
    ckpt_path = f"{PROJECT_ROOT}/checkpoints/checkpoint-10000.pt"
    pi_ckpt = f"{PROJECT_ROOT}/openpi/checkpoints/openpi-assets/checkpoints/pi05_droid"
    policy_type = 'pi05' if 'pi05' in pi_ckpt else ('pi0fast' if 'pi0_fast' in pi_ckpt else 'pi0')
    dataset_root_path = f"{PROJECT_ROOT}/init_conditions"
    dataset_subdir = "real_envs_rollouts" # Figure out why these arguments are not clear!
    save_root_path = f"{PROJECT_ROOT}/wm_outputs"

    # meta info
    dataset_meta_info_path = 'dataset_meta_info' #'/cephfs/cjyyj/code/video_evaluation/exp_cfg'#'dataset_meta_info'
    dataset_cfgs = dataset_subdir
    prob=[1.0]
    annotation_name='annotation' #'annotation_all_skip1'
    num_workers=4
    down_sample=3 # downsample 15hz to 5hz
    skip_step = 1

    # logs parameters
    debug = False
    tag = 'real_envs'
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
    video_num= 10

    ############################ model args ##############################
    # model parameters
    motion_bucket_id = 127
    fps = 7
    guidance_scale = 2
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
    action_adapter = 'models/action_adapter/model2_15_9.pth' # adapat action from joint vel to cartesian pose
    pred_step = 5 # predict 5 steps (1s) action each time
    policy_skip_step = 2 # horizon = (pred_step-1) * policy_skip_step
    interact_num = 12 # number of interactions (each interaction contains pred_step steps)

    # wm
    data_stat_path = 'dataset_meta_info/droid/stat.json'
    val_model_path = ckpt_path
    history_idx = [0,0,-12,-9,-6,-3]

    ########################### rollout args ############################
    # policy
    task_type: str = "irom" # choose from ['pickplace', 'towel_fold', 'wipe_table', 'tissue', 'close_laptop','tissue','drawer','stack']
    gripper_max_dict = {'irom':0.75}

    # select different traj for different tasks
    def __post_init__(self):
        # Select the appropriate chunk based on index variable
        idx = 1 # or whatever variable you're using
        self.save_dir = f"{self.save_root_path}/{self.policy_type}_{self.dataset_subdir}_pt{idx}"
        print("Saving to: ", self.save_dir)

        # Per-task gripper max
        self.gripper_max = self.gripper_max_dict.get(self.task_type, 0.75)
        # Default task_name
        self.task_name = f"Rollouts_interact_pi"
        if self.task_type == "replay":
            self.task_name = "Rollouts_replay"

        # Configure per-task eval sets
        self.interact_num = 25
        self.val_dataset_dir = f"{self.dataset_root_path}/{self.dataset_subdir}"
        subfolders = sorted([
            name for name in os.listdir(f"{self.val_dataset_dir}/videos")
            if os.path.isdir(os.path.join(self.val_dataset_dir, "videos", name))
        ])

        start = 0
        end = len(subfolders)

        self.val_id = subfolders[start:end]
        self.start_idx = [0] * len(self.val_id)
        self.instruction = [
            "pick and place",]


class agent():
    def __init__(self, args):
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # load pi policy
        if 'pi05' in args.policy_type:
            config = config_pi.get_config("pi05_droid")
        elif 'pi0fast' in args.policy_type:
            config = config_pi.get_config("pi0_fast_droid")
        elif 'pi0' in args.policy_type:
            config = config_pi.get_config("pi0_droid")
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")
        self.policy = policy_config.create_trained_policy(config, args.pi_ckpt)

        # load ctrl-world model
        self.model = CtrlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None, :]
            self.state_p99 = np.array(data_stat['state_99'])[None, :]

        if args.action_adapter is not None:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
            self.dynamics_model.load_state_dict(torch.load(args.action_adapter, map_location=self.device))

    def normalize_bound(
        self,
        data: np.ndarray,
        data_min: np.ndarray,
        data_max: np.ndarray,
        clip_min: float = -1,
        clip_max: float = 1,
        eps: float = 1e-8,
    ) -> np.ndarray:
        ndata = 2 * (data - data_min) / (data_max - data_min + eps) - 1
        return np.clip(ndata, clip_min, clip_max)

    def _parse_instruction(self, txt_path):
        """Parse instruction text from env_setup.txt."""
        pick_obj = ""
        place_obj = ""
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("pick object:"):
                        pick_obj = line.split(":", 1)[1].strip()
                    elif line.startswith("place object:") and not place_obj:
                        place_obj = line.split(":", 1)[1].strip()
        except FileNotFoundError:
            pass
        if pick_obj and place_obj:
            return f"pick {pick_obj} and place on {place_obj}"
        elif pick_obj:
            return f"pick {pick_obj}"
        return "pick and place"

    def load_init_condition(self, env_dir, instruction_override=None, initial_joint_pos=None, initial_eef_pos=None):
        """
        Load initial condition from a real_envs/envN directory.

        Expected directory structure:
            envN/
                initial_frame_resized_left.png
                initial_frame_resized_right.png
                initial_frame_resized_wrist.png
                env_setup.txt
                robot_pose_envN.h5  (optional)

        Returns:
            eef_pos:      np.ndarray (7,)  initial end-effector pose
            joint_pos:    np.ndarray (8,)  initial joint positions (7 joints + 1 gripper)
            video_dict:   list of np.ndarray, each (1, H, W, 3)  one per camera view
            video_latent: list of torch.Tensor, each (1, 4, h, w) VAE-encoded latent
            instruction:  str
        """
        env_dir = Path(env_dir)

        # --- instruction ---
        if instruction_override:
            instruction = instruction_override
        else:
            instruction = self._parse_instruction(env_dir / "env_setup.txt")
        print(f"Instruction: {instruction}")

        # --- images (alphabetical: left, right, wrist) ---
        image_files = sorted([
            p for p in env_dir.iterdir()
            if p.name.startswith("initial_frame_resized_") and p.name.endswith(".png")
        ])
        if len(image_files) == 0:
            raise FileNotFoundError(f"No resized initial frames found in {env_dir}")
        print(f"Loading {len(image_files)} camera views: {[p.name for p in image_files]}")

        video_dict = []
        video_latent = []
        vae = self.model.pipeline.vae
        
        for img_path in image_files:
            img = np.array(Image.open(img_path).convert("RGB"))  # (H, W, 3)
            img = img[np.newaxis]  # (1, H, W, 3)
            video_dict.append(img)

            frame = torch.from_numpy(img).to(self.dtype).to(self.device)
            x = frame.permute(0, 3, 1, 2) / 255.0 * 2 - 1  # (1, 3, H, W)
            with torch.no_grad():
                latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
            video_latent.append(latent)  # (1, 4, h, w)

        # --- robot pose ---
        if initial_joint_pos is not None and initial_eef_pos is not None:
            joint_pos = np.array(initial_joint_pos, dtype=np.float32)
            eef_pos = np.array(initial_eef_pos, dtype=np.float32)
        else:
            h5_files = sorted(env_dir.glob("robot_pose_*.h5"))
            if h5_files:
                with h5py.File(h5_files[0], 'r') as f:
                    keys = list(f.keys())
                    print(f"robot_pose h5 keys: {keys}")
                    if 'joint_position' in f:
                        joint_pos = f['joint_position'][()][0].astype(np.float32)
                    elif 'joints' in f:
                        joint_pos = f['joints'][()][0].astype(np.float32)
                    else:
                        joint_pos = np.zeros(8, dtype=np.float32)
                    if 'cartesian_position' in f:
                        eef_pos = f['cartesian_position'][()][0].astype(np.float32)
                    elif 'eef_pos' in f:
                        eef_pos = f['eef_pos'][()][0].astype(np.float32)
                    else:
                        eef_pos = np.zeros(7, dtype=np.float32)
                    # ensure gripper dimension is included in joint_pos
                    if 'gripper_position' in f and joint_pos.shape[0] == 7:
                        gripper = f['gripper_position'][()][0:1].astype(np.float32)
                        joint_pos = np.concatenate([joint_pos, gripper])
            else:
                print(f"No robot_pose h5 found in {env_dir}, using default h5 init state.")
                joint_pos = np.zeros(8, dtype=np.float32)
                eef_pos = np.zeros(7, dtype=np.float32)

        return eef_pos, joint_pos, video_dict, video_latent, instruction

    def forward_wm(self, action_cond, video_latent_cond, his_cond=None, text=None):
        """
        Run world model forward pass.

        Unlike interact_pi_irom.py, this version does not take ground-truth video
        latents and only returns the predicted video.

        Returns:
            videos:         np.uint8 (num_views, num_frames, H, W, 3) predicted frames
            predict_latents: torch.Tensor (num_views, num_frames, 4, h, w)
        """
        args = self.args
        image_cond = video_latent_cond

        action_cond = self.normalize_bound(action_cond, self.state_p01, self.state_p99, clip_min=-1, clip_max=1)
        action_cond = torch.tensor(action_cond).unsqueeze(0).to(self.device).to(self.dtype)
        assert image_cond.shape[1:] == (4, 72, 40)
        assert action_cond.shape[1:] == (args.num_frames + args.num_history, args.action_dim)

        with torch.no_grad():
            if text is not None:
                text_token = self.model.action_encoder(action_cond, text, self.model.tokenizer, self.model.text_encoder)
            else:
                text_token = self.model.action_encoder(action_cond)
            pipeline = self.model.pipeline

            st = time.time()
            _, latents = CtrlWorldDiffusionPipeline.__call__(
                pipeline,
                image=image_cond,
                text=text_token,
                width=args.width,
                height=int(args.height * 3),
                num_frames=args.num_frames,
                history=his_cond,
                num_inference_steps=args.num_inference_steps,
                decode_chunk_size=args.decode_chunk_size,
                max_guidance_scale=args.guidance_scale,
                fps=args.fps,
                motion_bucket_id=args.motion_bucket_id,
                mask=None,
                output_type='latent',
                return_dict=False,
                frame_level_cond=True,
            )
        print("World model inference time: ", time.time() - st)
        latents = einops.rearrange(latents, 'b f c (m h) (n w) -> (b m n) f c h w', m=3, n=1)

        # decode predicted video
        decoded_video = []
        bsz, frame_num = latents.shape[:2]
        x = latents.flatten(0, 1)
        decode_kwargs = {}
        for i in range(0, x.shape[0], args.decode_chunk_size):
            chunk = x[i:i + args.decode_chunk_size] / pipeline.vae.config.scaling_factor
            decode_kwargs["num_frames"] = chunk.shape[0]
            decoded_video.append(pipeline.vae.decode(chunk, **decode_kwargs).sample)
        videos = torch.cat(decoded_video, dim=0)
        videos = videos.reshape(bsz, frame_num, *videos.shape[1:])
        videos = ((videos / 2.0 + 0.5).clamp(0, 1) * 255)
        videos = videos.detach().to(torch.float32).cpu().numpy().transpose(0, 1, 3, 4, 2).astype(np.uint8)

        return videos, latents

    def forward_policy(self, videos, state, joints, text, time_step=1):
        """Run policy inference on current observations."""
        image1 = videos[1]
        image2 = videos[2]
        image1 = torch.from_numpy(image1).to(torch.uint8)
        image2 = torch.from_numpy(image2).to(torch.uint8)
        assert image1.shape == (192, 320, 3), "Image 1 shape should be (192, 320, 3), got {}".format(image1.shape)
        image1 = torch.nn.functional.interpolate(image1.permute(2, 0, 1).unsqueeze(0).float(), size=(180, 320), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image2 = torch.nn.functional.interpolate(image2.permute(2, 0, 1).unsqueeze(0).float(), size=(180, 320), mode='bilinear', align_corners=False).squeeze(0).permute(1, 2, 0).to(torch.uint8)
        image1 = image1.numpy()
        image2 = image2.numpy()
        example = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(image1, 224, 224),
            "observation/wrist_image_left": image_tools.resize_with_pad(image2, 224, 224),
            "observation/joint_position": joints[:7],
            "observation/gripper_position": joints[-1:],
            "prompt": text,
        }
        action_chunk = self.policy.infer(example)["actions"]  # (10, 8) velocity

        current_joint = joints[None, :][:, :7]
        current_gripper = joints[None, :][:, 7:]
        if 'pi05' in self.args.policy_type:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
        else:
            idx = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 9, 9, 9]
        joint_vel = action_chunk[:, :7]
        gripper_pos = action_chunk[:, 7:]
        joint_vel = joint_vel[idx]
        gripper_pos = gripper_pos[idx]
        gripper_max = self.args.gripper_max
        gripper_pos = np.clip(gripper_pos, 0, gripper_max)
        joint_pos = self.dynamics_model(current_joint, joint_vel, None, training=False)

        joint_pos = np.concatenate([current_joint, joint_pos], axis=0)[:15]
        gripper_pos = np.concatenate([current_gripper, gripper_pos], axis=0)[:15]
        joint_vel = joint_vel

        state_fk = []
        for i in range(joint_pos.shape[0]):
            current_state_fk = get_fk_solution(joint_pos[i, :7])
            xyz = current_state_fk[:3, 3]
            rotation_matrix = current_state_fk[:3, :3]
            r = R.from_matrix(rotation_matrix)
            euler = r.as_euler('xyz')
            state_fk.append(np.concatenate([xyz, euler, gripper_pos[i]], axis=0))
        state_fk = np.array(state_fk)

        skip = self.args.policy_skip_step
        valid_num = int(skip * (self.args.pred_step - 1))
        policy_in_out = {
            'joint_pos': joint_pos[:valid_num],
            'joint_vel': joint_vel[:valid_num],
            'state_fk': state_fk[:valid_num],
        }
        state_fk_skip = state_fk[::skip][:self.args.pred_step]
        joint_pos_skip = joint_pos[::skip][:self.args.pred_step]
        joint_pos_skip = np.concatenate([joint_pos_skip, state_fk_skip[:, -1:]], axis=-1)
        return policy_in_out, joint_pos_skip, state_fk_skip


if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser()
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default="/n/fs/irom-testing/world_models/Ctrl-World/clip-vit-base-patch32")
    parser.add_argument('--ckpt_path', type=str, default="/n/fs/irom-testing/world_models/Ctrl-World/checkpoints/checkpoint-10000.pt")
    parser.add_argument('--svd_model_path', type=str, default=f"{PROJECT_ROOT}/stable-video-diffusion-img2vid")
    parser.add_argument('--save_root_path', type=str, default=None)
    parser.add_argument('--task_type', type=str, default=None)
    parser.add_argument('--policy_type', type=str, default=None)
    parser.add_argument('--pi_ckpt', type=str, default="/n/fs/irom-testing/world_models/Ctrl-World/openpi/checkpoints/pi05_droid")

    # init condition args
    parser.add_argument('--init_conditions_dir', type=str,
                        default="/n/fs/irom-testing/world_models/Ctrl-World/init_conditions/real_envs",
                        help="Root directory containing envN subdirectories")
    parser.add_argument('--env_ids', type=str, default=None,
                        help="Comma-separated env IDs to run, e.g. '5,7,19'. If not set, all envs in init_conditions_dir are used.")
    parser.add_argument('--instruction', type=str, default=None,
                        help="Override instruction text (applied to all envs)")
    parser.add_argument('--initial_joint_pos', type=str, default=None,
                        help="Comma-separated initial joint positions (8 values: 7 joints + gripper)")
    parser.add_argument('--initial_eef_pos', type=str, default=None,
                        help="Comma-separated initial end-effector pose (7 values: xyz + euler + gripper)")
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(cfg, cli_args):
        for k, v in vars(cli_args).items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    args = merge_args(args, args_new)

    # parse initial robot state overrides
    initial_joint_pos = None
    initial_eef_pos = None
    if args_new.initial_joint_pos is not None:
        initial_joint_pos = [float(x) for x in args_new.initial_joint_pos.split(',')]
    if args_new.initial_eef_pos is not None:
        initial_eef_pos = [float(x) for x in args_new.initial_eef_pos.split(',')]

    # resolve env directories to evaluate
    init_conditions_dir = Path(args_new.init_conditions_dir)
    if args_new.env_ids is not None:
        env_dirs = [init_conditions_dir / f"env{eid.strip()}" for eid in args_new.env_ids.split(',')]
    else:
        env_dirs = sorted([p for p in init_conditions_dir.iterdir() if p.is_dir()])
    print(f"Running {len(env_dirs)} environments: {[p.name for p in env_dirs]}")

    # create agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    history_idx = args.history_idx

    print("beginning rollouts...")
    for env_dir in env_dirs:
        env_name = env_dir.name
        print(f"\n{'='*60}")
        print(f"Environment: {env_name}")

        # load initial condition from PNG images
        eef_gt0, joint_pos_gt0, video_dict, video_latents, text_i = Agent.load_init_condition(
            env_dir,
            instruction_override=args_new.instruction,
            initial_joint_pos=initial_joint_pos,
            initial_eef_pos=initial_eef_pos,
        )
        print(f"text: {text_i}")
        print(f"initial eef pose: {eef_gt0}")
        print(f"initial joint pos: {joint_pos_gt0}")

        # initialize history buffers
        video_to_save, info_to_save = [], []
        his_cond, his_joint, his_eef = [], [], []
        first_latent = torch.cat([v[0] for v in video_latents], dim=1).unsqueeze(0)  # (1, 4, 72, 40)
        assert first_latent.shape == (1, 4, 72, 40), f"Expected (1,4,72,40), got {first_latent.shape}"
        for i in range(Agent.args.num_history * 4):
            his_cond.append(first_latent)
            his_joint.append(joint_pos_gt0[np.newaxis, :])  # (1, 8)
            his_eef.append(eef_gt0[np.newaxis, :])          # (1, 7)
        video_dict_pred = [v[0:1] for v in video_dict]

        # rollout loop
        for i in range(interact_num):
            print(f"\n--- interact step {i}/{interact_num} ---")

            print("################ policy forward ####################")
            current_joint = his_joint[-1][0]
            current_pose = his_eef[-1][0]
            current_obs = [v[-1] for v in video_dict_pred]
            policy_in_out, joint_pos, cartesian_pose = Agent.forward_policy(
                current_obs, current_pose, current_joint, text=text_i
            )
            print("cartesian space action (first):", cartesian_pose[0])
            print("cartesian space action (last): ", cartesian_pose[-1])

            print("################ world model forward ################")
            history_idx = args.history_idx
            action_cond = np.concatenate([his_eef[idx] for idx in history_idx], axis=0)
            action_cond = np.concatenate([action_cond, cartesian_pose], axis=0)  # (num_history+num_frames, 7)
            his_latent = torch.cat([his_cond[idx] for idx in history_idx], dim=0).unsqueeze(0)
            current_latent = his_cond[-1]  # (1, 4, 72, 40)

            video_dict_pred, predict_latents = Agent.forward_wm(
                action_cond,
                current_latent,
                his_cond=his_latent,
                text=text_i if Agent.args.text_cond else None,
            )

            print("################ record information ################")
            his_joint.append(joint_pos[pred_step - 1][np.newaxis, :])
            his_eef.append(cartesian_pose[pred_step - 1][np.newaxis, :])
            his_cond.append(
                torch.cat([v[pred_step - 1] for v in predict_latents], dim=1).unsqueeze(0)
            )
            video_to_save.append(video_dict_pred[1][:pred_step - 1])  # save exterior view
            info_to_save.append(policy_in_out)

        # save rollout video and info
        print("\n" + "=" * 60)
        video = np.concatenate(video_to_save, axis=0)
        text_id = text_i.replace(' ', '_').replace(',', '').replace('.', '').replace("'", '').replace('"', '')[:40]
        uuid = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_video = f"{args.save_dir}/{args.task_name}/video/time_{uuid}_{env_name}_{args.policy_skip_step}_{text_id}.mp4"
        os.makedirs(os.path.dirname(filename_video), exist_ok=True)
        mediapy.write_video(filename_video, video, fps=4)
        print(f"Saving video to {filename_video}")

        info = {'success': 1, 'env': env_name, 'instructions': text_i}
        for key in info_to_save[0].keys():
            info[key] = []
            for j in range(len(info_to_save)):
                info[key] += info_to_save[j][key].tolist()
        filename_info = f"{args.save_dir}/{args.task_name}/info/time_{uuid}_{env_name}_{pred_step}_{text_id}.json"
        os.makedirs(os.path.dirname(filename_info), exist_ok=True)
        with open(filename_info, 'w') as f:
            json.dump(info, f, indent=4)
        print(f"Saving trajectory info to {filename_info}")
        print("=" * 60)
