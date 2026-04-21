'''
Script to tests the inputs fed into Ctrl-World model.
'''
import numpy as np
import torch
import numpy as np
import torch.nn as nn
from accelerate import Accelerator
from openpi.training import config as config_pi
from openpi.policies import policy_config
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from decord import VideoReader, cpu
from models.ctrl_world import CrtlWorld
import json

class agent():
    def __init__(self,args):
          
        # args = Args()
        self.args = args
        self.accelerator = Accelerator()
        self.device = self.accelerator.device
        self.dtype = args.dtype

        # load pi policy
        if 'pi05' in args.policy_type:
            config = config_pi.get_config("pi05_droid")
            # checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets-preview/checkpoints/pi05_droid' 
        elif 'pi0fast' in args.policy_type:
            config = config_pi.get_config("pi0fast_droid")
            # checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0fast_droid'
        elif 'pi0' in args.policy_type:
            config = config_pi.get_config("pi0_droid")
            # checkpoint_dir = '/cephfs/shared/llm/openpi/openpi-assets/checkpoints/pi0_droid'
        else:
            raise ValueError(f"Unknown policy type: {args.policy_type}")
        self.policy = policy_config.create_trained_policy(config, args.pi_ckpt)

        # load ctrl-world model
        self.model = CrtlWorld(args)
        self.model.load_state_dict(torch.load(args.val_model_path))
        self.model.to(self.accelerator.device).to(self.dtype)
        self.model.eval()
        print("load world model success")
        with open(f"{args.data_stat_path}", 'r') as f:
            data_stat = json.load(f)
            self.state_p01 = np.array(data_stat['state_01'])[None,:]
            self.state_p99 = np.array(data_stat['state_99'])[None,:]
        
        # Since the official Pi-Droid model output joint velocity, and crtl-world is train on cartesian space, we need to load an light-weight adapter to transform joint velocity action into cartesian pose action. 
        if args.action_adapter is not None:
            from models.action_adapter.train2 import Dynamics
            self.dynamics_model = Dynamics(action_dim=7, action_num=15, hidden_size=512).to(self.device)
            self.dynamics_model.load_state_dict(torch.load(args.action_adapter, map_location=self.device))    

    def get_traj_info(self, id, start_idx=0, steps=8,skip=1):
        val_dataset_dir = self.args.val_dataset_dir
        num_frames = steps
        annotation_path = f"{val_dataset_dir}/annotation/val/{id}.json"
        with open(annotation_path) as f:
            anno = json.load(f)
            try:
                length = len(anno['action'])
            except:
                length = anno["video_length"]
        frames_ids = np.arange(start_idx, start_idx + num_frames * skip, skip)
        max_ids = np.ones_like(frames_ids) * (length - 1)
        frames_ids = np.min([frames_ids, max_ids], axis=0).astype(int)
        print("Ground truth frames ids", frames_ids)
        
        # get action and joint pos
        instruction = anno['texts'][0]
        car_action = np.array(anno['states'])
        car_action = car_action[frames_ids]
        joint_pos = np.array(anno['joints'])
        joint_pos = joint_pos[frames_ids]

        # get videos
        video_dict =[]
        video_latent = []
        for id in range(len(anno['videos'])):
            video_path = anno['videos'][id]['video_path']
            video_path = f"{val_dataset_dir}/{video_path}"
            # load videos from all views
            vr = VideoReader(video_path, ctx=cpu(0), num_threads=2)
            try:
                true_video = vr.get_batch(range(length)).asnumpy()
            except:
                true_video = vr.get_batch(range(length)).numpy()
            true_video = true_video[frames_ids]
            video_dict.append(true_video)

            # encode video
            device = self.device
            true_video = torch.from_numpy(true_video).to(self.dtype).to(device)
            x = true_video.permute(0,3,1,2).to(device) / 255.0*2-1
            vae = self.model.pipeline.vae
            with torch.no_grad():
                batch_size = 32
                latents = []
                for i in range(0, len(x), batch_size):
                    batch = x[i:i+batch_size]
                    latent = vae.encode(batch).latent_dist.sample().mul_(vae.config.scaling_factor)
                    latents.append(latent)
                x = torch.cat(latents, dim=0)
            video_latent.append(x)
        return car_action, joint_pos, video_dict, video_latent, instruction

if __name__ == "__main__":
    from config import wm_args
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--clip_model_path', type=str, default="/n/fs/irom-testing/world_models/Ctrl-World/clip-vit-base-patch32")
    parser.add_argument('--ckpt_path', type=str, default="/n/fs/irom-testing/world_models/Ctrl-World/checkpoints/checkpoint-10000.pt")
    parser.add_argument('--dataset_root_path', type=str, default="dataset_example")
    parser.add_argument('--dataset_meta_info_path', type=str, default="dataset_meta_info")
    parser.add_argument('--dataset_names', type=str, default="droid_subset")
    parser.add_argument('--task_type', type=str, default="pickplace")
    parser.add_argument('--pi_ckpt', type=str, default='/n/fs/irom-testing/world_models/Ctrl-World/openpi/checkpoints/pi05_droid')
    args_new = parser.parse_args()

    args = wm_args(task_type=args_new.task_type)

    def merge_args(cfg, cli_args):
        for k, v in vars(cli_args).items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    args = merge_args(args, args_new)

    # create agent
    Agent = agent(args)
    interact_num = args.interact_num
    pred_step = args.pred_step
    num_history = args.num_history
    num_frames = args.num_frames
    history_idx = args.history_idx

    # run len(val_id) trajectory
    for val_id_i, text_i, start_idx_i in zip(args.val_id, args.instruction, args.start_idx):
        print("Evaluating traj id:", val_id_i)
        # get initial state and groud truth
        id = val_id_i
        eef_gt, joint_pos_gt, video_dict, video_latents,_ = Agent.get_traj_info(val_id_i, start_idx=start_idx_i, steps=int(pred_step*interact_num+8))
        print("text_i:",text_i, "eef pose at t=0", eef_gt[0], "joint at t=0", joint_pos_gt[0])
        breakpoint()
