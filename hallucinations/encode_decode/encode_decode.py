from PIL import Image
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
import numpy as np
import torch
from accelerate import Accelerator
import einops
from models.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline

source_img_path = "/n/fs/ug-ctrl-wrld/ctrl_world/hallucinations/encode_decode/test_encode_decode.png"
concat_img = Image.open(source_img_path)
imgs_latents = []
accelerator = Accelerator()
device = accelerator.device
pipeline = StableVideoDiffusionPipeline.from_pretrained("/n/fs/ug-ctrl-wrld/ctrl_world/svd")
vae = pipeline.vae

# separate views
for i in range(3):
    img = concat_img.crop((i*320, 0, (i+1)*320, 192))
    # for verifying correct crop
    # img.save(f"{source_img_path[:source_img_path.rfind('/')]}/{i+1}.png")

    # based on ctrl world encoding code:
    img = np.array(img.convert("RGB"))  # (H, W, 3)
    # PROBABLY REDUNDANT - img = cv2.resize(img, (320, 192))  # (192, 320, 3)
    img = img[np.newaxis]  # (1, 192, 320, 3)
    frame = torch.from_numpy(img).to(torch.bfloat16).to(device)
    x = frame.permute(0, 3, 1, 2) / 255.0 * 2 - 1  # (1, 3, H, W)
    vae_dtype = next(vae.parameters()).dtype
    vae = vae.to(device)
    x = x.to(vae_dtype)
    with torch.no_grad():
        latent = vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)
    imgs_latents.append(latent)

# based on ctrl world decoding code:
init_frames_latent = torch.cat([i[0] for i in imgs_latents], dim=1).unsqueeze(0)
init_frames_latent = einops.rearrange(init_frames_latent, 'b c (m h) w -> (b m) c h w', m=3) # (1, 4, 72, 40) -> (3, 4, 24, 40)
scaled_latent = init_frames_latent / pipeline.vae.config.scaling_factor
vae_dtype = next(pipeline.vae.parameters()).dtype
scaled_latent = scaled_latent.to(vae_dtype)
decoded = pipeline.vae.decode(scaled_latent, num_frames=1).sample

# process decoded imgs
img = ((decoded / 2.0 + 0.5).clamp(0, 1) * 255)
img = img.detach().to(torch.float32).cpu().numpy().transpose(0, 2, 3, 1).astype(np.uint8)
img_tiled = np.concatenate([v for v in img], axis=1)  # concat along width
Image.fromarray(img_tiled).save(f"{source_img_path[:source_img_path.index('.')]}_output.png")