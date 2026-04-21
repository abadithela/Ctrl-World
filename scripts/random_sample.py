import random
import shutil
import os 

all_envs = "/n/fs/irom-testing/world_models/Ctrl-World/init_conditions/envs"
real_envs = "/n/fs/irom-testing/world_models/Ctrl-World/init_conditions/real_envs"
os.makedirs(real_envs, exist_ok=True)
N = 700
weight = 0.1*N

# Coin flip for probability of sampling the environment
def sample_env(env):
    if random.random() <= weight/N:
        return env
    else:
        return None

for i in range(N):
    env = f"{all_envs}/env{i}"
    sampled_env = sample_env(env)
    if sampled_env is not None:
        # Copy the sampled environment to the output directory
        destination_folder = f"{real_envs}/env{i}"
        source_folder = f"{all_envs}/env{i}"
        shutil.copytree(source_folder, destination_folder)
        print(f"Copied {source_folder} to {destination_folder}")
