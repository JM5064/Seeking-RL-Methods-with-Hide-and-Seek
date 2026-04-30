import torch as th
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from shimmy import GymV21CompatibilityV0
from stable_baselines3.common.callbacks import CheckpointCallback

import os


def train(log_dir, model_architecture, model_name, features_dim):
    if th.cuda.is_available():
        device = th.device("cuda")
        print("Using device: CUDA (Nvidia GPU)")
    elif th.backends.mps.is_available() and th.backends.mps.is_built():
        device = th.device("mps")
        print("Using device: MPS (Apple Silicon GPU)")
    else:
        device = th.device("cpu")
        print("Using device: CPU")

    os.makedirs(log_dir, exist_ok=True)

    channel = EngineConfigurationChannel()
    channel.set_configuration_parameters(time_scale=10)

    print("Connecting to Unity...")
    unity_env = UnityEnvironment(side_channels=[channel], timeout_wait=120)

    env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)
    env = GymV21CompatibilityV0(env=env, render_mode="rgb_array")
    env = Monitor(env, log_dir)

    checkpoint_callback = CheckpointCallback(
        save_freq=1024,          
        save_path=log_dir,       
        name_prefix=model_name   
    )




    policy_kwargs = dict(
        features_extractor_class=model_architecture,
        features_extractor_kwargs=dict(features_dim=features_dim),
    )

    model = PPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        verbose=1,
        device=device,
        tensorboard_log=log_dir,
    )

    model.learn(total_timesteps=5_000, callback=checkpoint_callback)
    model.save("hider_policy_"+model_name)
    env.close()
    print("Environment closed.")  




