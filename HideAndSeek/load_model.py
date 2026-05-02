import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=5.0)

print("Connecting to Unity... Press Play in Unity Editor first")
unity_env = UnityEnvironment(side_channels=[channel])
env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)

print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

model = PPO.load(
    "/Users/justinmao/Documents/GitHub/Seeking-RL-Methods-with-Hide-and-Seek/HideAndSeek/hider_policy_Meta_QNN_model.zip",
    #"/Users/justinmao/Documents/GitHub/Seeking-RL-Methods-with-Hide-and-Seek/HideAndSeek/hider_policy_base_model.zip",
    device="cpu",
    custom_objects={
        "observation_space": env.observation_space,
        "action_space": env.action_space,
        "lr_schedule": lambda _: 3e-4,
        "clip_range": lambda _: 0.2,
    }
)

print("Model loaded, running inference...")
obs = env.reset()

episode_count = 100
tot_rewards = 0
num_episodes = 0
total_steps = 0
while True:
    action, _ = model.predict(obs, deterministic=True)
    result = env.step(action)
    
    obs, reward, done, info = result  
    tot_rewards += reward
    total_steps += 1

    if done:
        num_episodes += 1
        obs = env.reset()

        if num_episodes == episode_count:
            break
      
avg_reward_t = tot_rewards / total_steps
avg_reward_ep = tot_rewards / episode_count
timestep_ep = total_steps/episode_count

print(avg_reward_ep)
print(avg_reward_t)
print(timestep_ep)


