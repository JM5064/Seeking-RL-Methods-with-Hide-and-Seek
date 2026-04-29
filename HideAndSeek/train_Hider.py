import torch as th
import torch.nn as nn
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.monitor import Monitor
import os


class BaseCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=512):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.LeakyReLU(), 
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.LeakyReLU(),
            nn.Flatten(),
        )

        with th.no_grad():
            n_flatten = self.cnn(th.as_tensor(observation_space.sample()[None]).float()).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flatten, features_dim), 
            nn.LeakyReLU()
        )

    def forward(self, observations):
        return self.linear(self.cnn(observations))



log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

channel = EngineConfigurationChannel()
channel.set_configuration_parameters(time_scale=20.0) 
print("Connecting to Unity...")                         
unity_env = UnityEnvironment(side_channels=[channel], timeout_wait=120)


env = UnityToGymWrapper(unity_env, allow_multiple_obs=False)
env = Monitor(env, log_dir)

policy_kwargs = dict(
    features_extractor_class=BaseCNN,
    features_extractor_kwargs=dict(features_dim=512),
)


model = PPO(
    "CnnPolicy",
    env,
    policy_kwargs=policy_kwargs,
    learning_rate=3e-4,
    n_steps=2048,       
    batch_size=64,
    verbose=1,
    tensorboard_log="./logs/"
)

model.learn(total_timesteps=1_000_000)  
model.save("hider_policy")
env.close()
