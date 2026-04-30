import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from train_Hider import train
from load_model import load


class MetaQNN(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=1024):
        super().__init__(observation_space, features_dim)

        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 256, kernel_size=3, padding='same'), 
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(5,3),
            nn.Conv2d(128, 128, kernel_size=1, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.Conv2d(128, 64, kernel_size=3, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool2d(3,2),
            nn.Flatten(),
        )

    def forward(self, observations):
        return self.cnn(observations)
    

if __name__=='__main__':
    name = 'Meta_QNN_model'
    log_dir = f"./logs_{name}/"
    features_dim = 1024
    train(log_dir, MetaQNN, name, features_dim)
    load(name)
