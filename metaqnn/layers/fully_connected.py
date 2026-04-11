import torch.nn as nn


class FullyConnected(nn.Module):

    def __init__(self, in_features, num_neurons, layer_depth, representation_size):
        super().__init__()

        self.in_channels = in_features
        self.layer_depth = layer_depth
        self.representation_size = representation_size

        self.fc = nn.Linear(in_features=in_features, out_features=num_neurons)


    def forward(self, x):
        x = self.fc(x)

        return x

