import torch.nn as nn
import torch.nn.functional as F

class Convolution(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, layer_depth, representation_size):
        super().__init__()

        self.in_channels = in_channels
        self.layer_depth = layer_depth
        self.representation_size = representation_size

        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding='same')
        

    def forward(self, x):
        x = self.conv(x)
        x = F.relu(x)

        return x

