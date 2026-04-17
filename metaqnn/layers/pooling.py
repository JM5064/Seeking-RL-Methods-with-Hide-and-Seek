import torch.nn as nn


class Pooling(nn.Module):

    def __init__(self, kernel_size_stride, layer_depth, representation_size):
        super().__init__()

        self.layer_depth = layer_depth
        self.representation_size = representation_size

        self.pool = nn.MaxPool2d(kernel_size=kernel_size_stride[0], stride=kernel_size_stride[1])


    def forward(self, x):
        x = self.pool(x)

        return x

