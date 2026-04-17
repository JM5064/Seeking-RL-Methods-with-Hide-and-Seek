import torch.nn as nn
import math

from metaqnn.rl_config import *
from metaqnn.layers.convolution import Convolution
from metaqnn.layers.pooling import Pooling
from metaqnn.layers.fully_connected import FullyConnected
from metaqnn.layers.termination import Termination


class MetaQNN(nn.Module):

    def __init__(self, layer_configs, input_size, input_channels):
        super().__init__()

        self.layers = nn.ModuleList()

        current_channels = input_channels
        current_resolution = input_size
        num_consecutive_fc_layers = 0

        for i, layer_config in enumerate(layer_configs):
            layer_type = layer_config['layer_type']
            if layer_type == CONVOLUTION:
                layer = Convolution(
                    in_channels=current_channels, 
                    out_channels=layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                    layer_depth=i,
                    representation_size=current_resolution
                )
                current_channels = layer_config['out_channels']
                current_resolution = (current_resolution - layer_config['kernel_size']) // layer_config['stride'] + 1
                num_consecutive_fc_layers = 0

            elif layer_type == POOLING:
                layer = Pooling(
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    layer_depth=i,
                    representation_size=current_resolution
                )
                current_resolution = (current_resolution - layer_config['kernel_size']) // layer_config['stride'] + 1
                num_consecutive_fc_layers = 0

            elif layer_type == FULLY_CONNECTED:
                # If first FC layer, flatten input
                if num_consecutive_fc_layers == 0:
                    in_features = current_channels * current_resolution * current_resolution
                    self.layers.append(nn.Flatten())
                else:
                    in_features = current_channels

                layer = FullyConnected(
                    in_features=in_features,
                    num_neurons=layer_config['num_neurons'],
                    layer_depth=i,
                    representation_size=current_resolution
                )
                current_channels = layer_config['num_neurons']
                current_resolution = 1
                num_consecutive_fc_layers += 1

            else:
                # If not flattened, flatten
                if num_consecutive_fc_layers == 0:
                    in_features = current_channels * current_resolution * current_resolution
                    self.layers.append(nn.Flatten())
                else:
                    in_features = current_channels

                layer = Termination(
                    in_features=in_features,
                )


            self.layers.append(layer)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
