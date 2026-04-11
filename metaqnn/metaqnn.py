import torch.nn as nn
from metaqnn.layers.convolution import Convolution
from metaqnn.layers.pooling import Pooling
from metaqnn.layers.fully_connected import FullyConnected
from metaqnn.layers.termination import Termination


class MetaQNN(nn.Module):

    def __init__(self, layer_configs):
        super().__init__()

        self.layers = []
        
        for i, layer_config in enumerate(layer_configs):
            if layer_config['layer_type'] == 'convolution':
                layer = Convolution(
                    in_channels=1, 
                    out_channels=layer_config['out_channels'],
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    layer_depth=i,
                    representation_size=234234234
                )
            elif layer_config['layer_type'] == 'pooling':
                layer = Pooling(
                    kernel_size=layer_config['kernel_size'],
                    stride=layer_config['stride'],
                    layer_depth=i,
                    representation_size=234234234
                )
            elif layer_config['layer_type'] == 'fully_connected':
                layer = FullyConnected(
                    in_features=1,
                    num_neurons=layer_config['num_neurons'],
                    layer_depth=i,
                    representation_size=234234234
                )
            else:
                layer = Termination(
                    in_features=1,
                    termination_type=layer_config['termination_type']
                )

            self.layers.append(layer)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        return x
