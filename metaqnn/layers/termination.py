import torch.nn as nn


class Termination(nn.Module):

    def __init__(self, in_features, termination_type):
        super().__init__()


        if termination_type == 'softmax':
            self.termination = nn.Softmax(dim=in_features)
        else:
            self.termination = nn.AdaptiveAvgPool2d(output_size=1)


    def forward(self, x):
        x = self.termination(x)

        return x

