"""The purpose of this file is to see whether the training works, will prolly be deleted later"""

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import torchvision

from metaqnn.train import train, get_scheduler
from metaqnn.metaqnn import MetaQNN
from metaqnn.config.rl_config import *
from metaqnn.config.train_config import DEVICE


def main():
    # Define transformations
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]),
    ])

    # Fetch datasets and create train/val/test split
    cifar10_train = torchvision.datasets.CIFAR10(root='metaqnn/dataset', train=True, download=True, transform=transform)
    cifar10_test = torchvision.datasets.CIFAR10(root='metaqnn/dataset', train=False, download=True, transform=transform)

    val_size = int(len(cifar10_train) * 0.1)
    cifar10_train, cifar10_val = train_test_split(cifar10_train, test_size=val_size, shuffle=True, random_state=5064)

    # Create dataloaders
    train_loader = DataLoader(dataset=cifar10_train, batch_size=128, shuffle=True, num_workers=1)
    val_loader = DataLoader(dataset=cifar10_val, batch_size=128, shuffle=False, num_workers=1)
    test_loader = DataLoader(dataset=cifar10_test, batch_size=128, shuffle=False, num_workers=1)


    layer_configs = [
        {'layer_type': 0, 'out_channels': 256, 'kernel_size': 3, 'layer_depth': 1, 'representation_size': 32},
        {'layer_type': 1, 'kernel_size': 5, 'stride': 3, 'layer_depth': 2, 'representation_size': 10},
        {'layer_type': 0, 'out_channels': 64, 'kernel_size': 5, 'layer_depth': 3, 'representation_size': 10},
        {'layer_type': 0, 'out_channels': 128, 'kernel_size': 5, 'layer_depth': 4, 'representation_size': 10},
        {'layer_type': 1, 'kernel_size': 2, 'stride': 2, 'layer_depth': 5, 'representation_size': 5},
        {'layer_type': 3}
    ]
    model = MetaQNN(layer_configs=layer_configs, input_size=32, input_channels=3)
    model = model.to(DEVICE)

    adamW_params = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = optim.AdamW(model.parameters(), **adamW_params)
    scheduler = get_scheduler(optimizer)

    train(model=model, num_epochs=15, train_loader=train_loader, val_loader=val_loader, 
          loss_func=nn.CrossEntropyLoss(), optimizer=optimizer, scheduler=scheduler)


if __name__ == "__main__":
    main()