import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from sklearn.model_selection import train_test_split
import torchvision

from metaqnn.config.train_config import DEVICE, IMAGE_SIZE
from metaqnn.metaqnn import MetaQNN
from metaqnn.state_actions import parse_state


def validate(model, val_loader):
    model.eval()
    num_correct = 0
    num_total = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            preds = model(inputs)

            pred_classes = preds.argmax(dim=1)

            num_correct += (pred_classes == labels).sum().item()
            num_total += len(labels)

    accuracy = num_correct / num_total

    print(f'Model accuracy: {accuracy}')

    return accuracy


def train(model, num_epochs, train_loader, val_loader, loss_func, optimizer, scheduler):
    # Training loop
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}")
        model.train()
        for inputs, labels in tqdm(train_loader):
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            preds = model(inputs)

            loss = loss_func(preds, labels)
            loss.backward()
            optimizer.step()

        # Evaluate model on first epoch
        if epoch == 0:
            accuracy = validate(model, val_loader)

            # Model is similar to random chance
            # TODO: restart it a few times if this happens
            if accuracy <= 0.15:
                return accuracy

        # Step scheduler
        if scheduler:
            scheduler.step()


    # Evaluate model
    accuracy = validate(model, val_loader)

    return accuracy


def initialize_datasets():
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

    return train_loader, val_loader, test_loader


def create_model(action_sequence):
    # Start from 1 to ignore the 'None' initial state
    model = MetaQNN(layer_configs=action_sequence[1:], input_size=IMAGE_SIZE, input_channels=3)
    model = model.to(DEVICE)

    return model


def get_optimizer(model):
    adamW_params = {
        "lr": 1e-3,
        "weight_decay": 1e-2,
        "betas": (0.9, 0.999),
        "eps": 1e-8
    }
    optimizer = optim.AdamW(model.parameters(), **adamW_params)

    return optimizer


def get_scheduler(optimizer):
    # Multiply LR by 0.2 every 5 epochs
    return optim.lr_scheduler.StepLR(optimizer, 5, 0.2)
