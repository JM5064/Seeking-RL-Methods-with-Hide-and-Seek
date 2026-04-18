import torch

# For CIFAR-10
IMAGE_SIZE = 32
NUM_EPOCHS = 15

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() 
    else "mps" if torch.backends.mps.is_available() 
    else "cpu"
)