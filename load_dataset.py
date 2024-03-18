import torch
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
from configs import *

# Set random seed for reproducibility
torch.manual_seed(RANDOM_SEED)

# Define transformations
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# Load Fashion MNIST dataset
train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)

# Define sizes for train, validation, and test sets
test_size = int(VALSPLIT * len(test_dataset))
val_size = len(test_dataset) - test_size

# Split train dataset into train and validation sets
test_dataset, val_dataset = random_split(test_dataset, [test_size, val_size])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
