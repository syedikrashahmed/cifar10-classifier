import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

def get_dataloaders(batch_size=64):
    # Convert images to PyTorch tensors
    transform = transforms.ToTensor()

    # Download/load the training dataset
    train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=transform
    )

    # Download/load the testing dataset
    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=transform
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader
