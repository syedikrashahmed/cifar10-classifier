import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

def get_dataloaders(batch_size=64, model_type="cnn", augment=False, data_dir="./data", val_split=0.1, seed=42):
    if model_type not in ["cnn", "resnet"]:
        raise ValueError("model_type must be 'cnn' or 'resnet'")

    if model_type == "cnn":
        if augment:
            train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),])
        else:
            train_transform = transforms.Compose([transforms.ToTensor()])

        eval_transform = transforms.Compose([transforms.ToTensor()])

    else:
        if augment:
            train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)])
        else:
            train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)])

        eval_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)])

    full_train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_transform)
    val_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=eval_transform)
    val_size = int(len(full_train_dataset) * val_split)
    train_size = len(full_train_dataset) - val_size
    generator = torch.Generator().manual_seed(seed)

    train_dataset, _ = random_split(full_train_dataset, [train_size, val_size], generator=generator)
    _, val_dataset = random_split(val_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader