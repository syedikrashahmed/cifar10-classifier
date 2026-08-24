import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

from models.mlp import MLP
from models.cnn import CNN
from models.resnet18 import ResNet18
from utils import get_dataloaders
from training.trainer import Trainer
from configs.experiments import EXPERIMENTS

def get_model(model_name):
    if model_name == "mlp":
        return MLP()

    if model_name == "cnn":
        return CNN()

    if model_name == "resnet_feature":
        return ResNet18(pretrained=True, feature_extract=True)

    if model_name == "resnet_finetune":
        return ResNet18(pretrained=True, feature_extract=False)
    
    raise ValueError(f"Unknown model: {model_name}")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_optimizer(model, config):
    if config["optimizer"] == "adam":
        return optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], weight_decay=config["weight_decay"])

    if config["optimizer"] == "sgd":
        return optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config["lr"], momentum=0.9, weight_decay=config["weight_decay"])

    raise ValueError(f"Unknown optimizer: {config['optimizer']}")

def main(experiment_name):

    if experiment_name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment_name}")

    config = EXPERIMENTS[experiment_name]
    set_seed(config["seed"])
    print(f"Experiment: {experiment_name}")
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    train_loader, val_loader, test_loader = get_dataloaders(batch_size=config["batch_size"], model_type=config["model_type"], augment=config["augment"], seed=config["seed"])

    model = get_model(config["model"])
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = get_optimizer(model, config)

    os.makedirs("checkpoints", exist_ok=True)
    checkpoint_path = (f"checkpoints/{experiment_name}.pth")

    trainer = Trainer(model=model, criterion=criterion, optimizer=optimizer, device=device, checkpoint_path=checkpoint_path)
    history = trainer.fit(train_loader, val_loader, epochs=config["epochs"], experiment_name=experiment_name, config=config)

    print("\nExperiment complete.")
    print(f"Best Validation Accuracy: {trainer.best_accuracy:.2f}%")
    print(f"Checkpoint: {checkpoint_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    args = parser.parse_args()
    main(args.experiment)