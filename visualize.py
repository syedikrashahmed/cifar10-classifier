import argparse
import os
import json
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.mlp import MLP
from models.cnn import CNN
from models.resnet18 import ResNet18
from utils import get_dataloaders

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

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

def plot_training_curves(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    experiment_name = checkpoint["experiment_name"]
    history = checkpoint["history"]
    os.makedirs("results", exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 8))
    plt.subplot(2, 1, 1)
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{experiment_name} - Training Curves")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    plt.plot(epochs, history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    training_path = (f"results/{experiment_name}_training_curves.png")
    plt.savefig(training_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {training_path}")

def plot_confusion_matrix(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    experiment_name = checkpoint["experiment_name"]
    result_path = (f"results/{experiment_name}.json")

    with open(result_path, "r") as f:
        results = json.load(f)

    matrix = np.array(results["confusion_matrix"])
    plt.figure(figsize=(10, 8))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"{experiment_name} - Confusion Matrix")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, matrix[i, j], ha="center", va="center")
    plt.tight_layout()
    matrix_path = (f"results/{experiment_name}_confusion_matrix.png")
    plt.savefig(matrix_path)
    plt.close()
    print(f"Saved: {matrix_path}")


def plot_misclassified_images(checkpoint_path, max_images=16):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    experiment_name = checkpoint["experiment_name"]
    config = checkpoint["config"]
    model = get_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    _, _, test_loader = get_dataloaders(batch_size=config["batch_size"], model_type=config["model_type"], augment=config["augment"], seed=config["seed"])

    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)

            predictions = torch.argmax(outputs, dim=1)

            for image, label, prediction in zip(images, labels, predictions):
                if label != prediction:
                    misclassified.append((image.cpu(), label.cpu().item(), prediction.cpu().item()))
                    if len(misclassified) >= max_images:
                        break

            if len(misclassified) >= max_images:
                break

    rows = 4
    cols = 4

    plt.figure(figsize=(12, 12))
    for i, (image, label, prediction) in enumerate(misclassified):
        image = image.permute(1, 2, 0)
        if config["model_type"] == "resnet":
            mean = torch.tensor([0.4914, 0.4822, 0.4465])
            std = torch.tensor([0.2023, 0.1994, 0.2010])
            image = image * std + mean

        image = torch.clamp(image, 0, 1)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(image)
        plt.title(f"True: {CLASS_NAMES[label]}\n Pred: {CLASS_NAMES[prediction]}")
        plt.axis("off")

    plt.tight_layout()

    misclassified_path = (f"results/{experiment_name}_misclassified.png")
    plt.savefig(misclassified_path, bbox_inches="tight")
    plt.close()
    print(f"Saved: {misclassified_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    plot_training_curves(args.checkpoint)
    plot_confusion_matrix(args.checkpoint)
    plot_misclassified_images(args.checkpoint)