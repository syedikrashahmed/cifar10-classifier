import argparse
import json
import os
import torch
import torch.nn as nn
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix)

from models.mlp import MLP
from models.cnn import CNN
from models.resnet18 import ResNet18
from utils import get_dataloaders

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

def evaluate_checkpoint(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    config = checkpoint["config"]
    experiment_name = checkpoint["experiment_name"]

    print(f"Experiment: {experiment_name}")
    print("\nConfiguration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print(f"\nCheckpoint validation accuracy: {checkpoint['val_accuracy']:.2f}%")

    model = get_model(config["model"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    _, _, test_loader = get_dataloaders(batch_size=config["batch_size"], model_type=config["model_type"], augment=config["augment"], seed=config["seed"])
    model.eval()

    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predictions.cpu().tolist())

    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
    recall = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
    f1 = f1_score(all_labels, all_predictions, average="macro", zero_division=0)
    matrix = confusion_matrix(all_labels, all_predictions)

    print("\nTest Results")
    print("-" * 40)

    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nConfusion Matrix:")
    print(matrix)

    os.makedirs("results", exist_ok=True)

    results = {
        "experiment": experiment_name,
        "validation_accuracy": checkpoint["val_accuracy"],
        "test_accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": matrix.tolist(),
    }

    result_path = f"results/{experiment_name}.json"

    with open(result_path, "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nResults saved to: {result_path}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint)