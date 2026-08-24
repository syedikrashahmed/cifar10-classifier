import time
import torch

class Trainer:
    def __init__(self, model, criterion, optimizer, device, checkpoint_path=None):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.best_accuracy = 0.0
        self.history = {"train_loss": [], "train_accuracy": [], "val_accuracy": []}

    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100.0 * correct / total
        return epoch_loss, epoch_accuracy

    @torch.no_grad()
    def evaluate(self, test_loader, return_predictions=False):
        self.model.eval()
        correct = 0
        total = 0

        all_predictions = []
        all_labels = []
        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if return_predictions:
                all_predictions.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())

        accuracy = 100.0 * correct / total

        if return_predictions:
            return accuracy, all_labels, all_predictions

        return accuracy

    def save_checkpoint(self, epoch, val_accuracy, experiment_name, config):
        if self.checkpoint_path is None:
            return

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_accuracy": val_accuracy,
            "experiment_name": experiment_name,
            "config": config,
            "history": self.history
        }

        torch.save(checkpoint, self.checkpoint_path)

    def fit(self, train_loader, val_loader, epochs, experiment_name, config):
        start_time = time.time()

        for epoch in range(epochs):
            train_loss, train_accuracy = self.train_epoch(train_loader)
            val_accuracy = self.evaluate(val_loader)

            self.history["train_loss"].append(train_loss)
            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)

            if val_accuracy > self.best_accuracy:
                self.best_accuracy = val_accuracy

                self.save_checkpoint(
                    epoch=epoch + 1,
                    val_accuracy=val_accuracy,
                    experiment_name=experiment_name,
                    config=config
                )

            print(
                f"Epoch [{epoch + 1}/{epochs}] "
                f"Train Loss: {train_loss:.4f} "
                f"Train Acc: {train_accuracy:.2f}% "
                f"Val Acc: {val_accuracy:.2f}%"
            )

        total_time = time.time() - start_time

        print(f"\nBest Validation Accuracy: {self.best_accuracy:.2f}%")
        print(f"Training Time: {total_time:.2f} seconds")
        return self.history