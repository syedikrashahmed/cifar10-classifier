import torch
import torch.nn as nn
import torch.optim as optim

# from models.cnn import CNN
from models.mlp import MLP
from utils import get_dataloaders

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

train_loader, test_loader = get_dataloaders(batch_size=64)
# model = CNN().to(device)
model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001,
    weight_decay=1e-4
)
print(model)

best_accuracy = 0.0
num_epochs = 10
for epoch in range(num_epochs):
    model.train()

    running_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, dim=1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()
    
    train_loss = running_loss / len(train_loader)
    train_accuracy = 100 * train_correct / train_total

    #evaluate
    model.eval()

    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            _, predicted = torch.max(outputs, dim=1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    test_accuracy = 100 * test_correct / test_total

    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(
            model.state_dict(),
            "saved_models/mlp.pth"
        ) 
        print("Model saved to saved_models/mlp.pth")

    print(
        f"Epoch [{epoch+1}/{num_epochs}] "
        f"Train Loss: {train_loss:.4f} "
        f"Train Acc: {train_accuracy:.2f}% "
        f"Test Acc: {test_accuracy:.2f}%"
    )  

print(f"Best Test Accuracy: {best_accuracy:.2f}%")