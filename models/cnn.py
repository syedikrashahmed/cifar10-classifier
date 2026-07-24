import torch
import torch.nn as nn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(
            kernel_size=2, 
            stride=2
        )

        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            padding=1
        )

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 8 * 8, 512)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x): # initial shape of x: Batch × 3 × 32 × 32
        x = self.conv1(x) # Batch × 32 × 32 × 32
        x = self.relu(x)
        
        x = self.pool(x) # Batch × 32 × 16 × 16

        x = self.conv2(x) # Batch × 64 × 16 × 16
        x = self.relu(x)

        x = self.pool(x) # Batch × 64 × 8 × 8

        x = self.flatten(x) # Batch × 4096

        x = self.fc1(x) # Batch × 512
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x) # Batch × 10
        return x
    