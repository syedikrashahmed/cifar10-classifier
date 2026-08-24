import torch.nn as nn
from torchvision import models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, feature_extract=True):
        super().__init__()
        weights = (models.ResNet18_Weights.DEFAULT if pretrained else None)
        self.model = models.resnet18(weights=weights)

        if feature_extract:
            # Freeze every layer
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            # Train only the classifier
            for parameter in self.model.fc.parameters():
                parameter.requires_grad = True
                
        else:
            # Freeze everything first
            for parameter in self.model.parameters():
                parameter.requires_grad = False

            # Unfreeze layer4
            for parameter in self.model.layer4.parameters():
                parameter.requires_grad = True

            # Unfreeze classifier
            for parameter in self.model.fc.parameters():
                parameter.requires_grad = True

        num_features = self.model.fc.in_features # Replace classifier
        self.model.fc = nn.Linear(
            in_features=num_features,
            out_features=10
        )

        for parameter in self.model.fc.parameters(): # Make classifier trainable
            parameter.requires_grad = True

    def forward(self, x):
        return self.model(x)