import torch
import torch.nn as nn


class LeNet5(nn.Module):
    """LeNet-5 CNN architecture for digit classification"""

    def __init__(self, num_classes=10):
        super().__init__()

        # Feature extraction layers
        self.features = nn.Sequential(
            # Conv Block 1: 1x32x32 -> 6x28x28 -> 6x14x14
            nn.Conv2d(1, 6, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Block 2: 6x14x14 -> 16x10x10 -> 16x5x5
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Classification layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 6 * 6, 120),  # 16 channels, 6x6 feature maps
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x