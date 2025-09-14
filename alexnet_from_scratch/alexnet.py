import torch
import torch.nn as nn


class AlexNet(nn.Module):
    """AlexNet architecture (Krizhevsky et al., 2012)

    Revolutionary CNN that won ImageNet 2012 competition.
    Input: 227x227x3 images
    """

    def __init__(self, num_classes=1000):
        super().__init__()

        # Feature extraction: 5 convolutional layers
        self.features = nn.Sequential(
            # Conv1: 227x227x3 -> 55x55x96 -> 27x27x96
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2: 27x27x96 -> 27x27x256 -> 13x13x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3: 13x13x256 -> 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4: 13x13x384 -> 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5: 13x13x384 -> 13x13x256 -> 6x6x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # Classification: 3 fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


if __name__ == "__main__":
    # Test model
    model = AlexNet(num_classes=1000)
    x = torch.randn(1, 3, 227, 227)
    output = model(x)

    print(f"Input:  {x.shape}")
    print(f"Output: {output.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")