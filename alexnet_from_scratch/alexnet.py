import torch
import torch.nn as nn

class AlexNet(nn.Module):
    """
    Implementation of AlexNet architecture from the 2012 paper
    "ImageNet Classification with Deep Convolutional Neural Networks"
    by Alex Krizhevsky et al.
    """
    def __init__(self, num_classes=1000):
        """
        Initialize AlexNet model
        Args:
            num_classes (int): Number of output classes (default: 1000 for ImageNet)
        """
        super(AlexNet, self).__init__()

        # Feature extraction layers - processes input images through conv layers
        self.features = nn.Sequential(
            # Layer 1: First conv layer with 96 filters of size 11x11, stride 4
            # Input: 227x227x3, Output: 55x55x96 
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),  # Activation function
            nn.MaxPool2d(kernel_size=3, stride=2),  # Pooling reduces to 27x27x96

            # Layer 2: Second conv layer with 256 filters of size 5x5
            # Input: 27x27x96, Output: 27x27x256
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Reduces to 13x13x256

            # Layer 3: Third conv layer with 384 filters of size 3x3
            # Input: 13x13x256, Output: 13x13x384
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 4: Fourth conv layer with 384 filters of size 3x3
            # Input: 13x13x384, Output: 13x13x384
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Layer 5: Fifth conv layer with 256 filters of size 3x3
            # Input: 13x13x384, Output: 13x13x256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # Reduces to 6x6x256
        )

        # Classification layers - fully connected layers for classification
        self.classifier = nn.Sequential(
            # Layer 6: First fully connected layer
            nn.Dropout(p=0.5),  # Dropout for regularization
            nn.Linear(256 * 6 * 6, 4096),  # Input flattened from 6x6x256
            nn.ReLU(inplace=True),

            # Layer 7: Second fully connected layer
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            # Output layer: Final classification layer
            nn.Linear(4096, num_classes),  # Outputs class probabilities
        )

    def forward(self, x):
        """
        Forward pass of the network
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 227, 227)
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, num_classes)
        """
        x = self.features(x)  # Pass through convolutional layers
        x = torch.flatten(x, 1)  # Flatten the features for fully connected layers
        x = self.classifier(x)  # Pass through classification layers
        return x

# Example usage
if __name__ == "__main__":
    # Create model instance
    model = AlexNet(num_classes=1000)

    # Create a sample input tensor (batch_size, channels, height, width)
    sample_input = torch.randn(1, 3, 227, 227)

    # Get output
    output = model(sample_input)
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")