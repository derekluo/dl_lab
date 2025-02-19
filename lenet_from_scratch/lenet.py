import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # First fully connected layer
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        # Second fully connected layer
        self.fc2 = nn.Linear(120, 84)
        # Output layer
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # First conv layer followed by ReLU and max pooling
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        # Second conv layer followed by ReLU and max pooling
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        # Flatten the tensor for the fully connected layers
        x = x.view(-1, 16 * 5 * 5)
        # First fully connected layer with ReLU
        x = F.relu(self.fc1(x))
        # Second fully connected layer with ReLU
        x = F.relu(self.fc2(x))
        # Output layer
        x = self.fc3(x)
        return x 