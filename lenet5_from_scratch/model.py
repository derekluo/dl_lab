import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self, num_classes=10):
        super(LeNet5, self).__init__()
        
        # First convolutional block
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Second convolutional block
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        # Uncomment prints to debug dimensions
        # print(f"Input shape: {x.shape}")
        
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        # print(f"After pool1: {x.shape}")
        
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print(f"After pool2: {x.shape}")
        
        x = torch.flatten(x, 1)
        # print(f"After flatten: {x.shape}")
        
        x = self.relu3(self.fc1(x))
        x = self.relu4(self.fc2(x))
        x = self.fc3(x)
        
        return x 