# LeNet From Scratch

This project implements the classic LeNet-5 convolutional neural network architecture from scratch using PyTorch. LeNet-5 was designed by Yann LeCun in 1998 and was one of the first successful applications of convolutional neural networks to handwritten digit recognition.

## Architecture

LeNet-5 consists of:
- **Input Layer**: 32×32 grayscale images
- **Conv Layer 1**: 6 feature maps, 5×5 kernel, followed by ReLU and max pooling (2×2)
- **Conv Layer 2**: 16 feature maps, 5×5 kernel, followed by ReLU and max pooling (2×2)
- **Fully Connected Layers**: 120 → 84 → 10 (for 10 digit classes)

## Files

- `lenet.py` - Complete LeNet-5 model implementation
- `train.py` - Training script with MNIST dataset

## Usage

### Training the Model
```bash
python train.py
```

This will:
- Download the MNIST dataset automatically
- Train the LeNet model for 10 epochs
- Display training progress with loss and accuracy
- Save the trained model as `lenet_mnist.pth`

### Model Architecture Details

```python
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # First convolutional layer: 1 input channel (grayscale), 6 output channels, 5x5 kernel
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        # Second convolutional layer: 6 input channels, 16 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        # Fully connected layers
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
```

## Key Features

- **Pure PyTorch Implementation**: No pre-built CNN models used
- **MNIST Dataset**: Automatic download and preprocessing
- **Training Monitoring**: Real-time loss and accuracy tracking
- **GPU Support**: Automatic device detection (CUDA/MPS/CPU)
- **Model Persistence**: Saves trained weights for later use

## Expected Results

- **Training Accuracy**: ~99%
- **Test Accuracy**: ~98%
- **Training Time**: ~2-3 minutes on CPU, <1 minute on GPU

## Learning Objectives

This implementation helps understand:
- Convolutional layer operations and feature extraction
- Max pooling and spatial dimension reduction
- Fully connected layers for classification
- Training loop structure and optimization
- Loss functions and backpropagation in CNNs

## Requirements

```bash
pip install torch torchvision
```

## Historical Context

LeNet-5 was revolutionary because it:
- Introduced the concept of hierarchical feature extraction
- Demonstrated the effectiveness of CNNs for computer vision
- Established the pattern of convolution → pooling → fully connected layers
- Proved that neural networks could achieve high accuracy on real-world tasks

## Comparison with Modern CNNs

While LeNet-5 was groundbreaking for its time, modern CNNs differ in:
- **Depth**: Modern networks have many more layers
- **Activations**: ReLU is now preferred over tanh/sigmoid
- **Regularization**: Batch normalization and dropout are standard
- **Architecture**: Skip connections and attention mechanisms are common

This implementation serves as an excellent starting point for understanding the fundamentals before moving to more complex architectures like AlexNet, ResNet, or modern transformers.
