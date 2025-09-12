# LeNet-5 From Scratch

This project provides a from-scratch implementation of the classic LeNet-5 convolutional neural network (CNN) architecture using PyTorch. LeNet-5, designed by Yann LeCun in 1998, was one of the earliest and most influential CNNs, particularly for its success in handwritten digit recognition.

This implementation is intended for educational purposes to demonstrate a foundational CNN architecture.

## Architecture

The LeNet-5 architecture is composed of two sets of convolutional and average pooling layers, followed by a flattening convolutional layer, and then two fully-connected layers.

-   **Input**: 32×32 grayscale images.
-   **C1**: Convolutional layer with 6 feature maps and a 5×5 kernel.
-   **S2**: Subsampling (average pooling) layer with a 2×2 kernel.
-   **C3**: Convolutional layer with 16 feature maps and a 5×5 kernel.
-   **S4**: Subsampling (average pooling) layer with a 2×2 kernel.
-   **C5**: Convolutional layer with 120 feature maps and a 5×5 kernel.
-   **F6**: Fully connected layer with 84 units.
-   **Output**: A final fully connected layer with 10 units for the 10 digit classes.

*Note: This implementation uses MaxPooling as is common in modern interpretations, and the input is resized to 32x32 to match the original architecture's expectations.*

## File Structure

-   `lenet.py`: Contains the complete implementation of the LeNet-5 model architecture.
-   `train.py`: A script for training and evaluating the LeNet-5 model on the MNIST dataset.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch torchvision
```

### 2. Training the Model
To train the model, simply run the `train.py` script:
```bash
python train.py
```
This script will:
-   Automatically download the MNIST dataset.
-   Preprocess the data (resize images to 32x32 and normalize).
-   Initialize the LeNet-5 model, optimizer, and loss function.
-   Run the training loop for 10 epochs, printing the training loss and test accuracy after each epoch.
-   Save the trained model weights to `lenet_mnist.pth`.

## Key Features

-   **Pure PyTorch Implementation**: The model is built from scratch using fundamental PyTorch modules.
-   **Classic CNN Architecture**: A faithful implementation of the pioneering LeNet-5 model.
-   **MNIST Dataset**: Automatically downloads and trains on the standard handwritten digit dataset.
-   **GPU Acceleration**: Automatically uses a CUDA-enabled GPU if available, otherwise falls back to the CPU.

## Learning Objectives

-   Understand the structure of a pioneering convolutional neural network.
-   Learn how to implement a simple yet effective CNN for image classification in PyTorch.
-   See the fundamental pattern of alternating convolution and pooling layers.
-   Gain insight into a standard training and evaluation pipeline for a classification task.