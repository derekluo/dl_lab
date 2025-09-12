# AlexNet From Scratch

This project provides a from-scratch implementation of the AlexNet convolutional neural network (CNN) architecture using PyTorch. AlexNet was a groundbreaking model that won the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and is considered a major catalyst for the deep learning revolution in computer vision.

This implementation is intended for educational purposes to demonstrate the core components and structure of AlexNet.

## Architecture

The AlexNet architecture consists of five convolutional layers followed by three fully connected layers.

-   **Input**: 227×227×3 RGB images.
-   **5 Convolutional Layers**: Feature extraction using stacked convolutional filters, ReLU activations, and max pooling.
-   **3 Fully Connected Layers**: Classification layers that process the extracted features, with dropout for regularization.
-   **Output Layer**: A final layer with a configurable number of classes (defaulting to 1000 for ImageNet).

### Layer Details:
1.  **Conv1**: 96 kernels (11×11), stride 4 → ReLU → MaxPool
2.  **Conv2**: 256 kernels (5×5), stride 1 → ReLU → MaxPool
3.  **Conv3**: 384 kernels (3×3), stride 1 → ReLU
4.  **Conv4**: 384 kernels (3×3), stride 1 → ReLU
5.  **Conv5**: 256 kernels (3×3), stride 1 → ReLU → MaxPool
6.  **FC1**: 4096 neurons → ReLU → Dropout
7.  **FC2**: 4096 neurons → ReLU → Dropout
8.  **FC3**: Output layer (num_classes neurons)

## File Structure

-   `alexnet.py`: Contains the complete implementation of the AlexNet model architecture.
-   `train.py`: A script for training the AlexNet model. It includes data loading, preprocessing, and the main training loop.
-   `inference.py`: A script for running inference on new images using a trained AlexNet model. It provides top-k predictions.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch torchvision pillow numpy matplotlib
```

### 2. Training the Model
The `train.py` script is set up to train the model on a dataset structured like the ImageNet dataset. You will need to provide your own dataset and update the path in the script.

To start training:
```bash
python train.py
```
This will train the model and save the learned weights to a file named `alexnet_model.pth`.

### 3. Running Inference
Once the model is trained, you can use `inference.py` to classify a new image.

Update the `model_path` and `image_path` variables in the script, then run:
```bash
python inference.py
```
The script will load the trained model, process the image, and print the top-5 predicted classes along with their confidence scores.

## Key Features

-   **Pure PyTorch Implementation**: The model is built from scratch using fundamental PyTorch modules, without relying on pre-built models.
-   **Configurable Architecture**: The number of output classes can be easily configured to adapt the model to different datasets.
-   **GPU Acceleration**: The code automatically leverages a CUDA-enabled GPU or Apple Silicon (MPS) if available, falling back to the CPU otherwise.
-   **Standard Preprocessing**: Implements the standard ImageNet-style preprocessing (resizing, cropping, and normalization).

## Learning Objectives

-   Understand the architecture of a deep convolutional neural network.
-   Learn how to implement a classic CNN model in PyTorch.
-   See how key components like ReLU, MaxPooling, and Dropout are used in practice.
-   Gain insight into a standard training and inference pipeline for image classification.