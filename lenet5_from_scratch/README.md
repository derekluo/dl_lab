# LeNet-5 From Scratch (Extended Version)

This project provides an enhanced, from-scratch implementation of the LeNet-5 architecture using PyTorch. It builds upon the classic model by incorporating modern deep learning practices and a more complete project structure, including training, custom test generation, and inference pipelines.

This implementation is designed to be a comprehensive, educational example of a full image classification workflow.

## Architecture

This implementation uses a modernized version of the LeNet-5 architecture.

-   **Input**: 32×32 grayscale images.
-   **Convolutional Blocks**: Two blocks of `Conv2d` -> `ReLU` -> `MaxPool2d` for feature extraction.
-   **Fully Connected Layers**: Three `Linear` layers with `ReLU` activations for classification.
-   **Modern Practices**:
    -   Uses `ReLU` activation instead of the original `tanh`.
    -   Uses `MaxPool2d` instead of average pooling.
    -   Employs the `Adam` optimizer for more efficient training.

## File Structure

-   `model.py`: Contains the implementation of the LeNet-5 model architecture.
-   `train.py`: The main training script. It handles data loading, training, evaluation, and plots the training loss and test accuracy curves.
-   `generate_test_digits.py`: A utility script to generate custom test images of digits (0-9) using the PIL library.
-   `inference.py`: A script to run inference on the generated test digits or any other image using the trained model.
-   `data/`: Directory where the MNIST dataset is stored.
-   `test_digits/`: Directory where the generated test images are saved.
-   `README.md`: This documentation file.

## Usage

The project is designed to be run in a sequence: generate test data, train the model, and then run inference.

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch torchvision matplotlib pillow numpy
```

### 2. Generate Custom Test Data (Optional)
You can create your own simple test images of digits 0-9 by running:
```bash
python generate_test_digits.py
```
This will create a `test_digits/` directory and save a `digit_N.png` image for each digit `N`.

### 3. Training the Model
To train the model on the MNIST dataset, run:
```bash
python train.py
```
This script will:
-   Automatically download the MNIST dataset.
-   Train the LeNet-5 model for 10 epochs.
-   Print the loss and accuracy for each epoch.
-   Save the trained model weights to `lenet5_model.pth`.
-   Save a plot of the training curves to `training_results.png`.

### 4. Running Inference
After training, you can test the model's performance on the custom-generated digits:
```bash
python inference.py
```
This will load the trained `lenet5_model.pth` and print the model's prediction for each of the generated digit images, indicating whether the prediction was correct.

## Key Features

-   **Complete Workflow**: Provides a full, end-to-end example from training to inference.
-   **Training Visualization**: Automatically generates and saves plots for training loss and test accuracy, which is crucial for monitoring model performance.
-   **Modern Optimizer**: Uses the `Adam` optimizer, which generally leads to faster convergence than traditional SGD.
-   **Custom Test Set Generation**: Includes a script to create a synthetic test set, allowing for simple, intuitive model validation.
-   **Modular and Clear Code**: The logic is separated into distinct files for the model, training, and inference, making the code easy to understand and extend.

## Learning Objectives

-   Understand how to structure a deep learning project with separate scripts for different tasks.
-   Learn to implement a full training pipeline that includes performance monitoring and visualization.
-   See how to save and load a trained model for inference.
-   Gain experience with a simple computer vision pipeline, from data preparation to model evaluation on custom images.