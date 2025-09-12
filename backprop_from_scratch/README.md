# Backpropagation From Scratch

This project provides a detailed, from-scratch implementation of a neural network and its backpropagation algorithm using only NumPy. It is designed as an educational tool to offer a deep and intuitive understanding of the core mechanics of how neural networks learn.

The implementation is transparent, with every step of the forward and backward passes exposed for analysis. The project includes extensive demonstrations and visualizations to demystify concepts like gradient descent, activation functions, and loss landscapes.

## File Structure

-   `model.py`: Contains the core neural network components, including `Dense` layers, `Activation` functions (ReLU, Sigmoid, Tanh), and `Loss` functions, all implemented manually.
-   `train.py`: Provides scripts to train the neural network on various tasks, such as binary classification and regression, and includes functions for plotting results.
-   `demo.py`: A detailed, step-by-step demonstration of the backpropagation process on the classic XOR problem, showing the exact calculations at each layer.
-   `visualize.py`: A collection of powerful visualization tools to generate diagrams of the network architecture, activation functions, loss landscapes, and gradient flow.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries. `networkx` is used for some visualizations.
```bash
pip install numpy matplotlib scikit-learn networkx
```

### 2. Run Training Demonstrations
To see the network train on different tasks (e.g., classification and regression), run:
```bash
python train.py
```
This will train the models and generate plots like `training_results.png` and `classification_results.png`.

### 3. See a Detailed Backpropagation Walkthrough
To understand the exact mathematical operations during backpropagation, run the demo script:
```bash
python demo.py
```
This will print a step-by-step breakdown of the forward and backward passes for the XOR problem.

### 4. Generate Visualizations
To generate a comprehensive set of visualizations, run:
```bash
python visualize.py
```
This will produce several insightful plots, including:
-   `network_architecture.png`: A diagram of the neural network structure.
-   `activation_functions.png`: A plot of different activation functions and their derivatives.
-   `loss_landscape.png`: A 3D and 2D visualization of the loss surface.
-   `gradient_flow.png`: A bar chart showing the magnitude of gradients in each layer.

## Key Features

-   **Manual Implementation**: Every component, from layers to loss functions, is built from scratch, providing full transparency into the inner workings of a neural network.
-   **Step-by-Step Analysis**: The `demo.py` script offers a granular, print-based walkthrough of the forward and backward passes, making it easy to trace the flow of data and gradients.
-   **Comprehensive Visualizations**: The project includes a rich set of tools to visualize:
    -   Network architecture.
    -   Activation functions and their derivatives (highlighting the vanishing gradient problem).
    -   3D and 2D loss landscapes to build an intuition for gradient descent.
    -   The magnitude of gradients and weight updates across different layers.
-   **Classic Problem Demonstrations**: Includes implementations for solving the non-linearly separable XOR problem and fitting a sine wave for regression.
-   **Educational Focus**: The code is heavily commented and structured for clarity and learning, not for performance.

## Learning Objectives

-   Gain a fundamental understanding of the **forward and backward propagation** algorithms.
-   Learn how the **chain rule** is applied to compute gradients in a multi-layer network.
-   Understand the role of **activation functions** and their derivatives in training.
-   Visualize how **gradient descent** navigates a loss landscape to minimize error.
-   Build an intuition for concepts like **vanishing gradients** and the effect of **learning rates**.
