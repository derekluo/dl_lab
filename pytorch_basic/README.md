# PyTorch Basics

This directory contains a Jupyter notebook, `my_model.ipynb`, that serves as a hands-on introduction to the fundamental concepts of building neural networks in PyTorch.

It is the recommended starting point for anyone new to PyTorch before diving into the more complex, from-scratch model implementations in the other project directories.

## Contents

The `my_model.ipynb` notebook covers the following key concepts through a simple, executable example:

-   **Defining a Model**: How to create a custom neural network by subclassing `torch.nn.Module`.
-   **Initializing Layers**: How to define layers like `nn.Linear` and `nn.ReLU` within the model's `__init__` method.
-   **Weight Initialization**: Demonstrates how to apply custom weight initializations, such as Xavier Uniform, to linear layers.
-   **The `forward` Method**: Implementing the `forward` pass to define how input data flows through the network.
-   **Using `nn.Sequential`**: How to use `nn.Sequential` to chain layers together for a cleaner model definition.
-   **Model Instantiation**: Creating an instance of the model and running a sample tensor through it.
-   **Inspecting Parameters**: How to view the model's layers and named parameters using `model.named_parameters`.
-   **Saving and Loading**:
    -   Accessing the model's state dictionary with `model.state_dict()`.
    -   Saving the entire model object with `torch.save()`.
    -   Loading the model back into memory with `torch.load()`.

## Usage

### 1. Prerequisites
Install the required Python libraries.
```bash
pip install torch jupyter
```

### 2. Running the Notebook
To explore the examples, start the Jupyter Notebook server:
```bash
jupyter notebook
```
Then, open the `my_model.ipynb` file in your browser and run the cells.

## Learning Path

This notebook is the first step in the learning journey for this repository. After completing it, you will be well-prepared to understand the structure and concepts used in the more advanced projects.

A recommended learning path is:
1.  **`pytorch_basic` (This project)**: Understand the fundamentals of `nn.Module`.
2.  **`lenet_from_scratch`**: Apply these fundamentals to build your first simple CNN.
3.  **`backprop_from_scratch`**: Go deeper to understand the underlying mechanics of what PyTorch's `autograd` does automatically.
4.  **Advanced Architectures**: Move on to more complex models like `transformer_from_scratch` and `diffusion_from_scratch`.