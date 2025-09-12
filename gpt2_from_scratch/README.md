# GPT-2 From Scratch

This project is a from-scratch implementation of a GPT-2 style transformer model for character-level language modeling and text generation. The model is built using PyTorch and is designed for educational purposes to demonstrate the core components of the GPT architecture.

## Architecture

The model is a decoder-only transformer, following the principles of the GPT-2 architecture.

-   **Token and Positional Embeddings**: The model uses a token embedding table to convert input characters into vectors and adds sinusoidal positional encodings to provide sequence information.
-   **Transformer Blocks**: The core of the model is a stack of `TransformerBlock` modules. Each block consists of:
    -   **Multi-Head Self-Attention**: A causal self-attention mechanism with multiple heads to allow the model to weigh the importance of different characters in the context. The causal mask ensures that predictions for a position can only depend on the preceding characters.
    -   **Feed-Forward Network**: A two-layer feed-forward network applied after the attention mechanism.
    -   **Residual Connections and Layer Normalization**: Both the attention and feed-forward sub-layers include residual connections and are preceded by layer normalization for stable training.
-   **Output Layer**: A final linear layer projects the output of the transformer blocks to the vocabulary size to produce logits for the next character prediction.

## File Structure

-   `model.py`: Contains the complete implementation of the GPT-2 style model, including all sub-modules like `Attention`, `MultiHeadAttention`, and `TransformerBlock`.
-   `train.py`: The main training script. It handles data loading, model initialization, the training loop, and loss estimation. It also integrates with `aim` for experiment tracking.
-   `inference.py`: A script for generating new text from a trained model given a starting prompt.
-   `show_params.py`: A utility script to display the names, shapes, and total number of parameters in the trained model.
-   `bin/`: Contains helper scripts for data preparation.
    -   `download.py`: Downloads the training dataset.
    -   `concatenate.sh`: Concatenates multiple data files into one.
-   `data/`: Directory where the training data is stored.
-   `models/`: Directory where the trained model weights are saved.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries. `aim` is used for experiment tracking.
```bash
pip install torch aim
```

### 2. Data Preparation
First, download and prepare the training data using the provided scripts.
```bash
# Download the dataset
python bin/download.py

# Concatenate the data files into a single file
bash bin/concatenate.sh
```
This will create a `scifi.txt` file in the `data/` directory.

### 3. Training the Model
To start training the model, run the training script:
```bash
python train.py
```
The script will:
-   Load the `scifi.txt` dataset.
-   Initialize the model, optimizer, and experiment tracking with `aim`.
-   Run the training loop, periodically printing the training and validation loss.
-   Save the final trained model to `models/model-scifi.pth`.

To visualize the training metrics, run the `aim` UI in a separate terminal:
```bash
aim up
```

### 4. Generating Text
Once the model is trained, you can generate new text using the `inference.py` script.
```bash
python inference.py
```
This will load the trained model and generate a 500-token completion for a hardcoded prompt. You can modify the `prompt` variable in the script to provide your own starting text.

### 5. Inspecting the Model
To see the details of the model's parameters, you can run:
```bash
python show_params.py
```
This will list all the parameter tensors in the model and their shapes, along with the total parameter count.

## Key Features

-   **Pure PyTorch Implementation**: The model is built from scratch using fundamental PyTorch modules.
-   **Character-Level Model**: The model operates on individual characters, making the vocabulary small and the tokenization process simple.
-   **Causal Self-Attention**: Implements a masked multi-head self-attention mechanism, which is the core of autoregressive transformer models.
-   **Experiment Tracking**: Integrated with `aim` to log and visualize training and validation losses.

## Learning Objectives

-   Understand the architecture of a decoder-only transformer like GPT-2.
-   Learn how to implement causal multi-head self-attention from scratch.
-   See how token and positional embeddings are combined in a transformer.
-   Gain insight into a standard training and generation pipeline for an autoregressive language model.
