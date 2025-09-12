# Transformer From Scratch

This project provides a from-scratch implementation of a Transformer-based text classifier using PyTorch. The model is designed for educational purposes to demonstrate the core components of the Transformer architecture as described in the paper "Attention Is All You Need."

The project includes a complete workflow: generating synthetic training data, building a vocabulary, training the model, and running inference on new text.

## Architecture

The model is based on the encoder part of the original Transformer architecture.

-   **Embedding and Positional Encoding**: Input tokens are converted into embeddings, which are then combined with sinusoidal positional encodings to provide the model with information about the sequence order.
-   **Transformer Blocks**: The core of the model is a stack of `TransformerBlock` modules. Each block consists of:
    -   **Multi-Head Self-Attention**: Allows the model to weigh the importance of different words in the input text when encoding a specific word.
    -   **Feed-Forward Network**: A two-layer feed-forward network applied after the attention mechanism.
    -   **Residual Connections and Layer Normalization**: Both sub-layers (attention and feed-forward) include residual connections and are followed by layer normalization for stable and effective training.
-   **Classification Head**: The output from the transformer blocks is averaged across the sequence length and then passed through a final linear layer to produce classification scores for the different text categories.

## File Structure

-   `model.py`: Contains the complete implementation of the Transformer architecture, including `MultiHeadAttention`, `PositionalEncoding`, and `TransformerBlock`.
-   `train.py`: The main training script. It handles the creation of synthetic data, vocabulary building, training, evaluation, and plotting of results.
-   `inference.py`: A script for loading a trained model and running inference on new text, with an interactive mode.
-   `generate_test_texts.py`: A utility to generate a more diverse set of test sentences for each category, which can be used for more robust evaluation.
-   `test_texts/`: Directory where the generated test texts are saved.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch scikit-learn matplotlib numpy
```

### 2. Training the Model
To train the model, run the training script:
```bash
python train.py
```
This script will:
-   Generate a synthetic dataset for 5 text categories (Technology, Sports, Science, Music, Food).
-   Build a vocabulary from the training data.
-   Train the Transformer model for 15 epochs.
-   Print the loss and accuracy for each epoch.
-   Save the trained model, vocabulary, and configuration to `transformer_model.pth`.
-   Save a plot of the training curves to `training_results.png`.

### 3. Running Inference
After training, you can classify new text using the `inference.py` script:
```bash
python inference.py
```
The script will first run on a set of predefined sample sentences and then enter an interactive mode where you can input your own text to be classified.

### 4. Generating a Test Set (Optional)
For more thorough testing, you can generate a richer set of test sentences:
```bash
python generate_test_texts.py
```
This will create several files in the `test_texts/` directory, including a `mixed_test_set.json` file that can be used for a more formal evaluation.

## Key Features

-   **Pure PyTorch Implementation**: The model is built from scratch using fundamental PyTorch modules, providing a clear view of the inner workings of a Transformer.
-   **Self-Contained Project**: Includes scripts for data generation, training, and inference, making it a complete, runnable example.
-   **Clear Component-Based Architecture**: The code is organized into logical components (`MultiHeadAttention`, `TransformerBlock`, etc.), making it easy to understand the different parts of the model.
-   **Training Visualization**: Automatically generates plots for training loss and test accuracy to help monitor the training process.

## Learning Objectives

-   Understand the architecture of a Transformer encoder.
-   Learn how to implement multi-head self-attention and positional encoding from scratch.
-   See how to build a complete text classification pipeline using a Transformer model.
-   Gain insight into the process of creating a vocabulary, tokenizing text, and preparing data for an NLP model.
