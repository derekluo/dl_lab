# Diffusion Model From Scratch

This project implements a text-conditioned Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch. The model is trained on a small dataset of Pokémon images with corresponding text captions to generate new images based on textual prompts.

This implementation is designed for educational purposes to demonstrate the core components of a modern conditional diffusion model, including a U-Net with Spatial Transformer attention blocks.

## Architecture

The model is based on a U-Net architecture enhanced with attention mechanisms to incorporate text conditioning.

-   **U-Net Backbone**: The core of the model is a U-Net that predicts the noise added to an image at a given timestep. It consists of down-sampling blocks, a middle block, and up-sampling blocks with skip connections.
-   **Text Conditioning**: The model is conditioned on text prompts, which are encoded using a pre-trained CLIP text encoder.
-   **Spatial Transformer**: The U-Net is augmented with `SpatialTransformer` blocks, which use cross-attention to integrate the text embeddings into the image generation process. This allows the model to learn relationships between the text and image features.
-   **Noise Scheduler**: A cosine noise scheduler is used to manage the forward (noising) and reverse (denoising) processes.

## File Structure

-   `diffusion_model.py`: Contains the complete implementation of the U-Net architecture with `ResnetBlock` and `SpatialTransformer` components, the `NoiseScheduler`, and the sampling functions.
-   `train_diffusion.py`: The main training script. It handles data loading, text encoding, the training loop, model checkpointing, and logging to Weights & Biases.
-   `sample_diffusion.py`: A script for generating images from a trained model using a text prompt. It supports both standard and Classifier-Free Guidance (CFG) sampling.
-   `diffusion_results/`: A directory where model checkpoints and generated sample images are saved.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch torchvision pillow numpy tqdm datasets transformers wandb
```
You will also need to log in to Weights & Biases to track the training progress:
```bash
wandb login
```

### 2. Training the Model
To start training the diffusion model, run the training script:
```bash
python train_diffusion.py
```
The script will automatically download the Pokémon dataset from the Hugging Face Hub. During training, it will:
-   Log the training and validation loss to Weights & Biases.
-   Save model checkpoints periodically to the `diffusion_results/` directory.
-   Generate and save sample images at regular intervals to visualize the model's progress.

### 3. Generating Images
After training, you can generate new images using the `sample_diffusion.py` script.

1.  **Update the script**: Open `sample_diffusion.py` and set the path to your trained model checkpoint.
2.  **Set the text prompt**: Modify the `condition` variable to your desired text prompt.
3.  **Run the script**:
    ```bash
    python sample_diffusion.py
    ```
This will generate and save two images: one using standard DDPM sampling (`generated_image_pokemon.png`) and another using Classifier-Free Guidance (`generated_image_pokemon_cfg.png`).

## Key Features

-   **Text-Conditioned Generation**: The model can generate images based on descriptive text prompts.
-   **Modern U-Net Architecture**: Implements a U-Net with ResNet blocks and self-attention, similar to those used in state-of-the-art models.
-   **Cross-Attention Mechanism**: Utilizes a `SpatialTransformer` to effectively inject the text conditioning into the U-Net, enabling fine-grained control over the generated image.
-   **Classifier-Free Guidance (CFG)**: The sampling script includes an implementation of CFG, a technique that improves the quality and relevance of conditional image generation.
-   **Training Monitoring**: Integrated with Weights & Biases for real-time tracking of losses, learning rates, and generated image samples.

## Learning Objectives

-   Understand the architecture of a text-conditioned diffusion model.
-   Learn how a U-Net can be modified with attention mechanisms for conditional generation.
-   See how text embeddings from models like CLIP are used to guide the image synthesis process.
-   Gain insight into the training and sampling loops of a diffusion model, including the role of the noise scheduler and Classifier-Free Guidance.