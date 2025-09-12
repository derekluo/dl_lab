# Stable Diffusion From Scratch

This project implements a simplified, text-conditioned Latent Diffusion Model, similar in principle to Stable Diffusion, from scratch using PyTorch. The model is trained on a small dataset of Pokémon images with corresponding text captions to generate new, high-resolution (512x512) images from textual prompts.

This implementation is for educational purposes, demonstrating how to combine a Variational Autoencoder (VAE) and a text-conditioned diffusion model to create an efficient and powerful generative model.

## Architecture

The model consists of two main, pre-trained components: a VAE and a text-conditioned U-Net. The training process is focused on the U-Net, which operates in the compressed latent space created by the VAE.

1.  **Variational Autoencoder (VAE)**:
    -   The VAE is responsible for compressing high-resolution images (e.g., 3x512x512) into a much smaller latent representation (e.g., 4x64x64).
    -   Its decoder is used to transform the generated latent vectors back into high-resolution images.
    -   For this project, the VAE is pre-trained, and its weights are frozen during the diffusion model training.

2.  **Text-Conditioned U-Net (Diffusion Model)**:
    -   This is the core of the generative process. It operates entirely in the latent space.
    -   It is a U-Net architecture augmented with `SpatialTransformer` blocks that use cross-attention to inject text conditioning from a pre-trained CLIP text encoder.
    -   The U-Net is trained to predict the noise added to a latent representation at a given timestep.

### The Generative Process (Inference)
1.  A text prompt is converted into an embedding using a CLIP text encoder.
2.  A random tensor (noise) is created in the latent space.
3.  The U-Net iteratively denoises this latent tensor over a series of timesteps, guided by the text embedding.
4.  The final denoised latent tensor is passed through the VAE's decoder to produce the final, high-resolution image.

## File Structure

-   `stable_diffusion_model.py`: Defines the main `StableDiffusion` class that integrates the VAE and the U-Net, and handles the sampling logic.
-   `diffusion_model.py`: Contains the implementation of the text-conditioned U-Net with `ResnetBlock` and `SpatialTransformer` components.
-   `vae_model.py`: Contains the implementation of the VAE used for encoding and decoding images.
-   `train_stable_diffusion.py`: The main training script for the U-Net. It loads the pre-trained VAE, prepares the dataset by encoding all images into latents, and then runs the diffusion training loop.
-   `sample_stable_diffusion.ipynb`: A Jupyter notebook for generating images from a trained model using text prompts.
-   `stable_diffusion_results/`: A directory where model checkpoints and generated sample images are saved.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch torchvision pillow numpy tqdm datasets transformers wandb jupyter
```
You will also need to log in to Weights & Biases to track the training progress:
```bash
wandb login
```

### 2. Training the Model
The training process focuses on training the U-Net diffusion model. It assumes you have a pre-trained VAE model saved as `vae_model.pth`.

To start training:
```bash
python train_stable_diffusion.py
```
The script will:
-   Load the pre-trained VAE and freeze its weights.
-   Load the Pokémon dataset and encode all images into a latent dataset.
-   Train the U-Net to denoise these latents based on text captions.
-   Log progress to Weights & Biases and save checkpoints to the `stable_diffusion_results/` directory.

### 3. Generating Images
After training, use the `sample_stable_diffusion.ipynb` notebook to generate images.

1.  **Start Jupyter**:
    ```bash
    jupyter notebook
    ```
2.  **Open and Run the Notebook**: Open `sample_stable_diffusion.ipynb`.
    -   Update the path to your trained Stable Diffusion model checkpoint.
    -   Modify the text prompts in the cells.
    -   Run the cells to generate and display the images.

## Key Features

-   **Latent Diffusion**: Performs the computationally expensive diffusion process in the smaller latent space, making it much more efficient than pixel-space diffusion.
-   **Text-Conditioned Generation**: Uses embeddings from a CLIP model to guide the image generation process, allowing for creative control via text prompts.
-   **High-Resolution Output**: By using a VAE decoder, the model can generate high-resolution (512x512) images from low-resolution latent representations.
-   **Classifier-Free Guidance (CFG)**: Implements CFG during sampling to improve the quality and adherence of the generated images to the text prompt.

## Learning Objectives

-   Understand the two-stage architecture of a Latent Diffusion Model like Stable Diffusion.
-   Learn how a VAE and a diffusion model can be combined for efficient, high-resolution image generation.
-   See a practical application of cross-attention for conditioning a generative model on text embeddings.
-   Gain insight into the complete workflow of training and sampling from a state-of-the-art generative model.