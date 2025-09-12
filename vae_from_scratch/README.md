# Variational Autoencoder (VAE) From Scratch

This project implements a Variational Autoencoder (VAE) from scratch using PyTorch. The model is designed for educational purposes to demonstrate how a VAE can learn a compressed latent representation of image data and then use that representation to reconstruct the original images.

The VAE is trained on a small dataset of Pokémon images to learn a meaningful latent space for these images.

## Architecture

The VAE consists of two main components: an encoder and a decoder.

-   **Encoder**: A convolutional neural network that takes an input image (e.g., 3x512x512) and compresses it down to a lower-dimensional latent space. Instead of outputting a single vector, the encoder outputs the parameters (mean `mu` and log-variance `log_var`) of a probability distribution in the latent space.
-   **Reparameterization Trick**: A random sample `z` is drawn from the learned distribution using the `mu` and `log_var` from the encoder. This trick allows gradients to flow through the sampling process, making the model trainable.
-   **Decoder**: A convolutional transpose neural network that takes the latent vector `z` and upsamples it to reconstruct the original image.

### The VAE Loss Function
The training process is guided by a custom loss function composed of two parts:
1.  **Reconstruction Loss**: This is typically the Mean Squared Error (MSE) between the reconstructed image and the original input image. It pushes the model to learn to reconstruct the data accurately.
2.  **Kullback-Leibler (KL) Divergence**: This term acts as a regularizer. It measures how much the learned latent distribution (defined by `mu` and `log_var`) diverges from a standard normal distribution (mean=0, variance=1). This encourages the latent space to be smooth and well-structured, which is useful for generation.

## File Structure

-   `vae_model.py`: Contains the complete implementation of the VAE architecture, including the encoder, decoder, and reparameterization trick.
-   `train_vae.py`: The main training script. It handles data loading, data augmentation, the training loop, and saving the trained model.
-   `sample_vae.py`: A script for loading a trained VAE model to encode a sample image into its latent representation and then decode it back into an image.
-   `vae_results/`: A directory where reconstructed image samples are saved during training.
-   `README.md`: This documentation file.

## Usage

### 1. Prerequisites
Install the required Python libraries:
```bash
pip install torch torchvision pillow numpy matplotlib datasets
```

### 2. Training the VAE
To train the model, run the training script:
```bash
python train_vae.py
```
This script will:
-   Automatically download the Pokémon dataset from the Hugging Face Hub.
-   Apply data augmentations.
-   Train the VAE for 200 epochs.
-   Periodically save a comparison of original and reconstructed images to the `vae_results/` directory.
-   Save the final trained model weights to `vae_model.pth`.

### 3. Sampling and Reconstructing an Image
After training, you can see the VAE in action by running the sampling script:
```bash
python sample_vae.py
```
This will:
-   Load the trained `vae_model.pth`.
-   Load a sample image (`pokemon_sample_test.png`).
-   Encode the image into the latent space and then decode it back.
-   Display the original and reconstructed images side-by-side using Matplotlib.

## Key Features

-   **Pure PyTorch Implementation**: The model is built from scratch using fundamental PyTorch modules.
-   **Convolutional Architecture**: Uses a fully convolutional design for both the encoder and decoder, suitable for image data.
-   **Reparameterization Trick**: A clear implementation of this essential VAE technique.
-   **Data Augmentation**: The training script includes standard data augmentations like random flips and rotations to improve model robustness.
-   **Clear Separation of Concerns**: The code is well-organized into separate files for the model, training, and sampling.

## Learning Objectives

-   Understand the architecture of a Variational Autoencoder.
-   Learn the roles of the encoder, decoder, and the reparameterization trick.
-   Understand the VAE loss function and the balance between reconstruction and regularization (KL divergence).
-   See how a generative model can learn a compressed, meaningful representation of complex data like images.