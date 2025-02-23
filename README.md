# Deep Learning From Scratch

This repository contains implementations of various deep learning models built from scratch using PyTorch. Each project is designed to provide a clear understanding of the underlying architectures and principles.

## Projects

### 1. LeNet
Implementation of the classic LeNet-5 convolutional neural network architecture, designed by Yann LeCun in 1998. This pioneering CNN architecture demonstrated remarkable effectiveness for handwritten digit recognition on the MNIST dataset.

Files:
- `lenet_from_scratch/lenet.py` - Model architecture implementing convolutional layers, max pooling, and fully connected layers
- `lenet_from_scratch/train.py` - Training script with SGD optimizer, CrossEntropy loss, and MNIST dataset loader

### 2. LeNet5 (Extended Version)
An extended implementation of LeNet-5 with modern improvements and additional utilities:
- `lenet5_from_scratch/model.py` - Enhanced model architecture with modern improvements
- `lenet5_from_scratch/train.py` - Advanced training script with hyperparameter tuning
- `lenet5_from_scratch/inference.py` - Dedicated inference pipeline
- `lenet5_from_scratch/generate_test_digits.py` - Custom digit generation utilities

Planned improvements over base LeNet:
- Batch normalization layers
- Dropout for regularization
- Learning rate scheduling
- Model checkpointing
- TensorBoard integration

### 3. GPT-2
A from-scratch implementation of the GPT-2 architecture, focusing on the core transformer-based language model components.
- `gpt2_from_scratch/model.py` - Model architecture
- `gpt2_from_scratch/train.py` - Training pipeline
- `gpt2_from_scratch/tokenizer.py` - Custom tokenizer implementation

### 4. VAE
Implementation of a Variational Autoencoder for learning latent representations and generating new data samples.
- `vae_from_scratch/model.py` - VAE architecture
- `vae_from_scratch/train.py` - Training script with reconstruction and KL divergence losses
- `vae_from_scratch/visualize.py` - Latent space visualization tools

### 5. Diffusion Model
Basic implementation of a diffusion-based generative model, demonstrating the core concepts behind DALL-E and Stable Diffusion.
- `diffusion_from_scratch/model.py` - U-Net architecture
- `diffusion_from_scratch/train.py` - Training pipeline
- `diffusion_from_scratch/scheduler.py` - Noise scheduling implementation

### 6. Stable Diffusion
A simplified implementation of the Stable Diffusion architecture, focused on educational understanding.
- `stable_diffusion_from_scratch/model.py` - Core architecture including UNet with attention
- `stable_diffusion_from_scratch/train.py` - Training pipeline
- `stable_diffusion_from_scratch/inference.py` - Image generation utilities

### 7. AlexNet
Implementation of AlexNet, the revolutionary CNN architecture that won the 2012 ImageNet competition and sparked the deep learning revolution in computer vision.

Files:
- `alexnet_from_scratch/alexnet.py` - Model architecture with 5 convolutional layers and 3 fully connected layers
- `alexnet_from_scratch/train.py` - Training pipeline with ImageNet preprocessing and SGD optimizer
- `alexnet_from_scratch/inference.py` - Inference utilities with support for top-k predictions

Features:
- Original AlexNet architecture
- ImageNet-style preprocessing
- GPU support
- Configurable number of output classes
- Top-k prediction capabilities
- Support for custom class labels
