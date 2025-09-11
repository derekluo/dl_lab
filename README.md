# Deep Learning From Scratch

This repository contains implementations of various deep learning models and neural network architectures built from scratch using PyTorch and NumPy. Each project is designed to provide a clear understanding of the underlying architectures, algorithms, and principles.

## Projects Overview

### 1. LeNet
Implementation of the classic LeNet-5 convolutional neural network architecture, designed by Yann LeCun in 1998. This pioneering CNN architecture demonstrated remarkable effectiveness for handwritten digit recognition on the MNIST dataset.

**Files:**
- `lenet_from_scratch/lenet.py` - Model architecture implementing convolutional layers, max pooling, and fully connected layers
- `lenet_from_scratch/train.py` - Training script with SGD optimizer, CrossEntropy loss, and MNIST dataset loader

### 2. LeNet5 (Extended Version)
An extended implementation of LeNet-5 with modern improvements and additional utilities:

**Files:**
- `lenet5_from_scratch/model.py` - Enhanced model architecture with modern improvements
- `lenet5_from_scratch/train.py` - Advanced training script with hyperparameter tuning and visualization
- `lenet5_from_scratch/inference.py` - Dedicated inference pipeline for testing
- `lenet5_from_scratch/generate_test_digits.py` - Custom digit generation utilities

**Features:**
- Modern improvements over base LeNet
- Training visualization and results plotting
- Model checkpointing
- Test digit generation and evaluation

### 3. AlexNet
Implementation of AlexNet, the revolutionary CNN architecture that won the 2012 ImageNet competition and sparked the deep learning revolution in computer vision.

**Files:**
- `alexnet_from_scratch/alexnet.py` - Model architecture with 5 convolutional layers and 3 fully connected layers
- `alexnet_from_scratch/train.py` - Training pipeline with ImageNet preprocessing and SGD optimizer
- `alexnet_from_scratch/inference.py` - Inference utilities with support for top-k predictions

**Features:**
- Original AlexNet architecture
- ImageNet-style preprocessing
- GPU support and configurable number of output classes
- Top-k prediction capabilities

### 4. Transformer From Scratch
A complete implementation of the Transformer architecture for text classification, built from fundamental components without using pre-built transformer layers.

**Files:**
- `transformer_from_scratch/model.py` - Complete transformer implementation with multi-head attention, positional encoding, and transformer blocks
- `transformer_from_scratch/train.py` - Training script with synthetic text data and visualization
- `transformer_from_scratch/inference.py` - Text classification inference with interactive mode
- `transformer_from_scratch/generate_test_texts.py` - Test data generation for multiple categories

**Features:**
- Multi-head self-attention mechanism
- Positional encoding with sine/cosine functions
- Layer normalization and residual connections
- Text classification for 5 categories (Technology, Sports, Science, Music, Food)
- Training visualization and performance monitoring

### 5. Backpropagation From Scratch
Educational implementation of neural networks with manual backpropagation, demonstrating every step of gradient computation and parameter updates.

**Files:**
- `backprop_from_scratch/model.py` - Neural network components with manual gradient computation
- `backprop_from_scratch/train.py` - Training demonstrations for classification and regression
- `backprop_from_scratch/demo.py` - Step-by-step backpropagation analysis including XOR problem
- `backprop_from_scratch/visualize.py` - Comprehensive visualizations of networks, gradients, and training dynamics

**Features:**
- Manual implementation of forward and backward propagation
- Multiple activation functions (ReLU, Sigmoid, Tanh)
- Detailed gradient computation with step-by-step logging
- XOR problem demonstration with detailed analysis
- Extensive visualizations: network architecture, activation functions, loss landscapes, gradient flow

### 6. GPT-2
A from-scratch implementation of the GPT-2 architecture, focusing on the core transformer-based language model components.

**Files:**
- `gpt2_from_scratch/model.py` - GPT-2 model architecture
- `gpt2_from_scratch/train.py` - Training pipeline
- `gpt2_from_scratch/inference.py` - Text generation utilities
- `gpt2_from_scratch/show_params.py` - Model parameter analysis

### 7. VAE (Variational Autoencoder)
Implementation of a Variational Autoencoder for learning latent representations and generating new data samples.

**Files:**
- `vae_from_scratch/vae_model.py` - VAE architecture with encoder and decoder
- `vae_from_scratch/train_vae.py` - Training script with reconstruction and KL divergence losses
- `vae_from_scratch/sample_vae.py` - Sampling and visualization utilities

**Features:**
- Complete VAE implementation with reparameterization trick
- Training with Pokemon dataset
- Latent space visualization and analysis
- Sample generation from learned distributions

### 8. Diffusion Model
Basic implementation of a diffusion-based generative model, demonstrating the core concepts behind modern image generation models.

**Files:**
- `diffusion_from_scratch/diffusion_model.py` - U-Net architecture for diffusion
- `diffusion_from_scratch/train_diffusion.py` - Training pipeline
- `diffusion_from_scratch/sample_diffusion.py` - Image generation utilities

**Features:**
- Forward and reverse diffusion processes
- U-Net architecture for noise prediction
- MNIST dataset training and generation

### 9. Stable Diffusion
A simplified implementation of the Stable Diffusion architecture, focused on educational understanding.

**Files:**
- `stable_diffusion_from_scratch/stable_diffusion_model.py` - Core architecture including UNet with attention
- `stable_diffusion_from_scratch/train_stable_diffusion.py` - Training pipeline
- `stable_diffusion_from_scratch/sample_stable_diffusion.ipynb` - Interactive generation notebook
- `stable_diffusion_from_scratch/vae_model.py` - VAE component for latent space operations

## Learning Path Recommendations

### For Beginners:
1. **Backpropagation From Scratch** - Understand fundamental neural network principles
2. **LeNet/LeNet5** - Learn basic CNN architectures
3. **Transformer From Scratch** - Understand attention mechanisms

### For Intermediate:
1. **VAE** - Learn generative modeling and latent representations
2. **GPT-2** - Advanced transformer architectures for language modeling
3. **AlexNet** - Modern CNN techniques and computer vision

### For Advanced:
1. **Diffusion Models** - State-of-the-art generative modeling
2. **Stable Diffusion** - Complex multi-component architectures

## Key Learning Concepts

### Neural Network Fundamentals
- Forward and backward propagation
- Gradient computation and parameter updates
- Loss functions and optimization
- Activation functions and their properties

### Computer Vision
- Convolutional layers and feature extraction
- Pooling operations and spatial hierarchies
- Image preprocessing and data augmentation
- Classification and object recognition

### Natural Language Processing
- Attention mechanisms and self-attention
- Positional encoding and sequence modeling
- Text classification and language generation
- Tokenization and embedding strategies

### Generative Models
- Variational inference and latent spaces
- Diffusion processes and noise scheduling
- Autoregressive generation
- Latent diffusion and multi-modal generation

## Installation and Usage

### Prerequisites
```bash
pip install torch torchvision numpy matplotlib scikit-learn
```

### Running Projects
Each project can be run independently:
```bash
cd <project_directory>
python train.py        # Train the model
python inference.py    # Run inference (where available)
python demo.py         # Run demonstrations (where available)
python visualize.py    # Generate visualizations (where available)
```

## Project Structure
Each project follows a consistent structure:
- `model.py` - Model architecture and components
- `train.py` - Training script with evaluation
- `inference.py` - Inference and testing utilities (where applicable)
- `README.md` - Detailed project documentation
- Additional utilities and visualization scripts as needed

## Educational Value
This repository is designed for:
- **Students** learning deep learning fundamentals
- **Researchers** understanding architectural details
- **Engineers** implementing models from scratch
- **Anyone** interested in the mathematical foundations of deep learning

Each implementation prioritizes **clarity and educational value** over performance optimization, making the code easy to understand and modify for learning purposes.

## References and Further Reading
- **Deep Learning (Ian Goodfellow, Yoshua Bengio, Aaron Courville)**
- **Neural Networks and Deep Learning (Michael Nielsen)**
- **CS231n: Convolutional Neural Networks for Visual Recognition (Stanford)**
- **The Annotated Transformer (Harvard NLP)**
- Original papers for each architecture (linked in individual README files)

## Contributing
Contributions are welcome! Please focus on:
- **Educational clarity** over performance
- **Detailed documentation** and comments
- **Consistent code structure** across projects
- **Mathematical accuracy** in implementations

## License
This project is open source and available under the MIT License.