---
noteId: "a98f58308ef911f0a16d2d94aac5ab6c"
tags: []

---

# AlexNet From Scratch

This project implements the revolutionary AlexNet convolutional neural network architecture from scratch using PyTorch. AlexNet won the 2012 ImageNet Large Scale Visual Recognition Challenge (ILSVRC) and sparked the deep learning revolution in computer vision.

## Architecture

AlexNet consists of:
- **Input Layer**: 224×224×3 RGB images
- **5 Convolutional Layers** with ReLU activations and max pooling
- **3 Fully Connected Layers** with dropout for regularization
- **Output Layer**: 1000 classes (ImageNet) or configurable for other datasets

### Layer Details:
1. **Conv1**: 96 kernels (11×11×3), stride 4, followed by ReLU and max pooling
2. **Conv2**: 256 kernels (5×5×96), stride 1, followed by ReLU and max pooling
3. **Conv3**: 384 kernels (3×3×256), stride 1, followed by ReLU
4. **Conv4**: 384 kernels (3×3×384), stride 1, followed by ReLU
5. **Conv5**: 256 kernels (3×3×384), stride 1, followed by ReLU and max pooling
6. **FC1**: 4096 neurons with ReLU and dropout
7. **FC2**: 4096 neurons with ReLU and dropout
8. **FC3**: Output layer (num_classes neurons)

## Files

- `alexnet.py` - Complete AlexNet model implementation
- `train.py` - Training script with ImageNet-style preprocessing
- `inference.py` - Inference utilities with top-k predictions

## Usage

### Training the Model
```bash
python train.py
```

Features:
- ImageNet-style data preprocessing and augmentation
- SGD optimizer with momentum
- Learning rate scheduling
- GPU acceleration support
- Training progress monitoring

### Running Inference
```bash
python inference.py
```

Features:
- Single image classification
- Top-k predictions with confidence scores
- Support for custom class labels
- Batch inference capabilities

## Key Innovations of AlexNet

1. **ReLU Activation**: First major CNN to use ReLU instead of tanh/sigmoid
2. **GPU Implementation**: Pioneered GPU training for deep networks
3. **Dropout**: Used dropout in fully connected layers to prevent overfitting
4. **Data Augmentation**: Extensive use of image transformations
5. **Local Response Normalization**: Early form of normalization (now replaced by batch norm)

## Implementation Features

- **Pure PyTorch**: No pre-built models, implemented from scratch
- **Flexible Architecture**: Configurable number of output classes
- **GPU Support**: CUDA and MPS (Apple Silicon) acceleration
- **Modern Training**: Includes techniques like learning rate scheduling
- **Inference Tools**: Top-k prediction and confidence scoring

## Expected Performance

On ImageNet (original dataset):
- **Top-1 Accuracy**: ~57.1%
- **Top-5 Accuracy**: ~80.2%

Note: Performance may vary based on training setup and hyperparameters.

## Learning Objectives

This implementation demonstrates:
- Deep convolutional neural network architecture design
- The importance of ReLU activations in deep networks
- Dropout regularization techniques
- GPU-accelerated training workflows
- Image classification pipeline from preprocessing to inference

## Requirements

```bash
pip install torch torchvision pillow numpy matplotlib
```

## Historical Impact

AlexNet's victory in ImageNet 2012 was significant because:
- It demonstrated the power of deep learning for computer vision
- It showed that CNNs could outperform traditional computer vision methods
- It sparked the "deep learning revolution" in AI research
- It established many practices still used in modern deep learning

## Architecture Comparison

| Aspect | LeNet-5 (1998) | AlexNet (2012) |
|--------|----------------|----------------|
| Layers | 7 | 8 |
| Parameters | ~60K | ~60M |
| Input Size | 32×32 | 224×224 |
| Activation | tanh | ReLU |
| Dataset | MNIST | ImageNet |
| GPU Training | No | Yes |

## Modern Improvements

While AlexNet was groundbreaking, modern CNNs include:
- **Batch Normalization**: Replaces Local Response Normalization
- **Skip Connections**: As in ResNet architectures
- **Attention Mechanisms**: For better feature selection
- **Efficient Architectures**: Like MobileNet and EfficientNet
- **Transfer Learning**: Pre-trained models for quick adaptation

This implementation serves as an excellent bridge between simple CNNs (LeNet) and modern architectures (ResNet, Vision Transformers).
