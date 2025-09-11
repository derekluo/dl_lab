---
noteId: "c1d1abf08ef911f0a16d2d94aac5ab6c"
tags: []

---

# LeNet5 From Scratch (Extended Version)

This is an enhanced implementation of LeNet-5 with modern improvements and comprehensive utilities. Unlike the basic LeNet implementation, this version includes advanced training features, visualization tools, and inference capabilities.

## Architecture

Enhanced LeNet-5 with modern improvements:
- **Input Layer**: 32×32 grayscale images (MNIST)
- **Conv Layer 1**: 6 feature maps, 5×5 kernel + ReLU + MaxPool
- **Conv Layer 2**: 16 feature maps, 5×5 kernel + ReLU + MaxPool  
- **Fully Connected**: 16×6×6 → 120 → 84 → 10 classes
- **Improvements**: Better weight initialization, modern activation functions

## Files

- `model.py` - Enhanced LeNet5 model with modern improvements
- `train.py` - Advanced training script with visualization and monitoring
- `inference.py` - Comprehensive inference pipeline with testing utilities
- `generate_test_digits.py` - Custom test digit generation for evaluation

## Usage

### 1. Training the Model
```bash
python train.py
```

**Features:**
- Automatic MNIST dataset download and preprocessing
- Training progress visualization with loss and accuracy plots
- Model checkpointing with best weights saving
- Support for MPS (Apple Silicon), CUDA, and CPU
- Real-time training monitoring

**Outputs:**
- `lenet5_model.pth` - Trained model weights
- `training_results.png` - Training curves visualization

### 2. Generating Test Data
```bash
python generate_test_digits.py
```

**Features:**
- Creates custom test digits (0-9) using PIL
- Generates images in `test_digits/` directory
- Configurable image size and styling
- Font-based digit rendering for testing

### 3. Running Inference
```bash
python inference.py
```

**Features:**
- Tests model on generated digits
- Batch evaluation of all test images
- Detailed prediction results with confidence
- Visual feedback with checkmarks/crosses

## Key Improvements Over Basic LeNet

### Training Enhancements:
- **Modern Optimizers**: Adam optimizer with better convergence
- **Improved Initialization**: Xavier/He initialization for faster training
- **Learning Rate Scheduling**: Adaptive learning rate adjustments
- **Progress Monitoring**: Real-time loss and accuracy tracking
- **Visualization**: Training curves and results plotting

### Inference Pipeline:
- **Custom Test Generation**: Create your own test cases
- **Batch Processing**: Evaluate multiple images efficiently  
- **Confidence Scoring**: Get prediction probabilities
- **Error Analysis**: Visual feedback on correct/incorrect predictions

### Code Quality:
- **Modular Design**: Separate files for different functionalities
- **Comprehensive Logging**: Detailed training information
- **Error Handling**: Robust file operations and error checking
- **Documentation**: Clear code comments and docstrings

## Expected Results

- **Training Accuracy**: >99%
- **Test Accuracy**: >98%
- **Training Time**: 1-2 minutes (GPU), 3-5 minutes (CPU)
- **Generated Test Accuracy**: ~90-95% (depends on font rendering)

## Generated Files

After running the complete pipeline:
```
lenet5_from_scratch/
├── model.py
├── train.py  
├── inference.py
├── generate_test_digits.py
├── lenet5_model.pth          # Trained weights
├── training_results.png      # Training visualization
└── test_digits/              # Generated test images
    ├── digit_0.png
    ├── digit_1.png
    └── ... (digit_9.png)
```

## Learning Objectives

This enhanced implementation teaches:

### Deep Learning Concepts:
- CNN architecture design and layer composition
- Training loop implementation with proper monitoring
- Model evaluation and validation techniques
- Hyperparameter tuning and optimization strategies

### Software Engineering:
- Project organization and modular code design
- File I/O operations and data management
- Visualization and result presentation
- Testing and validation pipelines

### Computer Vision:
- Image preprocessing and normalization
- Data augmentation techniques
- Custom dataset creation and evaluation
- Performance analysis and model interpretation

## Requirements

```bash
pip install torch torchvision matplotlib pillow numpy
```

## Advanced Features

### Custom Dataset Testing
The `generate_test_digits.py` creates synthetic digits using different fonts and styles, allowing you to:
- Test model robustness on different digit styles
- Evaluate generalization beyond MNIST training data
- Create custom evaluation scenarios
- Debug model predictions on edge cases

### Visualization and Monitoring
The training script provides:
- Real-time loss and accuracy curves
- Training progress with epoch-by-epoch results
- Saved plots for later analysis
- Performance comparison across training runs

### Production-Ready Inference
The inference pipeline includes:
- Batch processing capabilities
- Confidence scoring for predictions
- Error analysis and debugging tools
- Easy integration with other applications

## Next Steps

This implementation serves as a foundation for:
- **Experimenting with architectures**: Modify layers and observe effects
- **Advanced techniques**: Add batch normalization, dropout, data augmentation
- **Transfer learning**: Use pre-trained features for new tasks
- **Model optimization**: Quantization, pruning, and deployment techniques

## Comparison with Other Implementations

| Feature | Basic LeNet | LeNet5 Extended | Modern CNNs |
|---------|-------------|-----------------|-------------|
| Training Visualization | ❌ | ✅ | ✅ |
| Custom Test Generation | ❌ | ✅ | ✅ |
| Inference Pipeline | ❌ | ✅ | ✅ |
| Model Checkpointing | ❌ | ✅ | ✅ |
| Batch Processing | ❌ | ✅ | ✅ |
| Performance Monitoring | ❌ | ✅ | ✅ |

This extended version bridges the gap between educational implementations and production-ready deep learning systems.
