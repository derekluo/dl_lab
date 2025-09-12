# CLIP from Scratch

A complete implementation of CLIP (Contrastive Language-Image Pre-training) from scratch using PyTorch. This implementation includes the full training pipeline, inference capabilities, and comprehensive examples.

## Overview

CLIP is a neural network that efficiently learns visual concepts from natural language supervision. It can be applied to any visual classification benchmark by simply providing the names of the visual categories to be recognized, similar to the "zero-shot" capabilities of GPT-2 and GPT-3.

### Key Features

- **Complete Implementation**: Full CLIP architecture with Vision Transformer (ViT) image encoder and Transformer text encoder
- **Flexible Model Sizes**: Support for tiny, small, base, and large model configurations
- **Training Pipeline**: End-to-end training with contrastive loss
- **Zero-Shot Classification**: Classify images without training on specific categories
- **Image-Text Retrieval**: Find matching images for text queries and vice versa
- **Synthetic Dataset Generation**: Create demo datasets for testing
- **Comprehensive Inference Tools**: Multiple inference modes with visualization

## Architecture

### Image Encoder (Vision Transformer)
- Patch embedding with configurable patch size
- Multi-head self-attention layers
- Layer normalization and residual connections
- Global average pooling and projection head
- L2 normalization of output features

### Text Encoder (Transformer)
- Character-level tokenization (simple implementation)
- Positional embeddings
- Multi-head self-attention layers
- Layer normalization and residual connections
- Final token representation and projection head
- L2 normalization of output features

### Contrastive Learning
- Symmetric cross-entropy loss
- Learnable temperature parameter
- Batch-wise contrastive learning

## Installation

```bash
# Clone the repository
cd clip_from_scratch

# Install dependencies
pip install torch torchvision transformers pillow matplotlib tqdm numpy
```

## Quick Start

### 1. Train a Model

```bash
# Train with synthetic dataset (automatic generation)
python train.py --create_synthetic --epochs 100 --batch_size 32 --model_size base

# Train with custom dataset
python train.py --data_file your_data.json --image_root your_images/ --epochs 100
```

### 2. Run Inference

```bash
# Demo mode (uses trained model with sample data)
python inference.py

# Image-text similarity
python inference.py --model_path clip_model.pth --mode similarity \
    --image test_image.jpg --texts "a red circle" "a blue square" "a green triangle"

# Zero-shot classification
python inference.py --model_path clip_model.pth --mode classification \
    --image test_image.jpg --classes "red circle" "blue square" "green triangle"

# Image retrieval
python inference.py --model_path clip_model.pth --mode retrieval \
    --text "a red shape" --image_dir test_images/
```

## Dataset Format

The training script expects a JSON file with image-caption pairs:

```json
[
  {
    "image": "image_001.jpg",
    "caption": "a red circle on white background"
  },
  {
    "image": "image_002.jpg", 
    "caption": "a blue square shape"
  }
]
```

Directory structure:
```
data/
├── dataset.json
└── images/
    ├── image_001.jpg
    ├── image_002.jpg
    └── ...
```

## Model Configurations

| Model | Embed Dim | Heads | Layers | Parameters |
|-------|-----------|-------|--------|------------|
| Tiny  | 256       | 4     | 6      | ~5M        |
| Small | 384       | 6     | 8      | ~15M       |
| Base  | 512       | 8     | 12     | ~35M       |
| Large | 768       | 12    | 24     | ~125M      |

## Training Options

```bash
python train.py --help
```

Key parameters:
- `--model_size`: Model configuration (tiny/small/base/large)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate (default: 1e-4)
- `--create_synthetic`: Generate synthetic dataset if data file not found

## Inference Modes

### 1. Image-Text Similarity
Compare an image with multiple text descriptions:

```python
from inference import CLIPInference

clip_model = CLIPInference('clip_model.pth')
similarities = clip_model.image_text_similarity(
    'test_image.jpg', 
    ['a red circle', 'a blue square', 'a green triangle']
)
```

### 2. Zero-Shot Classification
Classify images into predefined categories:

```python
results = clip_model.zero_shot_classification(
    'test_image.jpg',
    ['red circle', 'blue square', 'green triangle'],
    template="a photo of a {}"
)
```

### 3. Image Retrieval
Find images matching a text query:

```python
ranked_images, similarities = clip_model.text_image_retrieval(
    'a red shape',
    ['img1.jpg', 'img2.jpg', 'img3.jpg']
)
```

### 4. Text Retrieval
Find text descriptions matching an image:

```python
ranked_texts, similarities = clip_model.image_text_retrieval(
    'test_image.jpg',
    ['description 1', 'description 2', 'description 3']
)
```

## Code Structure

```
clip_from_scratch/
├── model.py           # CLIP architecture and components
├── train.py          # Training script with data loading
├── inference.py      # Inference tools and demo
└── README.md        # This file
```

### Key Classes

- **`CLIP`**: Main model combining image and text encoders
- **`ImageEncoder`**: Vision Transformer for image processing
- **`TextEncoder`**: Transformer for text processing
- **`CLIPLoss`**: Contrastive loss implementation
- **`CLIPInference`**: Inference wrapper with utilities
- **`SimpleTokenizer`**: Character-level text tokenizer

## Training Process

1. **Data Loading**: Load image-text pairs and apply transforms
2. **Forward Pass**: Encode images and texts separately
3. **Similarity Computation**: Calculate cosine similarity matrix
4. **Contrastive Loss**: Maximize similarity for correct pairs
5. **Backpropagation**: Update model parameters
6. **Validation**: Evaluate on held-out data

## Results and Metrics

The training script tracks:
- **Training Loss**: Contrastive loss during training
- **Validation Loss**: Loss on validation set
- **Image-to-Text Accuracy**: Top-1 retrieval accuracy
- **Text-to-Image Accuracy**: Top-1 retrieval accuracy

Training curves are automatically saved as `training_results.png`.

## Example Usage

### Train on Synthetic Data
```bash
# Generate and train on 1000 synthetic colored shapes
python train.py --create_synthetic --epochs 50 --batch_size 16
```

### Test Zero-Shot Classification
```bash
# Create test image
python -c "
from PIL import Image, ImageDraw
import numpy as np
img = Image.new('RGB', (224, 224), 'white')
draw = ImageDraw.Draw(img)
draw.ellipse([50, 50, 174, 174], fill='red')
img.save('test_red_circle.jpg')
"

# Classify the image
python inference.py --model_path clip_model.pth --mode classification \
    --image test_red_circle.jpg \
    --classes "red circle" "blue circle" "red square" "green triangle"
```

## Synthetic Dataset

The training script can automatically generate a synthetic dataset of colored shapes:
- **Colors**: red, blue, green, yellow, purple, orange
- **Shapes**: circle, square, triangle
- **Captions**: "a {color} {shape}"

This is useful for:
- Testing the implementation
- Understanding CLIP's learning process
- Debugging training issues

## Performance Tips

1. **Model Size**: Start with 'small' or 'base' for faster training
2. **Batch Size**: Use largest batch size that fits in memory
3. **Learning Rate**: 1e-4 works well, reduce if training unstable
4. **Epochs**: 50-100 epochs usually sufficient for synthetic data
5. **Data Quality**: Higher quality image-text pairs improve performance

## Limitations

1. **Simple Tokenizer**: Uses character-level tokenization instead of BPE
2. **Small Scale**: Designed for educational purposes, not production
3. **Limited Datasets**: Best suited for simple, controlled datasets
4. **Computational**: Requires GPU for reasonable training times

## Future Improvements

- [ ] BPE tokenizer integration
- [ ] Data augmentation strategies
- [ ] Multi-GPU training support
- [ ] Additional evaluation metrics
- [ ] Pre-trained model downloads
- [ ] Integration with popular datasets (COCO, Flickr30k)

## References

- [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

## License

This implementation is for educational purposes. Please refer to the original CLIP paper and OpenAI's terms for any commercial usage considerations.
