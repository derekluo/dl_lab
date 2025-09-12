---
noteId: "9cb990d08fa811f0a16d2d94aac5ab6c"
tags: []

---

# Linear Regression From Scratch (No PyTorch)

Pure NumPy implementation of linear regression (`y = Wx + B`) demonstrating:
- Manual forward pass computation
- Manual gradient calculation using calculus  
- Manual gradient descent optimization
- Training on synthetic data with noise
- Model evaluation with metrics and visualization

## Key Differences from PyTorch Version
- **No automatic differentiation**: Gradients computed manually using chain rule
- **No built-in optimizers**: SGD implemented from scratch
- **No tensor operations**: Pure NumPy arrays for all computations
- **Manual parameter updates**: Direct weight and bias updates
- **Custom loss functions**: MSE, MAE, R² implemented manually

## Architecture
- **Model**: `y_pred = W * X + B` (manual linear transformation)
- **Loss**: Mean Squared Error (manual computation)
- **Optimizer**: Stochastic Gradient Descent (manual implementation)
- **Gradients**: Computed analytically using calculus derivatives

## Mathematical Foundation

### Forward Pass
```
y_pred = W * X + B
```

### Loss Function (MSE)
```
L = (1/N) * Σ(y_pred - y_true)²
```

### Gradients (Chain Rule)
```
dL/dW = (2/N) * Σ((y_pred - y_true) * X)
dL/dB = (2/N) * Σ(y_pred - y_true)
```

### Parameter Updates (SGD)
```
W = W - learning_rate * dW
B = B - learning_rate * dB
```

## Files
- `model.py` - LinearRegressionModel class with manual operations
- `generate_data.py` - Create synthetic dataset using NumPy
- `train.py` - Train model with manual gradient descent
- `inference.py` - Evaluate trained model with custom metrics
- `data/` - Generated datasets and results

## Usage

### Quick Start
```bash
cd linear_regression_from_scratch_without_pytorch
python generate_data.py --num_samples 1000
python train.py --epochs 100
python inference.py
```

### Advanced Usage
```bash
# Generate data
python generate_data.py --num_samples 1000 --train_split 0.8

# Train with gradient checking
python train.py --learning_rate 0.01 --epochs 100 --grad_check

# Evaluate with function comparison
python inference.py --compare
```

### Parameters
- **generate_data.py**: `--num_samples 1000 --train_split 0.8`
- **train.py**: `--learning_rate 0.01 --epochs 100 --grad_check`
- **inference.py**: `--data_dir data --compare`

## Features

### Gradient Verification
Built-in numerical gradient checking to verify analytical gradients:
```bash
python train.py --grad_check
```

### Function Comparison
Visual comparison between learned and true functions:
```bash
python inference.py --compare
```

### Custom Metrics
- **MSE**: Mean Squared Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination

## Data Format
CSV files with columns: `X` (features), `y` (noisy targets), `y_true` (ground truth)
- True parameters: W=2.0, B=1.0
- Relationship: y = 2x + 1 + noise

## Outputs
- **Training**: Loss curve plot, pickled model parameters
- **Evaluation**: Test metrics (MSE, MAE, R²), predictions visualization
- **Comparison**: True vs learned function plots

## Dependencies
```bash
pip install numpy matplotlib pandas
```
*No PyTorch required!*

## Learning Goals
- Understand gradient computation from first principles
- Learn manual implementation of gradient descent
- Practice deriving and coding mathematical formulas
- Gain deep insight into optimization fundamentals
- Compare manual vs automatic differentiation approaches

## Educational Value
This implementation helps understand:
1. **Mathematical foundations**: How gradients are actually computed
2. **Optimization mechanics**: How parameters are updated step-by-step  
3. **Numerical stability**: Challenges in manual implementations
4. **Performance trade-offs**: Manual vs framework implementations
5. **Debugging skills**: Gradient checking and verification techniques
